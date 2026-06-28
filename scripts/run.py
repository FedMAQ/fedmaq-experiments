"""Main experiment runner using Hydra and Flower simulation."""

import logging
import os
import random
from typing import Dict, Optional, Tuple
import flwr as fl
from flwr.common import Scalar, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.clientapp import ClientApp
from flwr.simulation import run_simulation
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from fedmaq.core.client import (
    CompressionHook,
    FedProxLossHook,
    GenericClient,
    LossHook,
)
from fedmaq.core.models import (
    get_model,
    get_model_parameters,
    set_model_parameters,
    SimpleCNN,
    TinyCNN,
)
from fedmaq.core.partitioning import (
    generate_partition_indices,
    get_client_loader,
    get_server_loaders,
)
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedmaq")


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Set seed
    set_seed(cfg.seed)

    # Convert Hydra config to standard Python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Running simulation with config:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Generate partitioning (Option A - Server reserve is always active)
    public_indices, client_indices_dict = generate_partition_indices(
        dataset_name=cfg.dataset.name,
        num_clients=cfg.experiment.num_clients,
        alpha=cfg.heterogeneity.alpha,
        num_public_samples=cfg.experiment.num_public_samples,
        seed=cfg.seed,
    )

    # 2. Setup Telemetry
    telemetry = TelemetryManager(cfg_dict)
    telemetry.init_wandb()

    # 3. Define client app components
    def client_fn(context: fl.app.Context) -> fl.client.Client:
        # Retrieve partition ID
        partition_id = context.node_config["partition-id"]

        # Get client dataset loader
        train_loader = get_client_loader(
            dataset_name=cfg.dataset.name,
            client_id=partition_id,
            client_indices_dict=client_indices_dict,
            batch_size=cfg.experiment.batch_size,
            train=True,
        )

        # Get public dataset loader for distillation
        public_loader, _ = get_server_loaders(
            cfg.dataset.name, public_indices, batch_size=cfg.experiment.batch_size
        )

        # Get local model
        if cfg.algorithm.name == "fedkd":
            if "cifar" in cfg.dataset.name.lower():
                model = SimpleCNN(in_channels=3, num_classes=cfg.dataset.num_classes)
            else:
                model = TinyCNN(in_channels=1, num_classes=cfg.dataset.num_classes)
        else:
            model = get_model(cfg.dataset.name, cfg.dataset.num_classes)

        # Determine loss hook based on algorithm
        loss_hook = LossHook()
        if cfg.algorithm.name == "fedprox":
            loss_hook = FedProxLossHook(mu=cfg.algorithm.mu)

        # Determine compressor hook based on algorithm
        compressor_hook = CompressionHook()
        if cfg.algorithm.name == "fedpaq":
            from fedmaq.baselines.quantization import FedPAQCompressionHook

            compressor_hook = FedPAQCompressionHook(q=cfg.algorithm.q)
        elif cfg.algorithm.name == "dadaquant":
            from fedmaq.baselines.quantization import DAdaQuantCompressionHook

            compressor_hook = DAdaQuantCompressionHook(q=int(cfg.algorithm.q_min))
        elif cfg.algorithm.name == "fedkd":
            from fedmaq.baselines.compression import FedKDCompressionHook

            compressor_hook = FedKDCompressionHook(energy=float(cfg.algorithm.tmin))

        return GenericClient(
            cid=str(partition_id),
            trainloader=train_loader,
            testloader=train_loader,  # Client local evaluation evaluates on own data
            model=model,
            loss_hook=loss_hook,
            compressor_hook=compressor_hook,
            config=cfg_dict,
            public_loader=public_loader,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    # 4. Define server app components
    def server_fn(context: fl.app.Context) -> ServerAppComponents:
        # Instantiate global model for initialization
        if cfg.algorithm.name == "fedkd":
            if "cifar" in cfg.dataset.name.lower():
                global_model = SimpleCNN(
                    in_channels=3, num_classes=cfg.dataset.num_classes
                )
            else:
                global_model = TinyCNN(
                    in_channels=1, num_classes=cfg.dataset.num_classes
                )
        else:
            global_model = get_model(cfg.dataset.name, cfg.dataset.num_classes)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))

        # Setup server-side evaluation test loader
        _, test_loader = get_server_loaders(
            cfg.dataset.name, public_indices, batch_size=cfg.experiment.batch_size
        )

        # Global evaluation function run on the server
        def evaluate_fn(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, Scalar],
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            criterion = nn.CrossEntropyLoss()

            if cfg.algorithm.name == "fedmd":
                from pathlib import Path

                persistence_dir = cfg.experiment.get(
                    "persistence_dir", ".data_partitions/fedmd_models"
                )
                model_dir = Path(persistence_dir)
                client_paths = (
                    list(model_dir.glob("client_*.pth")) if model_dir.exists() else []
                )

                if not client_paths:
                    # Fallback to random global model if no client models are saved yet
                    global_model.eval()
                    global_model.to(device)
                    loss_sum = 0.0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = global_model(images)
                            loss_sum += criterion(outputs, labels).item() * len(labels)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    loss = loss_sum / total if total > 0 else 0.0
                    accuracy = correct / total if total > 0 else 0.0
                    return loss, {"accuracy": accuracy}

                # Evaluate each client model and average results
                total_loss = 0.0
                total_accuracy = 0.0
                num_eval_clients = 0

                for path in client_paths:
                    client_model = get_model(cfg.dataset.name, cfg.dataset.num_classes)
                    try:
                        client_model.load_state_dict(
                            torch.load(path, map_location=device)
                        )
                        client_model.eval()
                        client_model.to(device)

                        loss_sum = 0.0
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for images, labels in test_loader:
                                images, labels = images.to(device), labels.to(device)
                                outputs = client_model(images)
                                loss_sum += criterion(outputs, labels).item() * len(
                                    labels
                                )
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        client_loss = loss_sum / total if total > 0 else 0.0
                        client_acc = correct / total if total > 0 else 0.0
                        total_loss += client_loss
                        total_accuracy += client_acc
                        num_eval_clients += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to load or evaluate model at {path}: {e}"
                        )

                if num_eval_clients > 0:
                    loss = total_loss / num_eval_clients
                    accuracy = total_accuracy / num_eval_clients
                else:
                    loss = 0.0
                    accuracy = 0.0

                return loss, {"accuracy": accuracy}

            # Default FL path (FedAvg, FedProx, etc.)
            # Load weights
            set_model_parameters(global_model, parameters)

            global_model.eval()
            global_model.to(device)

            loss_sum = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    loss_sum += criterion(outputs, labels).item() * len(labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            loss = loss_sum / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0

            return loss, {"accuracy": accuracy}

        # Setup custom strategy
        strategy = TelemetryFedAvg(
            telemetry_manager=telemetry,
            config=cfg_dict,
            client_indices_dict=client_indices_dict,
            fraction_fit=cfg.experiment.client_fraction,
            fraction_evaluate=0.0,  # Disable client-side evaluation overhead
            min_fit_clients=max(
                1, int(cfg.experiment.num_clients * cfg.experiment.client_fraction)
            ),
            min_available_clients=cfg.experiment.num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )

        server_config = ServerConfig(num_rounds=cfg.experiment.total_rounds)

        return ServerAppComponents(strategy=strategy, config=server_config)

    server_app = ServerApp(server_fn=server_fn)

    # 5. Run FL simulation
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0,
        }
    }

    logger.info("Starting Flower Simulation...")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.experiment.num_clients,
        backend_config=backend_config,
    )

    # Finish telemetry
    telemetry.finish()


if __name__ == "__main__":
    main()
