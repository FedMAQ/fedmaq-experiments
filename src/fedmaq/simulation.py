"""Decorator-free Flower simulation core.

Houses the full experiment-runner logic so it is importable as
``from fedmaq.simulation import run`` and testable in-process. ``scripts/run.py``
is a thin ``@hydra.main`` wrapper around :func:`run`; tests build ``cfg`` via
Hydra's ``compose`` API and call :func:`run` directly.
"""

import logging
import random
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.clientapp import ClientApp
from flwr.common import Scalar, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from omegaconf import DictConfig, OmegaConf

from fedmaq.baselines import get_compressor_hook
from fedmaq.core.client import GenericClient, get_loss_hook
from fedmaq.core.evaluation import evaluate_fedmd_ensemble, evaluate_global_model
from fedmaq.core.models import (
    DEVICE,
    get_client_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import (
    generate_partition_indices,
    get_client_loader,
    get_server_loaders,
)
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager

logger = logging.getLogger("fedmaq")


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg: DictConfig) -> TelemetryManager:
    """Execute a single Flower simulation for the given composed config.

    Returns the :class:`TelemetryManager` so callers/tests can inspect
    ``cumulative_bytes`` and the emitted JSONL/CSV logs after the run.
    """
    # Set seed
    set_seed(cfg.seed)

    # Check GPU availability and warn if not detected
    if not torch.cuda.is_available():
        logger.warning(
            "\n"
            "========================================================================\n"
            "WARNING: GPU (CUDA) is not detected! The simulation will run on the CPU.\n"
            "This will significantly increase execution time, especially for large\n"
            "datasets/models. Please ensure CUDA drivers and PyTorch CUDA build are\n"
            "installed correctly to utilize the GPU.\n"
            "========================================================================"
        )
    else:
        logger.info(
            f"GPU (CUDA) detected. Using device: {torch.cuda.get_device_name(0)}"
        )

    # Convert Hydra config to standard Python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Running simulation with config:\n{OmegaConf.to_yaml(cfg)}")

    alg_name: str = cfg.algorithm.name
    dataset_name: str = cfg.dataset.name
    num_classes: int = int(cfg.dataset.num_classes)

    # 1. Generate partitioning (server reserve is always active)
    public_indices, client_indices_dict = generate_partition_indices(
        dataset_name=dataset_name,
        num_clients=cfg.experiment.num_clients,
        alpha=cfg.heterogeneity.alpha,
        num_public_samples=cfg.experiment.num_public_samples,
        seed=cfg.seed,
        partition=OmegaConf.select(cfg, "heterogeneity.partition", default="dirichlet"),
    )

    # 2. Setup Telemetry
    telemetry = TelemetryManager(cfg_dict)
    telemetry.init_wandb()

    # 3. Define client app components
    def client_fn(context: fl.app.Context) -> fl.client.Client:
        partition_id = context.node_config["partition-id"]

        train_loader = get_client_loader(
            dataset_name=dataset_name,
            client_id=partition_id,
            client_indices_dict=client_indices_dict,
            batch_size=cfg.experiment.batch_size,
            train=True,
        )
        public_loader, _ = get_server_loaders(
            dataset_name, public_indices, batch_size=cfg.experiment.batch_size
        )
        model = get_client_model(alg_name, dataset_name, num_classes)

        alg_cfg_dict = OmegaConf.to_container(cfg.algorithm, resolve=True)
        loss_hook = get_loss_hook(alg_name, alg_cfg_dict)
        compressor_hook = get_compressor_hook(
            alg_name,
            alg_cfg_dict,
            rng=np.random.default_rng(cfg.seed + partition_id),
            state=context.state,
        )

        return GenericClient(
            cid=str(partition_id),
            trainloader=train_loader,
            testloader=train_loader,  # Client local eval uses own partition data
            model=model,
            loss_hook=loss_hook,
            compressor_hook=compressor_hook,
            config=cfg_dict,
            public_loader=public_loader,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    # 4. Define server app components
    def server_fn(context: fl.app.Context) -> ServerAppComponents:
        initial_model = get_client_model(alg_name, dataset_name, num_classes)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

        # Server-side test loader (uses full held-out test split, not public reserve)
        _, test_loader = get_server_loaders(
            dataset_name, public_indices, batch_size=cfg.experiment.batch_size
        )

        def evaluate_fn(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: dict[str, fl.common.Scalar],
        ) -> tuple[float, dict[str, Scalar]] | None:
            device_str = OmegaConf.select(cfg, "device", default=None)
            device = torch.device(device_str) if device_str else DEVICE

            if alg_name == "fedmd":
                persistence_dir = cfg.experiment.get(
                    "persistence_dir", f".data_partitions/{alg_name}_models"
                )
                model_dir = Path(persistence_dir)
                client_paths = (
                    list(model_dir.glob("client_*.pth")) if model_dir.exists() else []
                )
                if not client_paths:
                    # Fallback to random global model if no client models are saved yet
                    eval_model = get_client_model(alg_name, dataset_name, num_classes)
                    return evaluate_global_model(
                        eval_model,
                        test_loader,
                        num_classes=num_classes,
                        device=device,
                    )
                return evaluate_fedmd_ensemble(
                    client_paths=client_paths,
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    test_loader=test_loader,
                    device=device,
                )

            # Default FL path: reconstruct a fresh model each round to avoid
            # mutable-closure issues in async simulation scenarios.
            eval_model = get_client_model(alg_name, dataset_name, num_classes)
            set_model_parameters(eval_model, parameters)
            return evaluate_global_model(
                eval_model,
                test_loader,
                num_classes=num_classes,
                device=device,
            )

        strategy = TelemetryFedAvg(
            telemetry_manager=telemetry,
            config=cfg_dict,
            client_indices_dict=client_indices_dict,
            public_indices=public_indices,
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
            "num_gpus": float(
                OmegaConf.select(cfg, "experiment.client_gpus", default=0.0)
            ),
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

    return telemetry
