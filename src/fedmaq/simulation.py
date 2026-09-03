"""Flower simulation core."""

import logging
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import flwr as fl
import numpy as np
import torch
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecordValues, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from omegaconf import DictConfig, OmegaConf

from fedmaq.baselines import get_compressor_hook
from fedmaq.core.checkpoint import write_final_global_model
from fedmaq.core.client import GenericClient, get_loss_hook
from fedmaq.core.client_manager import SeededPartitionClientManager
from fedmaq.core.evaluation import evaluate_fedmd_ensemble, evaluate_global_model
from fedmaq.core.manifest import write_run_manifest
from fedmaq.core.models import (
    DEVICE,
    get_client_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import (
    CACHE_DIR,
    generate_partition_indices,
    get_client_loader,
    get_partition_cache_info,
    get_server_loaders,
)
from fedmaq.core.protocol import register_protocol
from fedmaq.core.randomness import derive_numpy_rng, derive_seed
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager

logger = logging.getLogger("fedmaq")

# Ray appends its plasma-store socket path beneath the temp directory.
RAY_SOCKET_PATH_BUDGET = 107
RAY_SESSION_SUFFIX_BUDGET = 60


def build_ray_init_args(cfg: DictConfig) -> dict[str, ConfigRecordValues]:
    """Translate the ``ray:`` config block into ``ray.init()`` keyword arguments.

    Returns ``{}`` when nothing is configured, and the caller then omits
    ``init_args`` from ``backend_config`` entirely, which reproduces Flower's stock
    behaviour exactly. Both settings are host properties and both ship null, so the
    local development rig is unaffected by their existence.

    The pass-through is real but undocumented, so it is worth recording why this
    works. ``run_simulation`` merges ``init_args`` with its own ``logging_level`` /
    ``log_to_driver`` defaults and JSON-serialises the whole ``backend_config``
    (``flwr/simulation/run_simulation.py``), then ``RayBackend.init_ray`` copies
    every key of it verbatim into the ``ray.init()`` call
    (``raybackend.py:107-122``). There is no schema and no allowlist on that path,
    which is what lets ``object_store_memory`` and ``_temp_dir`` through. Flower's
    *other* ``init_args`` surface, the ``flwr run`` TOML federation config, is a
    closed four-field dataclass (``num_cpus``, ``num_gpus``, ``logging_level``,
    ``log_to_driver``); this thesis does not use that surface, and neither of the
    two settings below could be expressed through it. The only real constraint here
    is JSON-serialisability, which ``int`` and ``str`` satisfy.

    ``_temp_dir`` is a private ``ray.init`` argument, absorbed by its ``**kwargs``
    and popped in ``ray/_private/worker.py``. Ray requires it absolute and gives a
    poor error when it is not, so reject a relative path here instead.
    """
    init_args: dict[str, ConfigRecordValues] = {}

    object_store_gb = OmegaConf.select(cfg, "ray.object_store_gb", default=None)
    if object_store_gb is not None:
        init_args["object_store_memory"] = int(float(object_store_gb) * 1024**3)

    temp_dir = OmegaConf.select(cfg, "ray.temp_dir", default=None)
    if temp_dir:
        # ``os.path.expanduser`` rather than ``Path.expanduser``: this value is a path
        # on the host that runs Ray, and passing it through ``Path`` on Windows
        # rewrites '/tmp/ray-cjb' into '\tmp\ray-cjb'. Keep it a string and let Ray
        # receive exactly what was configured.
        resolved = os.path.expanduser(str(temp_dir))
        # Absoluteness is likewise checked against both path flavours rather than the
        # local one, because the hub is Linux while the config is edited and dry-run
        # from Windows, where ``PurePath('/tmp/ray-cjb').is_absolute()`` is False for
        # want of a drive letter. A platform-native check would reject the correct hub
        # value on the development rig.
        if not (PurePosixPath(resolved).is_absolute() or PureWindowsPath(resolved).is_absolute()):
            raise ValueError(
                f"ray.temp_dir must be an absolute path, got '{temp_dir}'. Ray builds "
                "its session directory beneath it and requires an absolute root."
            )
        if len(resolved) + RAY_SESSION_SUFFIX_BUDGET > RAY_SOCKET_PATH_BUDGET:
            logger.warning(
                f"ray.temp_dir '{resolved}' is {len(resolved)} characters. Ray appends "
                f"about {RAY_SESSION_SUFFIX_BUDGET} more to reach its plasma-store "
                f"socket, against a ~{RAY_SOCKET_PATH_BUDGET}-character limit. Prefer "
                f"a short path such as /tmp/ray-<user>."
            )
        init_args["_temp_dir"] = resolved

    return init_args


def set_seed(seed: int, strict: bool = True) -> None:
    """Set global seeds and (optionally) enforce deterministic kernels.

    Reproducibility spine for the paired confirmatory grid: seeds
    Python/NumPy/torch (+CUDA), pins cuDNN to deterministic mode, and requests
    deterministic algorithms. ``strict=True`` raises when an op lacks a
    deterministic implementation (surfacing it) rather than silently falling
    back to a non-deterministic kernel.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first CUDA context is
    created for deterministic cuBLAS GEMMs; we set it here (idempotently) so it
    is in place before any CUDA use and is inherited by Ray workers spawned
    afterwards. Note this function is also called inside ``client_fn`` to reseed
    each Ray worker process, since the driver's seed does not propagate across
    process boundaries.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    configure_torch_determinism(seed, strict=strict)


def configure_torch_determinism(seed: int, strict: bool = True) -> None:
    """Set torch RNG + deterministic-kernel flags, WITHOUT touching the global
    ``random``/``numpy`` RNGs.

    Ray client actors run in separate processes that do not inherit the driver's
    torch determinism settings, so training there uses non-deterministic cuDNN
    kernels unless re-pinned per worker. We call this inside ``client_fn`` to fix
    that — but deliberately leave Python's ``random`` alone, because Flower's
    ``ClientManager.sample()`` draws from it and reseeding in a worker would make
    client sampling timing-dependent across Ray restarts.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


@dataclass(frozen=True)
class SimulationAssembly:
    """The small set of Flower inputs required to execute one simulation."""

    client_app: Any
    server_app: Any
    num_supernodes: int
    backend_config: dict[str, dict[str, ConfigRecordValues]]


@dataclass
class SimulationRun:
    """Execute an assembled simulation and own its finalization lifecycle.

    The runner is injectable so this seam can exercise success, failure, and
    timeout cleanup without starting Flower or Ray. Finalization order is
    deliberate: telemetry is closed before the run-scoped persistence tree,
    matching the historical entry point and ensuring all final records can be
    written before temporary state disappears.
    """

    assembly: SimulationAssembly
    telemetry: TelemetryManager
    persistence_scope: Any | None = None
    runner: Callable[..., Any] = run_simulation

    def execute(self) -> TelemetryManager:
        """Run Flower, finalize owned resources, and re-raise run failures."""
        primary_error: BaseException | None = None
        try:
            logger.info("Starting Flower Simulation...")
            self.runner(
                server_app=self.assembly.server_app,
                client_app=self.assembly.client_app,
                num_supernodes=self.assembly.num_supernodes,
                backend_config=self.assembly.backend_config,
            )
        except BaseException as exc:
            primary_error = exc

        try:
            self.telemetry.finish()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                logger.error("Telemetry finalization failed after simulation failure: %s", exc)
        try:
            if self.persistence_scope is not None:
                self.persistence_scope.cleanup()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                logger.error("Persistence cleanup failed after simulation failure: %s", exc)

        if primary_error is not None:
            raise primary_error
        return self.telemetry


@dataclass(frozen=True)
class SimulationPlan:
    """Resolved inputs shared by the lazily constructed Flower applications."""

    config: dict[str, Any]
    algorithm_config: dict[str, Any]
    algorithm_name: str
    dataset_name: str
    num_classes: int
    seed: int
    strict_determinism: bool
    public_indices: Any
    val_indices: Any
    client_indices_dict: Any
    client_config: dict[str, Any]
    batch_size: int
    client_fraction: float
    num_clients: int
    total_rounds: int
    device: str | None
    backend_config: dict[str, dict[str, ConfigRecordValues]]
    split: str = "val"
    loader_used: str = "val"
    partition_cache_info: dict[str, Any] | None = None


@dataclass(frozen=True)
class SimulationApplications:
    """Build both Flower applications from one resolved simulation plan.

    The application callbacks are ordinary functions that capture only the
    plan and, for the server callback, the live telemetry manager. The builder
    itself and its driver-owned persistence scope never cross the Flower/Ray
    callback seam.
    """

    plan: SimulationPlan
    telemetry: TelemetryManager

    def build(self) -> SimulationAssembly:
        """Construct lazy client/server applications and their backend inputs."""
        plan = self.plan
        telemetry = self.telemetry

        def client_fn(context: fl.app.Context) -> fl.client.Client:
            partition_id = int(context.node_config["partition-id"])
            client_seed = derive_seed("client", plan.seed, partition_id, 1)
            configure_torch_determinism(client_seed, strict=plan.strict_determinism)
            train_loader = get_client_loader(
                dataset_name=plan.dataset_name,
                client_id=partition_id,
                client_indices_dict=plan.client_indices_dict,
                seed=derive_seed("training", plan.seed, partition_id, 1),
                batch_size=plan.batch_size,
                train=True,
            )

            def trainloader_for_round(server_round: int):
                return get_client_loader(
                    dataset_name=plan.dataset_name,
                    client_id=partition_id,
                    client_indices_dict=plan.client_indices_dict,
                    seed=derive_seed("training", plan.seed, partition_id, server_round),
                    batch_size=plan.batch_size,
                    train=True,
                )

            public_loader, _, _ = get_server_loaders(
                plan.dataset_name,
                plan.public_indices,
                batch_size=plan.batch_size,
            )
            model = get_client_model(plan.algorithm_name, plan.dataset_name, plan.num_classes)
            loss_hook = get_loss_hook(plan.algorithm_name, plan.algorithm_config)
            compressor_hook = get_compressor_hook(
                plan.algorithm_name,
                plan.algorithm_config,
                rng=derive_numpy_rng("compression", plan.seed, partition_id, 1),
                state=context.state,
            )
            return GenericClient(
                cid=str(partition_id),
                trainloader=train_loader,
                testloader=train_loader,
                model=model,
                loss_hook=loss_hook,
                compressor_hook=compressor_hook,
                config=plan.client_config,
                public_loader=public_loader,
                state=context.state,
                trainloader_factory=trainloader_for_round,
            ).to_client()

        def evaluate_fn(
            server_round: int,
            parameters: fl.common.NDArrays,
            _config: dict[str, fl.common.Scalar],
            eval_loader: Any,
        ) -> tuple[float, dict[str, float]] | None:
            device = torch.device(plan.device) if plan.device else DEVICE
            if plan.algorithm_name == "fedmd":
                persistence_dir = plan.config.get("experiment", {}).get(
                    "persistence_dir", f".data_partitions/{plan.algorithm_name}_models"
                )
                model_dir = Path(persistence_dir)
                client_paths = list(model_dir.glob("client_*.pth")) if model_dir.exists() else []
                if not client_paths:
                    eval_model = get_client_model(
                        plan.algorithm_name, plan.dataset_name, plan.num_classes
                    )
                    return evaluate_global_model(
                        eval_model,
                        eval_loader,
                        num_classes=plan.num_classes,
                        device=device,
                    )
                return evaluate_fedmd_ensemble(
                    client_paths=client_paths,
                    dataset_name=plan.dataset_name,
                    num_classes=plan.num_classes,
                    test_loader=eval_loader,
                    device=device,
                )

            eval_model = get_client_model(plan.algorithm_name, plan.dataset_name, plan.num_classes)
            set_model_parameters(eval_model, parameters)
            write_final_global_model(
                eval_model,
                telemetry.log_dir,
                server_round,
                plan.total_rounds,
            )
            return evaluate_global_model(
                eval_model,
                eval_loader,
                num_classes=plan.num_classes,
                device=device,
            )

        def server_fn(_context: fl.app.Context) -> ServerAppComponents:
            initial_model = get_client_model(
                plan.algorithm_name, plan.dataset_name, plan.num_classes
            )
            initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))
            public_loader, val_loader, test_loader = get_server_loaders(
                plan.dataset_name,
                plan.public_indices,
                validation_indices=plan.val_indices,
                batch_size=plan.batch_size,
            )
            eval_loader = val_loader if plan.split == "val" else test_loader
            strategy = TelemetryFedAvg(
                telemetry_manager=telemetry,
                config=plan.config,
                client_indices_dict=plan.client_indices_dict,
                public_indices=plan.public_indices,
                split=plan.split,
                fraction_fit=plan.client_fraction,
                fraction_evaluate=0.0,
                min_fit_clients=max(1, int(plan.num_clients * plan.client_fraction)),
                min_available_clients=plan.num_clients,
                evaluate_fn=lambda server_round, parameters, config: evaluate_fn(
                    server_round, parameters, config, eval_loader
                ),
                initial_parameters=initial_parameters,
            )
            client_manager = SeededPartitionClientManager(
                seed=plan.seed,
                num_clients=plan.num_clients,
            )
            return ServerAppComponents(
                strategy=strategy,
                config=ServerConfig(num_rounds=plan.total_rounds),
                client_manager=client_manager,
            )

        return SimulationAssembly(
            client_app=ClientApp(client_fn=client_fn),
            server_app=ServerApp(server_fn=server_fn),
            num_supernodes=plan.num_clients,
            backend_config=plan.backend_config,
        )


class SimulationBuilder:
    """Prepare deterministic state and build the two Flower applications.

    ``build`` is the sole construction seam. It resolves partition and runtime
    inputs, then delegates client/server behavior to focused factories while
    exposing only a ready-to-execute :class:`SimulationRun` to the caller.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        runner: Callable[..., Any] = run_simulation,
    ) -> None:
        self.cfg = cfg
        self.runner = runner
        self.telemetry: TelemetryManager | None = None
        self.persistence_scope: tempfile.TemporaryDirectory[str] | None = None
        self.cfg_dict: dict[str, Any] = {}
        self.client_config: dict[str, Any] = {}
        self.strict_determinism = True

    def build(self) -> SimulationRun:
        """Prepare one run and return its behavior-rich execution seam."""
        try:
            self._prepare()
            if self.telemetry is None:
                raise RuntimeError("simulation preparation did not create telemetry")
            # The Flower applications capture only the resolved plan and server
            # telemetry. Keep the live TemporaryDirectory on the driver, where
            # SimulationRun owns its cleanup, rather than serializing it to workers.
            persistence_scope = self.persistence_scope
            self.persistence_scope = None
            try:
                plan = self._plan()
                applications = SimulationApplications(plan, self.telemetry).build()
                return SimulationRun(
                    assembly=applications,
                    telemetry=self.telemetry,
                    persistence_scope=persistence_scope,
                    runner=self.runner,
                )
            except BaseException:
                self.persistence_scope = persistence_scope
                raise
        except BaseException:
            self._finish_resources()
            raise

    def _prepare(self) -> None:
        strict_determinism = bool(
            OmegaConf.select(self.cfg, "experiment.strict_determinism", default=True)
        )
        self.strict_determinism = strict_determinism
        set_seed(self.cfg.seed, strict=strict_determinism)

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
            logger.info(f"GPU (CUDA) detected. Using device: {torch.cuda.get_device_name(0)}")

        self.cfg_dict = cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True))
        logger.info(f"Running simulation with config:\n{OmegaConf.to_yaml(self.cfg)}")

        self.alg_name: str = self.cfg.algorithm.name
        self.dataset_name: str = self.cfg.dataset.name
        self.num_classes: int = int(self.cfg.dataset.num_classes)
        partition = str(OmegaConf.select(self.cfg, "heterogeneity.partition", default="dirichlet"))
        self.public_indices, self.val_indices, self.client_indices_dict = (
            generate_partition_indices(
                dataset_name=self.dataset_name,
                num_clients=self.cfg.experiment.num_clients,
                alpha=self.cfg.heterogeneity.alpha,
                num_public_samples=self.cfg.experiment.num_public_samples,
                seed=self.cfg.seed,
                partition=partition,
            )
        )

        if partition == "writer":
            cache_name = (
                f"{self.dataset_name.lower()}_clients_{self.cfg.experiment.num_clients}_"
                f"writer_pub_{self.cfg.experiment.num_public_samples}_seed_{self.cfg.seed}.json"
            )
        else:
            cache_name = (
                f"{self.dataset_name.lower()}_clients_{self.cfg.experiment.num_clients}_"
                f"alpha_{self.cfg.heterogeneity.alpha}_pub_{self.cfg.experiment.num_public_samples}_seed_{self.cfg.seed}.json"
            )
        cache_file = CACHE_DIR / cache_name
        self.partition_cache_info = get_partition_cache_info(cache_file)

        protocol_reg = register_protocol(self.cfg_dict, {"commit": None, "dirty": False})
        split_override = OmegaConf.select(self.cfg, "split", default=None)
        self.split = str(split_override if split_override is not None else protocol_reg.split)
        self.loader_used = "val" if self.split == "val" else "test"

        self.telemetry = TelemetryManager(self.cfg_dict)
        self.telemetry.init_wandb()
        write_run_manifest(
            self.cfg_dict,
            self.telemetry.log_dir,
            partition_cache=self.partition_cache_info,
            loader_used=self.loader_used,
        )

        self.client_config = self.cfg_dict
        if self.alg_name == "fedkd":
            state_root = Path(".data_partitions/fedkd_runs")
            state_root.mkdir(parents=True, exist_ok=True)
            self.persistence_scope = tempfile.TemporaryDirectory(prefix="run-", dir=state_root)
            self.client_config = {**self.cfg_dict, "_persistence_dir": self.persistence_scope.name}

    def _plan(self) -> SimulationPlan:
        return SimulationPlan(
            config=self.cfg_dict,
            algorithm_config=cast(
                dict[str, Any], OmegaConf.to_container(self.cfg.algorithm, resolve=True)
            ),
            algorithm_name=self.alg_name,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            seed=int(self.cfg.seed),
            strict_determinism=self.strict_determinism,
            public_indices=self.public_indices,
            val_indices=self.val_indices,
            client_indices_dict=self.client_indices_dict,
            client_config=self.client_config,
            batch_size=int(self.cfg.experiment.batch_size),
            client_fraction=float(self.cfg.experiment.client_fraction),
            num_clients=int(self.cfg.experiment.num_clients),
            total_rounds=int(self.cfg.experiment.total_rounds),
            device=OmegaConf.select(self.cfg, "device", default=None),
            backend_config=self._backend_config(),
            split=self.split,
            loader_used=self.loader_used,
            partition_cache_info=self.partition_cache_info,
        )

    def _backend_config(self) -> dict[str, dict[str, ConfigRecordValues]]:
        backend_config: dict[str, dict[str, ConfigRecordValues]] = {
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": float(
                    OmegaConf.select(self.cfg, "experiment.client_gpus", default=0.0)
                ),
            }
        }
        init_args = build_ray_init_args(self.cfg)
        if init_args:
            logger.info(f"Ray init_args: {init_args}")
            backend_config["init_args"] = init_args
        return backend_config

    def _finish_resources(self) -> None:
        if self.telemetry is not None:
            try:
                self.telemetry.finish()
            finally:
                if self.persistence_scope is not None:
                    self.persistence_scope.cleanup()


def run(
    cfg: DictConfig,
    *,
    simulation_runner: Callable[..., Any] = run_simulation,
) -> TelemetryManager:
    """Execute one composed config through the simulation lifecycle seam."""
    return SimulationBuilder(cfg, runner=simulation_runner).build().execute()
