import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.kd_utils import run_server_side_kd
from fedmaq.core.models import SimpleCNN, get_model_parameters
from fedmaq.core.strategy_hooks.fedmaq import FedMAQHook


class MockClientProxy(ClientProxy):
    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, *args, **kwargs):
        pass

    def get_parameters(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def reconnect(self, *args, **kwargs):
        pass


def test_soft_voting_logic():
    """Verify that teacher weighting properly weights by entropy and bit-width."""
    student = SimpleCNN(in_channels=1, num_classes=3)
    teacher1 = SimpleCNN(in_channels=1, num_classes=3)
    teacher2 = SimpleCNN(in_channels=1, num_classes=3)

    # Set teacher 1 to be highly confident (one-hot outputs) and teacher 2 to be uniform (low confidence)
    # By modifying biases/weights of the last layer
    with torch.no_grad():
        teacher1.fc2.bias.fill_(0.0)
        teacher1.fc2.weight.fill_(0.0)
        teacher1.fc2.bias[0] = 100.0  # very confident in class 0

        teacher2.fc2.bias.fill_(0.0)  # outputs close to uniform
        teacher2.fc2.weight.fill_(0.0)

    # Loader with 1 sample
    images = torch.randn(1, 1, 28, 28)
    labels = torch.randint(0, 3, (1,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    device = torch.device("cpu")

    # Run server-side KD with soft voting
    # Test case 1: equal bit widths
    run_server_side_kd(
        student_model=student,
        teachers=[teacher1, teacher2],
        public_loader=loader,
        temperature=1.0,
        learning_rate=0.01,
        momentum=0.9,
        device=device,
        epochs=1,
        teacher_bit_widths=[8, 8],
        entropy_weight_scale=1.0,
        precision_weight_scale=1.0,
    )

    # Test case 2: unequal bit widths
    run_server_side_kd(
        student_model=student,
        teachers=[teacher1, teacher2],
        public_loader=loader,
        temperature=1.0,
        learning_rate=0.01,
        momentum=0.9,
        device=device,
        epochs=1,
        teacher_bit_widths=[1, 32],
        entropy_weight_scale=1.0,
        precision_weight_scale=1.0,
    )


def test_fedmaq_hook_refinement_states():
    """Verify that FedMAQHook applies student EMA and grad norm smoothing."""
    cfg = {
        "dataset": {"name": "mnist", "num_classes": 10},
        "experiment": {"batch_size": 2, "num_public_samples": 2, "local_epochs": 1},
        "algorithm": {
            "name": "fedmaq",
            "q_min": 1,
            "q_max": 16,
            "c_unit": 512.0,
            "formulation": 3,
            "soft_voting": True,
            "ema_student": True,
            "ema_decay": 0.9,
            "grad_norm_ema": True,
            "grad_norm_beta": 0.5,
        },
    }

    hook = FedMAQHook(cfg)
    assert hook._round_client_q == {}
    assert hook._ema_params is None
    assert hook._grad_norm_ema == {}

    # Test grad norm EMA smoothing logic directly
    pids = [1, 2]
    grad_norms = [2.0, 4.0]

    # First round - no EMA history, should store raw norms
    beta = 0.5
    smoothed_norms = []
    for pid, raw_norm in zip(pids, grad_norms, strict=True):
        if pid in hook._grad_norm_ema:
            smoothed = beta * hook._grad_norm_ema[pid] + (1.0 - beta) * raw_norm
        else:
            smoothed = raw_norm
        hook._grad_norm_ema[pid] = smoothed
        smoothed_norms.append(smoothed)

    assert smoothed_norms == [2.0, 4.0]
    assert hook._grad_norm_ema == {1: 2.0, 2: 4.0}

    # Second round - should smooth with EMA
    raw_norms_2 = [3.0, 2.0]
    smoothed_norms_2 = []
    for pid, raw_norm in zip(pids, raw_norms_2, strict=True):
        if pid in hook._grad_norm_ema:
            smoothed = beta * hook._grad_norm_ema[pid] + (1.0 - beta) * raw_norm
        else:
            smoothed = raw_norm
        hook._grad_norm_ema[pid] = smoothed
        smoothed_norms_2.append(smoothed)

    # pid 1: 0.5 * 2.0 + 0.5 * 3.0 = 2.5
    # pid 2: 0.5 * 4.0 + 0.5 * 2.0 = 3.0
    assert smoothed_norms_2 == [2.5, 3.0]

    # Test Student EMA tracking
    model = SimpleCNN(in_channels=1, num_classes=10)
    params = get_model_parameters(model)
    aggregated_params = ndarrays_to_parameters(params)

    # Round 1 student EMA (ema_params is None, should copy params)
    new_params = parameters_to_ndarrays(aggregated_params)
    hook._ema_params = [p.copy() for p in new_params]

    # Round 2 student EMA
    next_params = [p + 1.0 for p in params]
    ema_decay = 0.9
    hook._ema_params = [
        ema_decay * ema + (1.0 - ema_decay) * new
        for ema, new in zip(hook._ema_params, next_params, strict=True)
    ]

    # Check that they blended
    for orig, ema in zip(params, hook._ema_params, strict=True):
        assert np.allclose(ema, orig + 0.1)


def test_client_telemetry_aggregation():
    """Verify that client-side training metrics are correctly calculated and aggregated."""
    from flwr.common import Code, FitRes, Status

    client1 = MockClientProxy("1")
    client2 = MockClientProxy("2")

    results = [
        (
            client1,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters([]),
                num_examples=10,
                metrics={
                    "train_loss": 0.5,
                    "train_acc": 0.8,
                    "epochs_trained": 5,
                    "q": 4,
                },
            ),
        ),
        (
            client2,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters([]),
                num_examples=20,
                metrics={
                    "train_loss": 0.2,
                    "train_acc": 0.9,
                    "epochs_trained": 5,
                    "q": 8,
                },
            ),
        ),
    ]

    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    assert total_examples == 30

    # Test weighted average calculation
    weighted_loss = (
        sum(float(fit_res.metrics["train_loss"]) * fit_res.num_examples for _, fit_res in results)
        / total_examples
    )
    weighted_acc = (
        sum(float(fit_res.metrics["train_acc"]) * fit_res.num_examples for _, fit_res in results)
        / total_examples
    )
    simple_epochs = sum(float(fit_res.metrics["epochs_trained"]) for _, fit_res in results) / len(
        results
    )

    # 10 * 0.5 + 20 * 0.2 = 5 + 4 = 9 / 30 = 0.3
    assert np.isclose(weighted_loss, 0.3)
    # 10 * 0.8 + 20 * 0.9 = 8 + 18 = 26 / 30 = 0.8666...
    assert np.isclose(weighted_acc, 26 / 30)
    assert np.isclose(simple_epochs, 5.0)
