"""SOTA FL baseline implementations (FedAvg, FedProx, DAdaQuant, ...)."""

from fedmaq.baselines.compression import (
    FedKDCompressionHook,
    compress_tensor,
    decompress_tensor,
)
from fedmaq.baselines.quantization import (
    DAdaQuantCompressionHook,
    FedPAQCompressionHook,
)

__all__ = [
    "FedKDCompressionHook",
    "compress_tensor",
    "decompress_tensor",
    "DAdaQuantCompressionHook",
    "FedPAQCompressionHook",
]
