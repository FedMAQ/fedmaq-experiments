"""SOTA FL baseline implementations (FedAvg, FedProx, DAdaQuant, ...)."""

from fedmaq.baselines.quantization import (
    FedPAQCompressionHook,
    DAdaQuantCompressionHook,
)
from fedmaq.baselines.compression import (
    FedKDCompressionHook,
    compress_tensor,
    decompress_tensor,
)
