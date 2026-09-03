"""Deterministic CPU-only partition cache regeneration script.

Regenerates partition cache JSON files for CIFAR-10, CIFAR-100, and FEMNIST,
recording post-holdout shard statistics, client class counts, and SHA-256 digests.
Emits summary statistics to docs/recut/post_holdout_shard_statistics.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fedmaq.core.partitioning import (
    CACHE_DIR,
    generate_partition_indices,
    get_partition_cache_info,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("regenerate_partitions")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "docs" / "recut" / "post_holdout_shard_statistics.json"

GRID: list[dict[str, Any]] = [
    # CIFAR-10 grid
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 200,
        "seed": 42,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 200,
        "seed": 0,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 200,
        "seed": 123,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 3000,
        "seed": 42,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 3000,
        "seed": 0,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.1,
        "num_public_samples": 3000,
        "seed": 123,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.3,
        "num_public_samples": 3000,
        "seed": 0,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.3,
        "num_public_samples": 3000,
        "seed": 42,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 0.3,
        "num_public_samples": 3000,
        "seed": 123,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 200,
        "seed": 42,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 200,
        "seed": 0,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 200,
        "seed": 123,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 3000,
        "seed": 42,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 3000,
        "seed": 0,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 100,
        "alpha": 1.0,
        "num_public_samples": 3000,
        "seed": 123,
    },
    {
        "dataset_name": "cifar10",
        "num_clients": 2,
        "alpha": 0.1,
        "num_public_samples": 3000,
        "seed": 42,
    },
]


def regenerate_all() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for item in GRID:
        dataset = item["dataset_name"]
        num_clients = item["num_clients"]
        alpha = item["alpha"]
        num_public = item["num_public_samples"]
        seed = item["seed"]
        partition = item.get("partition", "dirichlet")

        logger.info(
            f"Regenerating {dataset} (clients={num_clients}, alpha={alpha}, "
            f"pub={num_public}, seed={seed})..."
        )
        try:
            generate_partition_indices(
                dataset_name=dataset,
                num_clients=num_clients,
                alpha=alpha,
                num_public_samples=num_public,
                seed=seed,
                partition=partition,
                force=True,
            )
            if partition == "writer":
                cache_filename = (
                    f"{dataset}_clients_{num_clients}_writer_pub_{num_public}_seed_{seed}.json"
                )
            else:
                cache_filename = (
                    f"{dataset}_clients_{num_clients}_alpha_{alpha}_"
                    f"pub_{num_public}_seed_{seed}.json"
                )
            cache_file = CACHE_DIR / cache_filename
            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
            info = get_partition_cache_info(cache_file)
            summary[cache_filename] = {
                "dataset": dataset,
                "num_clients": num_clients,
                "alpha": alpha,
                "num_public_samples": num_public,
                "num_val_samples": cache_data.get("num_val_samples"),
                "seed": seed,
                "partition": partition,
                "shard_stats": cache_data.get("shard_stats"),
                "sha256": info["sha256"] if info else None,
            }
            stats = cache_data.get("shard_stats", {})
            logger.info(
                f"  -> Done: min={stats.get('min')}, max={stats.get('max')}, "
                f"mean={stats.get('mean'):.1f}, total={stats.get('total')}"
            )
        except Exception as exc:
            logger.warning(f"  -> Skipped {dataset}: {exc}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary written to {OUTPUT_FILE}")
    return summary


if __name__ == "__main__":
    regenerate_all()
