"""Resume Priority 1 Pass 1 soft-voting sweep from run index 11 (0-indexed).

Runs 0-10 of ``run_soft_voting_explore.py`` completed cleanly in
multirun/2026-07-18/03-30-59-soft-voting-explore-mobilenetv2/. Run 11
(ew=2.0, pw=1.0) died at round 34/50; runs 12-17 never started. This
resumes into the same output_dir_base so results.md/comments.md can treat
it as one campaign.
"""

from run_soft_voting_explore import (
    build_ablation_runs,
    build_sweep_runs,
    run_single,
    kill_ray_processes,
)

OUTPUT_DIR_BASE = "multirun/2026-07-18/03-30-59-soft-voting-explore-mobilenetv2"
RESUME_FROM_INDEX = 11


def main():
    runs = build_ablation_runs() + build_sweep_runs()
    for i, run in enumerate(runs):
        run["index"] = i

    remaining = runs[RESUME_FROM_INDEX:]
    total = len(runs)

    print("==================================================")
    print(f"Resuming Soft-Voting Explore at index {RESUME_FROM_INDEX}: {OUTPUT_DIR_BASE}")
    print(f"Remaining runs: {len(remaining)} of {total}")
    print("==================================================")

    for run in remaining:
        run_single(run, run["index"], total, OUTPUT_DIR_BASE)

    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
