"""Resume Priority 1 Pass 2 explore sweep from run index 1 (0-indexed).

Run 0 (ema_decay=0.3) completed cleanly in
multirun/2026-07-18/18-32-26-pass2-explore-mobilenetv2/. Run 1
(ema_decay=0.5) died at round ~30/50 (Ray ActorDiedError, Windows Ray
instability class per flower-patterns.md); the background task was then
killed before an automatic retry ran, so runs 2-13 never started. This
resumes into the same output_dir_base so results.md/comments.md can treat
it as one campaign.
"""

from run_pass2_explore import (
    build_ema_runs,
    build_grad_norm_runs,
    build_kd_reg_runs,
    run_single,
    kill_ray_processes,
)

OUTPUT_DIR_BASE = "multirun/2026-07-18/18-32-26-pass2-explore-mobilenetv2"
RESUME_FROM_INDEX = 1


def main():
    runs = build_ema_runs() + build_grad_norm_runs() + build_kd_reg_runs()
    for i, run in enumerate(runs):
        run["index"] = i

    remaining = runs[RESUME_FROM_INDEX:]
    total = len(runs)

    print("==================================================")
    print(f"Resuming Pass 2 Explore at index {RESUME_FROM_INDEX}: {OUTPUT_DIR_BASE}")
    print(f"Remaining runs: {len(remaining)} of {total}")
    print("==================================================")

    for run in remaining:
        run_single(run, run["index"], total, OUTPUT_DIR_BASE)

    print("\nFinal clean up of Ray processes...")
    kill_ray_processes()
    print("Done.")


if __name__ == "__main__":
    main()
