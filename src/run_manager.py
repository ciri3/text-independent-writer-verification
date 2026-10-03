from pathlib import Path


def create_run_dir(base_dir="outputs"):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_number = 1

    while (base_dir / f"run_{run_number:03d}").exists():
        run_number += 1

    run_dir = base_dir / f"run_{run_number:03d}"
    run_dir.mkdir()

    return run_dir