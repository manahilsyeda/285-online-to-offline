from pathlib import Path

import modal

from scripts.train_offline_online import main, setup_arguments


APP_NAME = "offline-to-online-project"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096  # MB
volume = modal.Volume.from_name("offline-to-online-project-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()
# Datasets are downloaded at runtime by ogbench (~3s); no need to bake them
# into the image (which would be invalidated on every dep change).
# Copy .netrc for wandb logging.
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
# Copy the current directory.
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 12, env=env, image=image, gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY)
def offline_to_online_modal_remote(*args: str) -> None:
    import os
    # Run from the volume root so relative outputs (exp/<run_group>/...) land on the volume.
    os.chdir(VOLUME_PATH)
    parsed_args = setup_arguments(args)
    if parsed_args.njobs is not None and len(parsed_args.job_specs) > 0:
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=parsed_args.job_specs, njobs=parsed_args.njobs)
    else:
        main(parsed_args)
    volume.commit()


@app.local_entrypoint()
def entrypoint(*args: str) -> None:
    """Forward CLI args to the remote training function.

    Example:
        uv run modal run src/scripts/modal_run.py -- \\
            --run_group=s1_fql \\
            --base_config=fql \\
            --env_name=cube-single-play-singletask-task1-v0 \\
            --seed=42 \\
            --alpha=30 \\
            --offline_training_steps=500000 \\
            --online_training_steps=100000
    """
    offline_to_online_modal_remote.remote(*args)
