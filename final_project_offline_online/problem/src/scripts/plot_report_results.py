import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SUCCESS_COL = "eval/success_rate"
STEP_COL = "step"
ONLINE_START_STEP = 500_000

DISPLAY_NAMES = {
    "sacbc": "SAC+BC",
    "fql": "FQL",
    "ifql": "IFQL",
    "dsrl": "DSRL",
    "qsm": "QSM",
    "yours": "Ours",
    "s2_offline": "Offline data",
    "s2_wsrl": "WSRL",
}

COLORS = {
    "sacbc": "#4C78A8",
    "fql": "#F58518",
    "ifql": "#54A24B",
    "dsrl": "#E45756",
    "qsm": "#B279A2",
    "yours": "#72B7B2",
}


def read_eval_csv(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if STEP_COL not in row or SUCCESS_COL not in row:
                continue
            rows.append((int(float(row[STEP_COL])), float(row[SUCCESS_COL])))
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def short_env(env_name: str) -> str:
    if "cube-single" in env_name:
        return "cube-single"
    if "cube-double" in env_name:
        return "cube-double"
    if "antsoccer" in env_name:
        return "antsoccer-arena"
    if "humanoidmaze" in env_name:
        return "humanoidmaze"
    if "puzzle" in env_name:
        return "puzzle-4x4"
    return env_name.replace("-play-singletask-task1-v0", "").replace("-navigate-singletask-task1-v0", "")


def infer_seed(name: str) -> int | None:
    match = re.search(r"sd(?P<seed>\d+)_", name)
    return int(match.group("seed")) if match else None


def infer_agent(name: str) -> str | None:
    for agent in ("sacbc", "ifql", "dsrl", "qsm", "fql", "yours"):
        if f"_{agent}_" in name or name.startswith(agent):
            return agent
    return None


def run_from_dir(run_dir: Path) -> dict[str, Any] | None:
    eval_path = run_dir / "eval.csv"
    flags_path = run_dir / "flags.json"
    if not eval_path.is_file():
        return None

    rows = read_eval_csv(eval_path)
    if not rows:
        return None

    flags = read_json(flags_path) if flags_path.is_file() else {}
    flags = flags or {}
    agent = flags.get("agent") or infer_agent(run_dir.name)
    if agent is None:
        return None

    seed = flags.get("seed")
    if seed is None:
        seed = infer_seed(run_dir.name)
    if seed is None:
        seed = -1

    agent_kwargs = flags.get("agent_kwargs", {})
    run_group = flags.get("run_group")
    if run_group is None:
        parts = run_dir.parts
        run_group = next((part for part in reversed(parts) if part.startswith(("s1_", "s2_", "s3_"))), "unknown")

    env_name = flags.get("env_name") or run_dir.name
    alpha = agent_kwargs.get("alpha")

    return {
        "agent": agent,
        "run_group": run_group,
        "env_name": env_name,
        "env_short": short_env(env_name),
        "seed": int(seed),
        "alpha": alpha,
        "expectile": agent_kwargs.get("expectile"),
        "noise_scale": agent_kwargs.get("noise_scale"),
        "inv_temp": agent_kwargs.get("inv_temp"),
        "path": run_dir,
        "rows": rows,
        "final_step": rows[-1][0],
        "final_success": rows[-1][1],
    }


def scan_runs(roots: list[Path]) -> list[dict[str, Any]]:
    runs = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for eval_path in root.rglob("eval.csv"):
            run_dir = eval_path.parent
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = run_from_dir(run_dir)
            if run is not None:
                runs.append(run)
    return runs


def mean_by_step(runs: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for step, success in run["rows"]:
            by_step[step].append(success)
    steps = sorted(by_step)
    means = [sum(by_step[step]) / len(by_step[step]) for step in steps]
    return steps, means


def best_per_seed(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for run in runs:
        seed = run["seed"]
        current = best.get(seed)
        if current is None or (run["final_step"], run["final_success"]) > (
            current["final_step"],
            current["final_success"],
        ):
            best[seed] = run
    return list(best.values())


def write_summary(runs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_group",
        "agent",
        "env_short",
        "env_name",
        "seed",
        "alpha",
        "expectile",
        "noise_scale",
        "inv_temp",
        "final_step",
        "final_success",
        "path",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r["run_group"], r["env_short"], r["agent"], r["seed"], str(r["path"]))):
            writer.writerow({field: run.get(field) for field in fields})


def label_for_run_group(run_group: str, agent: str) -> str:
    if run_group == "s2_offline":
        return "Offline data"
    if run_group == "s2_wsrl":
        return "WSRL"
    if run_group == "s3_yours":
        return "Ours"
    return DISPLAY_NAMES.get(agent, agent.upper())


def plot_runs(
    runs: list[dict[str, Any]],
    output_path: Path,
    title: str,
    group_by: str = "agent",
    individual: bool = False,
) -> None:
    plt.figure(figsize=(7.0, 4.2))
    plotted = 0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = str(run[group_by])
        grouped[key].append(run)

    for key, group in sorted(grouped.items()):
        if individual:
            plot_group = [max(group, key=lambda r: (r["final_step"], r["final_success"]))]
        else:
            plot_group = best_per_seed(group)

        steps, means = mean_by_step(plot_group)
        if not steps:
            continue

        sample = plot_group[0]
        label_base = label_for_run_group(sample["run_group"], sample["agent"]) if group_by == "run_group" else DISPLAY_NAMES.get(key, key)
        seed_count = len({r["seed"] for r in plot_group if r["seed"] != -1})
        label = f"{label_base} ({seed_count or len(plot_group)} seed(s))"
        color = COLORS.get(sample["agent"])
        plt.plot(steps, means, marker="o", linewidth=2.2, label=label, color=color)
        plotted += 1

    plt.axvline(ONLINE_START_STEP, color="black", linestyle="--", linewidth=1.2, label="Online begins")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation success rate")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend(frameon=False)
    else:
        plt.text(0.5, 0.5, "No matching runs found", transform=plt.gca().transAxes, ha="center")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def filter_runs(
    runs: list[dict[str, Any]],
    groups: list[str] | None = None,
    agents: list[str] | None = None,
    envs: list[str] | None = None,
) -> list[dict[str, Any]]:
    out = runs
    if groups:
        out = [run for run in out if run["run_group"] in groups]
    if agents:
        out = [run for run in out if run["agent"] in agents]
    if envs:
        out = [
            run
            for run in out
            if run["env_short"] in envs or any(env in run["env_name"] for env in envs)
        ]
    return out


def plot_default_report_figures(runs: list[dict[str, Any]], output_dir: Path) -> None:
    s1 = filter_runs(runs, groups=["s1_sacbc", "s1_fql"], envs=["cube-single"])
    plot_runs(
        filter_runs(s1, agents=["sacbc"]),
        output_dir / "naive_sacbc.png",
        "SAC+BC Naive Baseline on Cube Single",
        individual=True,
    )
    plot_runs(
        filter_runs(s1, agents=["fql"]),
        output_dir / "naive_fql.png",
        "FQL Naive Baseline on Cube Single",
        individual=True,
    )
    plot_runs(
        s1,
        output_dir / "naive_sacbc_fql.png",
        "Naive Offline-to-Online Baselines on Cube Single",
        group_by="agent",
        individual=True,
    )

    for env in ("cube-double", "antsoccer-arena"):
        improved = filter_runs(
            runs,
            groups=["s2_offline", "s2_wsrl", "s2_ifql", "s2_dsrl", "s2_qsm", "s2_qsm_final_cube", "s2_qsm_final_ant"],
            envs=[env],
        )
        plot_runs(
            improved,
            output_dir / f"improved_{env}.png",
            f"Improved Baselines on {env}",
            group_by="run_group",
        )

        final = filter_runs(
            runs,
            groups=["s2_offline", "s2_wsrl", "s2_ifql", "s2_dsrl", "s2_qsm", "s2_qsm_final_cube", "s2_qsm_final_ant", "s3_yours"],
            envs=[env],
        )
        plot_runs(
            final,
            output_dir / f"final_comparison_{env}.png",
            f"Final Method Comparison on {env}",
            group_by="run_group",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot report figures from experiment folders containing eval.csv and flags.json."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["report_data", "modal_exp", "exp"],
        help="Directories to scan recursively.",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional run_group filter, e.g. s2_ifql s2_dsrl.")
    parser.add_argument("--agents", nargs="*", default=None, help="Optional agent filter, e.g. fql qsm.")
    parser.add_argument("--envs", nargs="*", default=None, help="Optional env filter, e.g. cube-double antsoccer-arena.")
    parser.add_argument("--group_by", choices=["agent", "run_group"], default="run_group")
    parser.add_argument("--name", default=None, help="If set, write one custom plot with this filename stem.")
    parser.add_argument("--title", default=None, help="Title for --name custom plot.")
    parser.add_argument("--no_defaults", action="store_true", help="Only write summary/custom plot.")
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    output_dir = Path(args.output_dir)
    runs = scan_runs(roots)
    write_summary(runs, output_dir / "all_runs_summary.csv")

    selected = filter_runs(runs, groups=args.groups, agents=args.agents, envs=args.envs)
    if args.name:
        title = args.title or args.name.replace("_", " ").title()
        plot_runs(selected, output_dir / f"{args.name}.png", title, group_by=args.group_by)
        write_summary(selected, output_dir / f"{args.name}_summary.csv")

    if not args.no_defaults:
        plot_default_report_figures(runs, output_dir)

    print(f"Found {len(runs)} eval runs from: {', '.join(str(root) for root in roots)}")
    print(f"Wrote summary to {output_dir / 'all_runs_summary.csv'}")
    if args.name:
        print(f"Wrote custom plot to {output_dir / (args.name + '.png')}")
    if not args.no_defaults:
        print(f"Wrote default report plots to {output_dir}")


if __name__ == "__main__":
    main()
