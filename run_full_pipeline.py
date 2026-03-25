from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    description: str
    command: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full VOC experiment pipeline step by step with Markdown-formatted progress output."
    )
    parser.add_argument("--dataset", default="voc", choices=["voc"])
    parser.add_argument("--device", default="cuda:0", help="CUDA-only device. Default: cuda:0.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the three smoke-test training steps.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the three formal training steps.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmarking.")
    parser.add_argument("--skip-infer", action="store_true", help="Skip qualitative inference.")
    parser.add_argument("--start-at", help="Start from a specific step key, for example: train_baseline")
    parser.add_argument("--stop-after", help="Stop after a specific step key, for example: evaluate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = build_steps(dataset=args.dataset, device=args.device, skip_smoke=args.skip_smoke, skip_train=args.skip_train)
    steps = apply_step_window(steps, start_at=args.start_at, stop_after=args.stop_after)

    if args.skip_evaluate:
        steps = [step for step in steps if step.key != "evaluate"]
    if args.skip_benchmark:
        steps = [step for step in steps if step.key != "benchmark"]
    if args.skip_infer:
        steps = [step for step in steps if step.key != "infer"]

    if not steps:
        raise ValueError("No pipeline steps remain after applying the selected options.")

    print("# VOC Full Pipeline", flush=True)
    print("", flush=True)
    print(f"- Working directory: `{ROOT}`", flush=True)
    print(f"- Python: `{sys.executable}`", flush=True)
    print(f"- Dataset: `{args.dataset}`", flush=True)
    print(f"- Device: `{args.device}`", flush=True)

    for index, step in enumerate(steps, start=1):
        render_step_banner(index=index, total=len(steps), step=step)
        completed = run_step(step)
        print("", flush=True)
        print(f"### Completed: `{completed.key}`", flush=True)
        print(f"- Title: {completed.title}", flush=True)

    print("", flush=True)
    print("## Pipeline Finished", flush=True)
    print("- All selected steps completed successfully.", flush=True)
    return 0


def build_steps(dataset: str, device: str, skip_smoke: bool, skip_train: bool) -> list[Step]:
    python = sys.executable
    steps: list[Step] = [
        Step(
            key="prepare",
            title="Prepare VOC Config",
            description="Validate the VOC subset and regenerate the formal dataset YAML.",
            command=[python, "prepare_data.py", "--dataset", dataset],
        )
    ]

    if not skip_smoke:
        steps.extend(
            [
                Step(
                    key="smoke_baseline",
                    title="Smoke Test Baseline",
                    description="Run a short baseline smoke test to confirm the training path works.",
                    command=[
                        python,
                        "train.py",
                        "--dataset",
                        dataset,
                        "--model",
                        "baseline",
                        "--epochs",
                        "1",
                        "--fraction",
                        "0.01",
                        "--batch",
                        "4",
                        "--imgsz",
                        "320",
                        "--device",
                        device,
                    ],
                ),
                Step(
                    key="smoke_mobilenetv2",
                    title="Smoke Test MobileNetV2",
                    description="Run a short MobileNetV2 smoke test.",
                    command=[
                        python,
                        "train.py",
                        "--dataset",
                        dataset,
                        "--model",
                        "mobilenetv2",
                        "--epochs",
                        "1",
                        "--fraction",
                        "0.01",
                        "--batch",
                        "4",
                        "--imgsz",
                        "320",
                        "--device",
                        device,
                    ],
                ),
                Step(
                    key="smoke_pcg_ghost",
                    title="Smoke Test PCG-Ghost",
                    description="Run a short PCG-Ghost smoke test.",
                    command=[
                        python,
                        "train.py",
                        "--dataset",
                        dataset,
                        "--model",
                        "pcg_ghost",
                        "--epochs",
                        "1",
                        "--fraction",
                        "0.01",
                        "--batch",
                        "4",
                        "--imgsz",
                        "320",
                        "--device",
                        device,
                    ],
                ),
            ]
        )

    if not skip_train:
        steps.extend(
            [
                Step(
                    key="train_baseline",
                    title="Formal Train Baseline",
                    description="Train the baseline model with the default formal VOC settings.",
                    command=[python, "train.py", "--dataset", dataset, "--model", "baseline", "--device", device],
                ),
                Step(
                    key="train_mobilenetv2",
                    title="Formal Train MobileNetV2",
                    description="Train the MobileNetV2 variant with the default formal VOC settings.",
                    command=[python, "train.py", "--dataset", dataset, "--model", "mobilenetv2", "--device", device],
                ),
                Step(
                    key="train_pcg_ghost",
                    title="Formal Train PCG-Ghost",
                    description="Train the PCG-Ghost variant with the default formal VOC settings.",
                    command=[python, "train.py", "--dataset", dataset, "--model", "pcg_ghost", "--device", device],
                ),
            ]
        )

    steps.extend(
        [
            Step(
                key="evaluate",
                title="Evaluate All Models",
                description="Run validation and update the unified comparison CSV.",
                command=[python, "evaluate.py", "--dataset", dataset, "--split", "val", "--device", device],
            ),
            Step(
                key="benchmark",
                title="Benchmark All Models",
                description="Measure latency and FPS for all trained models on CUDA.",
                command=[python, "benchmark.py", "--dataset", dataset, "--device", device],
            ),
            Step(
                key="infer",
                title="Generate Qualitative Figures",
                description="Create final prediction figures for the three trained models.",
                command=[python, "infer.py", "--dataset", dataset, "--device", device],
            ),
        ]
    )
    return steps


def apply_step_window(steps: list[Step], start_at: str | None, stop_after: str | None) -> list[Step]:
    ordered_keys = [step.key for step in steps]

    if start_at and start_at not in ordered_keys:
        raise ValueError(f"Unknown --start-at step: {start_at}. Available: {', '.join(ordered_keys)}")
    if stop_after and stop_after not in ordered_keys:
        raise ValueError(f"Unknown --stop-after step: {stop_after}. Available: {', '.join(ordered_keys)}")

    start_index = ordered_keys.index(start_at) if start_at else 0
    stop_index = ordered_keys.index(stop_after) if stop_after else len(steps) - 1

    if start_index > stop_index:
        raise ValueError("--start-at must come before or match --stop-after.")

    return steps[start_index : stop_index + 1]


def render_step_banner(index: int, total: int, step: Step) -> None:
    print("", flush=True)
    print(f"## Step {index}/{total}: {step.title}", flush=True)
    print("", flush=True)
    print(step.description, flush=True)
    print("", flush=True)
    print(f"- Step key: `{step.key}`", flush=True)
    print("- Command:", flush=True)
    print("```powershell", flush=True)
    print(format_command(step.command), flush=True)
    print("```", flush=True)
    print("", flush=True)


def run_step(step: Step) -> Step:
    result = subprocess.run(step.command, cwd=ROOT)
    if result.returncode != 0:
        print("", flush=True)
        print(f"### Failed: `{step.key}`", flush=True)
        print(f"- Exit code: `{result.returncode}`", flush=True)
        raise SystemExit(result.returncode)
    return step


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
