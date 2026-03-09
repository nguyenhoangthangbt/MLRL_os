"""CLI entry point for ML/RL OS.

Usage:
    mlrl-os serve                        # Start the API server
    mlrl-os run <config.yaml>            # Run experiment from YAML
    mlrl-os validate <config.yaml>       # Validate config without running
    mlrl-os datasets list                # List registered datasets
    mlrl-os datasets import <file>       # Import a dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML experiment config file."""
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the API server."""
    import uvicorn

    from mlrl_os.api.app import create_app
    from mlrl_os.config.defaults import MLRLSettings

    settings = MLRLSettings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port or settings.api_port,
        log_level=settings.log_level.lower(),
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment from a YAML config file."""
    _setup_logging()

    from mlrl_os.config.defaults import MLRLSettings
    from mlrl_os.data.registry import DatasetRegistry
    from mlrl_os.experiment.runner import ExperimentRunner

    config = _load_yaml_config(Path(args.config))
    dataset_id = config.pop("dataset_id", None)
    if not dataset_id:
        print("Error: config must include 'dataset_id'", file=sys.stderr)
        sys.exit(1)

    settings = MLRLSettings()
    registry = DatasetRegistry(settings)

    try:
        meta = registry.get_meta(dataset_id)
    except KeyError:
        print(f"Error: dataset '{dataset_id}' not found", file=sys.stderr)
        sys.exit(1)

    runner = ExperimentRunner(settings=settings, dataset_registry=registry)
    result = runner.run(config, meta)

    print(f"\nExperiment: {result.experiment_id}")
    print(f"Status:     {result.status.value}")
    if result.best_algorithm:
        print(f"Best:       {result.best_algorithm}")
    if result.metrics:
        print("Metrics:")
        for k, v in result.metrics.items():
            print(f"  {k}: {v:.4f}")
    if result.error_message:
        print(f"Error:      {result.error_message}")
    if result.duration_seconds:
        print(f"Duration:   {result.duration_seconds:.1f}s")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a YAML config without running the experiment."""
    _setup_logging()

    from mlrl_os.config.defaults import MLRLSettings
    from mlrl_os.data.registry import DatasetRegistry
    from mlrl_os.experiment.runner import ExperimentRunner

    config = _load_yaml_config(Path(args.config))
    dataset_id = config.pop("dataset_id", None)
    if not dataset_id:
        print("Error: config must include 'dataset_id'", file=sys.stderr)
        sys.exit(1)

    settings = MLRLSettings()
    registry = DatasetRegistry(settings)

    try:
        meta = registry.get_meta(dataset_id)
    except KeyError:
        print(f"Error: dataset '{dataset_id}' not found", file=sys.stderr)
        sys.exit(1)

    runner = ExperimentRunner(settings=settings, dataset_registry=registry)
    result = runner.validate_only(config, meta)

    if result.valid:
        print("Validation PASSED")
    else:
        print("Validation FAILED")
        for e in result.errors:
            print(f"  [{e.code}] {e.message}")

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  {w.message}")


def cmd_datasets_list(args: argparse.Namespace) -> None:
    """List all registered datasets."""
    from mlrl_os.config.defaults import MLRLSettings
    from mlrl_os.data.registry import DatasetRegistry

    settings = MLRLSettings()
    registry = DatasetRegistry(settings)
    datasets = registry.list_datasets()

    if not datasets:
        print("No datasets registered.")
        return

    print(f"{'ID':<14} {'Name':<30} {'Type':<10} {'Snap':<6} {'Traj':<6} {'Registered'}")
    print("-" * 90)
    for m in datasets:
        snap = str(m.snapshot_row_count or "-")
        traj = str(m.trajectory_row_count or "-")
        print(f"{m.id:<14} {m.name[:28]:<30} {m.source_type:<10} {snap:<6} {traj:<6} {m.registered_at[:19]}")


def cmd_datasets_import(args: argparse.Namespace) -> None:
    """Import a dataset from a file."""
    _setup_logging()

    from mlrl_os.config.defaults import MLRLSettings
    from mlrl_os.data.external_loader import ExternalLoader
    from mlrl_os.data.registry import DatasetRegistry
    from mlrl_os.data.simos_loader import SimosLoader

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = SimosLoader().load(path)
    elif suffix == ".parquet":
        raw = ExternalLoader().load_parquet(path)
    else:
        raw = ExternalLoader().load_csv(path)

    settings = MLRLSettings()
    registry = DatasetRegistry(settings)
    name = args.name or path.stem
    meta = registry.register(raw, name=name)

    print(f"Dataset registered: {meta.id}")
    print(f"  Name:       {meta.name}")
    print(f"  Snapshots:  {meta.snapshot_row_count or 'none'}")
    print(f"  Trajectories: {meta.trajectory_row_count or 'none'}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mlrl-os",
        description="ML/RL OS — Predictive Intelligence for Operational Systems",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_p = sub.add_parser("serve", help="Start the API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=None)

    # run
    run_p = sub.add_parser("run", help="Run experiment from YAML config")
    run_p.add_argument("config", help="Path to YAML config file")

    # validate
    val_p = sub.add_parser("validate", help="Validate config without running")
    val_p.add_argument("config", help="Path to YAML config file")

    # datasets
    ds_p = sub.add_parser("datasets", help="Dataset management")
    ds_sub = ds_p.add_subparsers(dest="ds_command")

    ds_sub.add_parser("list", help="List registered datasets")

    imp_p = ds_sub.add_parser("import", help="Import a dataset file")
    imp_p.add_argument("file", help="Path to CSV, Parquet, or JSON file")
    imp_p.add_argument("--name", default=None, help="Dataset name")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "datasets":
        if args.ds_command == "list":
            cmd_datasets_list(args)
        elif args.ds_command == "import":
            cmd_datasets_import(args)
        else:
            ds_p.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
