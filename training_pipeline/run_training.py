#!/usr/bin/env python3
"""Python runner for SleepKit training with custom feature set registrations.

This script registers the custom FS-WEARABLE-EXT feature set before invoking
SleepKit's training pipeline. Use this instead of the CLI when using custom
features.

Usage:
    # Feature extraction with custom feature set
    python scripts/run_training.py --mode feature --config configs/wearable-ext-features.json

    # Train with custom features
    python scripts/run_training.py --mode train --config configs/wearable-ss-5-tcn.json

    # Evaluate
    python scripts/run_training.py --mode evaluate --config configs/wearable-ss-5-tcn.json

    # Export to TFLite
    python scripts/run_training.py --mode export --config configs/wearable-ss-5-tcn.json

For built-in feature sets (FS-W-PA-14), you can use the standard CLI instead:
    sleepkit --task stage --mode train --config configs/wearable-ss-5-tcn.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def register_custom_components():
    """Register custom feature sets and any other extensions."""
    import sleepkit as sk

    # Register custom extended wearable feature set
    try:
        from sleepkit_custom.features.fs_wearable_ext import FSWearableExt

        sk.FeatureFactory.register("FS-WEARABLE-EXT", FSWearableExt)
        logger.info("Registered custom feature set: FS-WEARABLE-EXT")
    except ImportError as e:
        logger.warning("Could not import custom feature set: %s", e)
        logger.warning("Only built-in feature sets will be available.")


def main():
    parser = argparse.ArgumentParser(
        description="SleepKit training runner with custom registrations"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["download", "feature", "train", "evaluate", "export", "demo"],
        help="SleepKit task mode",
    )
    parser.add_argument(
        "--task",
        default="stage",
        choices=["detect", "stage", "apnea", "arousal"],
        help="SleepKit task type (default: stage)",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Add project root to path for custom imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Register custom components
    register_custom_components()

    import sleepkit as sk

    # Load configuration
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config) as f:
        config_data = json.load(f)

    params = sk.TaskParams(**config_data)

    # Get task handler
    task_handler = sk.TaskFactory.get(args.task)

    # Execute requested mode
    logger.info("Running %s/%s with config: %s", args.task, args.mode, args.config)

    match args.mode:
        case "download":
            task_handler.download(params)
        case "feature":
            task_handler.feature(params)
        case "train":
            task_handler.train(params)
        case "evaluate":
            task_handler.evaluate(params)
        case "export":
            task_handler.export(params)
        case "demo":
            task_handler.demo(params)

    logger.info("Completed %s/%s", args.task, args.mode)


if __name__ == "__main__":
    main()
