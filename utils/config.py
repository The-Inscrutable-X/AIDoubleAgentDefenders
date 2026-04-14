from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# YAML config rules
# ---------------------------------------------------------------------------
# The YAML is organized into named sections for readability but you can define section
# names in the yaml however you want since loading the yaml flattens it, but the field
# names inside each section must match the argparse dest names exactly.
# Sections that are missing or incomplete are fine — argparse defaults fill in.

def _flatten_yaml_sections(raw: dict) -> dict:
    """Flatten a sectioned YAML dict into a single {dest: value} dict."""
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def apply_config_to_parser(parser: argparse.ArgumentParser, config_path: str | Path) -> None:
    """Load a YAML config and inject its values as new parser defaults.

    Call this BEFORE parser.parse_args() BUT AFTER THE ADD_ARGUMENT lines.  Argparse's built-in behaviour then
    gives us the correct priority for free:
        explicit CLI arg  >  config value  >  hardcoded argparse default in main

    Also relaxes ``required=True`` for any arg that the config supplies, so
    the user doesn't have to repeat those on the command line.
    """
    # Load and flatten configs
    raw = yaml.safe_load(Path(config_path).read_text())
    flat = _flatten_yaml_sections(raw)

    # Checks that only arguments specified in mainscript argparse are let through. 
    known_dests = {action.dest for action in parser._actions}
    unknown = set(flat) - known_dests
    if unknown:
        raise ValueError(f"Config keys not recognised by the argument parser: {sorted(unknown)}")

    # Populate arguments specified by yaml by setting defaults which override the prev defaults set in main_training_script.py
    parser.set_defaults(**flat)
    for action in parser._actions:
        if action.required and action.dest in flat:
            action.required = False

def _suspend_required(parser: argparse.ArgumentParser) -> list:
    """Set required=False on all required actions; return the actions that were modified."""
    suspended = [action for action in parser._actions if action.required]
    for action in suspended:
        action.required = False
    return suspended

def _restore_required(suspended: list) -> None:
    for action in suspended:
        action.required = True

def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse CLI args with optional ``--config`` YAML override.

    Expects the parser to already have a ``--config`` argument added.
    """
    suspended = _suspend_required(parser)
    config_path = parser.parse_args().config
    _restore_required(suspended)

    # Replace defaults with those from config, and turn off the .required field if the config specified it.
    if config_path:
        apply_config_to_parser(parser, config_path)

    # Parse all args again, but now some of the args have already been filled with values from the config.
    return parser.parse_args()


def compute_run_name(args: argparse.Namespace, runname_suffix: str = "") -> str:
    """Build a deterministic run name from training arguments.

    Mirrors the naming convention from launch_main_training_script_parameterized.sh.
    """
    parts = [
        runname_suffix or "run",
        args.dataset,
        args.attacker_type,
        args.judge_model,
        args.reward_functions,
        args.loss_type,
        f"lr{args.learning_rate}",
        f"ga{args.gradient_accumulation_steps}",
        args.training_strategy,
        Path(args.attacker_prompts_dir).name,
        args.judge_prompt_version,
        f"tseed{args.torch_seed}"
    ]
    return "+".join(parts) + f"/{args.engine}_lora_model"


def compute_model_savepath(args: argparse.Namespace, results_dir: str, runname_suffix: str = "") -> str:
    """Compute model_savepath from results_dir and run name when not explicitly set."""
    run_name = compute_run_name(args, runname_suffix)
    return str(Path(results_dir) / "checkpoints_dir" / run_name)