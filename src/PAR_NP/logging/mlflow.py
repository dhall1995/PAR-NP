from __future__ import annotations

import os
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from PAR_NP.logging import get_logger

logger = get_logger(__name__)


def flatten_config(
    nested_config: dict | DictConfig, parent_key: str = "", sep: str = "__"
) -> dict[str, Any]:
    flattened_config = {}
    for key, value in nested_config.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict | DictConfig):
            flattened_config.update(flatten_config(value, new_key, sep=sep))
        else:
            flattened_config[new_key] = value
    return flattened_config


def prepare_mlflow(
    mlflow_details: DictConfig,
    data_split: DictConfig,
    data_filtering: DictConfig,
    manual_mlflow_params: dict[str, Any],
) -> tuple[DictConfig, str, str]:
    # Validation
    non_null_fields = ["experiment_name_prefix", "experiment_name"]
    for field in non_null_fields:
        if mlflow_details[field] is None:
            raise ValueError(f"{field} must be specified in mlflow config")
    if mlflow_details.experiment_date is None:
        mlflow_details.experiment_date = pd.Timestamp.now().strftime("%Y%m%d")

    # Params for PyKEEN
    mlflow_experiment_name = (
        f"{mlflow_details.experiment_name_prefix}_"
        f"{mlflow_details.experiment_date}_"
        f"{mlflow_details.experiment_name}"
    )

    # Additional logging that PyKEEN doesn't handle
    for k, v in data_split.items():
        manual_mlflow_params[f"data_split.{k}"] = v

    flat_data_filt_cfg = flatten_config(data_filtering, sep=".")
    for k, v in flat_data_filt_cfg.items():
        manual_mlflow_params[f"data_filtering.{k}"] = v

    # Put in a unique identifier that we can search for later
    exp_unique_id = os.urandom(16).hex()

    return (
        mlflow_details,
        mlflow_experiment_name,
        exp_unique_id,
    )
