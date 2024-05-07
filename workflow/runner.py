from typing import Any

import hydra
import lightning as L
from lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from PAR_NP.data.abstract import NPDataLoader
from PAR_NP.lightning import LitNP
from PAR_NP.logging import patch_loggers_for_hydra
from PAR_NP.logging.mlflow import prepare_mlflow


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    patch_loggers_for_hydra()
    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    ##########
    # MLflow #
    ##########

    using_mlflow = "mlflow" in cfg and cfg.mlflow.get("enable_mlflow", False)
    manual_mlflow_params: dict[str, Any] = {}

    if using_mlflow:
        mlflow_details, mlflow_experiment_name, exp_unique_id = prepare_mlflow(
            cfg.mlflow,
            cfg.data_split,
            cfg.data_filtering,
            manual_mlflow_params,
        )
        mlflow_run_name = mlflow_details.get("run_name", None)
    else:
        mlflow_experiment_name = None
        mlflow_run_name = None

    logger = MLFlowLogger(
        experiment_name=mlflow_experiment_name,
        run_name=mlflow_run_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    ##################
    #   Data setup   #
    ##################
    data_generator = cfg.data_generator(
        **OmegaConf.to_container(cfg.data_generator_kwargs)
        if "data_generator_kwargs" in cfg
        else {}
    )
    train_loader = NPDataLoader(
        data_generator,
        batch_size=cfg.train.batch_size,
    )

    #########################
    # Lightning Model setup #
    #########################
    model = LitNP(
        cfg.model_name,
        **OmegaConf.to_container(cfg.model_kwargs) if "model_kwargs" in cfg else {}
    )

    #################
    # Training loop #
    #################
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        gpus=cfg.train.gpus,
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        progress_bar_refresh_rate=cfg.train.progress_bar_refresh_rate,
        callbacks=cfg.train.callbacks,
    )

    trainer.fit(model, train_loader)
