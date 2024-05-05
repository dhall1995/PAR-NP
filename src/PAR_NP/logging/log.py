import logging


def get_logger(name) -> logging.Logger:
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only add handlers if we're not using Hydra
    # Note: in most cases, `get_logger` is called outside of a class/function, so is called immediately
    # when a script is run. This is before Hydra has been initialised, so these checks don't do what we want.
    # But, I include them here anyway in case `get_logger` is called in a different context.
    # `patch_loggers_for_hydra` should be called after Hydra has been initialised to remove the StreamHandlers.
    try:
        from hydra.core.hydra_config import HydraConfig

        HydraConfig.get()
        using_hydra = True
    except ImportError:
        using_hydra = False
    except ValueError:
        using_hydra = False

    # add the handlers to the logger, except if we're using Hydra, in which case, rely on Hydra's logging
    if not logger.hasHandlers() and not using_hydra:
        logger.addHandler(ch)
    return logger


def patch_loggers_for_hydra():
    """
    This function will remove all StreamHandlers from loggers that are not propagating to the root logger.
    This is useful when using Hydra, as it will handle logging to the console.
    """
    for logger_name, logger_obj in logging.getLogger().manager.loggerDict.items():

        # If the logger isn't propagating to the root logger, then Hydra won't handle it, so we must skip.
        if (
            not isinstance(logger_obj, logging.Logger)
            or not logger_obj.propagate
            or logger_name.startswith("hydra")
        ):
            continue

        for handler in logger_obj.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger_obj.removeHandler(handler)
