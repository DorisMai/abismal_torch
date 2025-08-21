import logging

import lightning.pytorch as L

logger = logging.getLogger(__name__)


class DatasetLogger(L.Callback):
    def __init__(self):
        super().__init__()

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )
            print("Console logging configured by DatasetLogger")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log dataset information to console at the start of training"""
        datamodule = trainer.datamodule
        if hasattr(datamodule, "dataset") and hasattr(datamodule.dataset, "datasets"):
            print("=== Dataset Information ===")
            logger.info("=== Dataset Information ===")
            logger.info(f"Total datasets: {len(datamodule.dataset.datasets)}")
            logger.info(f"Total length: {len(datamodule.dataset)}")

            logger.info("Metadata keys for each dataset:")
            for i, ds in enumerate(datamodule.dataset.datasets):
                logger.info(
                    f"  Dataset {i} ({ds.__class__.__name__}): {ds.metadata_keys}"
                )
                logger.info(f"    - Handler type: {ds.__HANDLER_TYPE__}")
                logger.info(f"    - Length: {len(ds)}")
                logger.info(f"    - RASU ID: {ds.rasu_id}")

            logger.info("=== End Dataset Information ===")
