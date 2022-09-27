from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
import argparse
import yaml
from utils.validation_utils import validate_checkpoint

from models.CountingAnything import CountingAnything


def main():
    if CFG["seed"] != -1:
        seed_everything(CFG["seed"], workers=True)

    callbacks = get_callbacks()
    print(CFG["resume_path"])
    if CFG["test"]:
        validate_checkpoint(CFG)
    else:
        t_logger = TensorBoardLogger(
            CFG["log_dir"], name=CFG["name"], default_hp_metric=False
        )
        trainer = Trainer(
            gpus=-1,
            logger=t_logger,
            max_epochs=CFG["max_epochs"],
            max_steps=CFG["max_steps"],
            accelerator="gpu",
            strategy=DDPPlugin(find_unused_parameters=False),
            callbacks=callbacks,
            overfit_batches=CFG["overfit_batches"],
            check_val_every_n_epoch=CFG["val_every"],
            log_every_n_steps=1,
            num_sanity_val_steps=-1,
        )

        model = CountingAnything(CFG)
        trainer.fit(model)


def get_callbacks():
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor]

    if CFG["test_split"] == "val":
        if CFG["use_localisation_head"]:
            model_checkpoint_loc = ModelCheckpoint(
                monitor="val_DDP_Loc_MSE",
                save_last=True,
                save_top_k=3,
                every_n_epochs=1,
                filename="{epoch}_{val_DDP_Loc_MSE:.2f}",
            )
            callbacks.append(model_checkpoint_loc)

        if CFG["use_counting_head"]:
            model_checkpoint_MAE = ModelCheckpoint(
                monitor="val_DDP_MAE",
                save_last=True,
                save_top_k=3,
                every_n_epochs=1,
                filename="{epoch}_{val_DDP_MAE:.2f}_{val_DDP_RMSE:.2f}",
            )
            callbacks.append(model_checkpoint_MAE)
    return callbacks


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Train a 3D reconstruction model.")
    PARSER.add_argument("--config", "-c", type=str, help="Path to config file.")
    PARSER.add_argument(
        "--test", action="store_true", help="Only run a test script on the test set"
    )
    PARSER.add_argument(
        "--val", action="store_true", help="Only run a test script on the val set"
    )
    PARSER.add_argument(
        "--save_ims", action="store_true", help="save images to visualise"
    )
    ARGS = PARSER.parse_args()

    CFG = yaml.safe_load(open("configs/_DEFAULT.yml"))
    CFG_new = yaml.safe_load(open(f"configs/{ARGS.config}.yml"))

    CFG.update(CFG_new)

    CFG["name"] = ARGS.config
    CFG["test"] = ARGS.test or ARGS.val

    if ARGS.val:
        CFG["test_split"] = "val"
    elif ARGS.test:
        CFG["test_split"] = "test"

    CFG["save_ims"] = ARGS.save_ims and ARGS.test

    main()
