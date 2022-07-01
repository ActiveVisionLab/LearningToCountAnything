import argparse
from pathlib import Path
import os
import json

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from models.CountingAnything import CountingAnything
from utils.data_utils import get_loader_counting
from utils.validation_utils import get_checkpoint_list, validate_checkpoint


def main():

    if CFG["seed"] != -1:
        seed_everything(CFG["seed"], workers=True)

    if not CFG["test"]:
        train_loader, test_loader = get_loader_counting(CFG)

        t_logger = TensorBoardLogger(
            CFG["log_dir"], name=CFG["name"], default_hp_metric=False
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        model_checkpoint_MAE = ModelCheckpoint(
            monitor="val_DDP_MAE",
            save_last=True,
            save_top_k=3,
            every_n_epochs=1,
            filename="{epoch}-{val_DDP_MAE:.2f}-{val_DDP_RMSE:.2f}",
        )

        trainer = Trainer(
            gpus=-1,
            logger=t_logger,
            max_epochs=CFG["max_epochs"],
            max_steps=CFG["max_steps"],
            accelerator="gpu",
            strategy=DDPPlugin(find_unused_parameters=False),
            callbacks=[
                model_checkpoint_MAE,
                lr_monitor,
            ],
            overfit_batches=CFG["overfit_batches"],
            check_val_every_n_epoch=CFG["val_every"],
            log_every_n_steps=1,
            accumulate_grad_batches=CFG["accumulate_grad_batches"],
        )

        model = CountingAnything(CFG)
        # train
        trainer.fit(model, train_loader, test_loader)

    else:
        test_loader = get_loader_counting(CFG)

        t_logger_test = TensorBoardLogger(
            CFG["log_dir"],
            name=CFG["name"] + "_test_to_delete",
            default_hp_metric=False,
        )

        # get the file of all previously tested checkpoints
        if CFG["test_split"] == "val":
            val_res = "results_val.json"
        else:
            val_res = "results_test.json"
        if not os.path.exists(val_res):
            data = {"": ""}
            with open(val_res, "w") as outfile:
                json.dump(data, outfile)

        # get the list of all checkpoints to test and ignore previously tested checkpoints
        chkpt_list = get_checkpoint_list(
            CFG["resume_path"], CFG["name"], val_res=val_res
        )

        # test all previously untested checkpoints and save to the results file
        for ckpt in chkpt_list:
            if CFG["save_ims"]:
                resume_path = ckpt
                resume_path = resume_path.split("/")
                print("resume_path: ", resume_path)
                cnf = resume_path[1]
                vrs = resume_path[2]
                ckp = resume_path[4].split(".")[0]

                dir_pth = "output/visualised/" + cnf + "/" + vrs + "/" + ckp
                print("saving to: ", dir_pth)
                CFG["output_dir"] = dir_pth
                Path(dir_pth).mkdir(parents=True, exist_ok=True)

            validate_checkpoint(t_logger_test, test_loader, ckpt, CFG, val_res)


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
        "--save_ims",
        action="store_true",
        help="save images to visualise, only works when --test is also specified",
    )
    ARGS = PARSER.parse_args()

    CFG = yaml.safe_load(open("configs/_DEFAULT.yml"))
    CFG_new = yaml.safe_load(open("configs/{}.yml".format(ARGS.config)))

    CFG.update(CFG_new)
    CFG["name"] = ARGS.config
    CFG["test"] = ARGS.test or ARGS.val
    if CFG["test"]:
        CFG["resume_path"] = "logs/" + CFG["name"]

    if ARGS.val:
        CFG["test_split"] = "val"
    elif ARGS.test:
        CFG["test_split"] = "test"

    # save_ims only works when --test/val is also specified
    CFG["save_ims"] = ARGS.save_ims and CFG["test"]

    main()
