import os
import json
from pytorch_lightning import Trainer
from models.CountingAnything import CountingAnything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger


def validate_checkpoint(CFG):
    # given a single checkpoint path, update the results json
    print("test_split", CFG["test_split"])
    print("resume_path", CFG["resume_path"])
    model = CountingAnything(CFG)

    logger = TensorBoardLogger(
        CFG["log_dir"],
        name=f"{CFG['name']}_{CFG['test_split']}",
        default_hp_metric=False,
    )

    val_trainer = Trainer(
        gpus=-1,
        logger=logger,
        max_epochs=CFG["max_epochs"],
        max_steps=CFG["max_steps"],
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        overfit_batches=CFG["overfit_batches"],
        check_val_every_n_epoch=CFG["val_every"],
        log_every_n_steps=1,
    )

    if CFG["test_split"] == "val":
        val_res = "results_val.json"
    else:
        val_res = "results_test.json"

    if not os.path.exists(val_res):
        data = {"": ""}
        with open(val_res, "w") as outfile:
            json.dump(data, outfile)

    val_results = val_trainer.validate(model)
    val_results = val_results[0]

    val_results.update(CFG)

    if not CFG["save_ims"]:
        with open(val_res, "r") as f:
            dic = json.load(f)

        dic[CFG["resume_path"]] = val_results

        with open(val_res, "w") as f:
            json.dump(dic, f, indent=4, sort_keys=True)

