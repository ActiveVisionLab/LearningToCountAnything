import os
import json
from pytorch_lightning import Trainer
from models.CountingAnything import CountingAnything
from pytorch_lightning.plugins import DDPPlugin


def get_checkpoint_list(l_dir, config_name, val_res="results_val.json"):
    # get the list of all checkpoints to test and ignore previously tested checkpoints

    if l_dir == "":
        l_dir = "logs/" + config_name

    l_dir_split = l_dir.split("/")
    chkpt_list = []
    if l_dir.endswith(".ckpt"):
        chkpt_list.append(l_dir)
    else:
        if l_dir_split[-1].startswith("version"):
            versions = [l_dir_split[-1]]
            config_dir = "/".join(l_dir_split[:-1])
            print("l_dir_split", l_dir_split)
            print("config_dir", config_dir)
            print("remove the version ID")
            exit()

        else:
            config_dir = l_dir
            print("l_dir", l_dir)
            vers = os.listdir(config_dir)
            versions = [
                i
                for i in os.listdir(config_dir)
                if (i.startswith("version_")) and "." not in i
            ]

        print("versions", versions)

        for version in versions:
            chkpt_dir = config_dir + "/" + version + "/checkpoints/"
            chkpt_files = os.listdir(chkpt_dir)
            epoch_chkpts = [
                i
                for i in chkpt_files
                if (i.startswith("epoch") and i.endswith(".ckpt"))
            ]
            v_chkpt_list = [chkpt_dir + e for e in epoch_chkpts]

            chkpt_list.extend(v_chkpt_list)
    print("total checkpoints found {}".format(len(chkpt_list), chkpt_list))

    with open(val_res, "r") as f:
        dic_already_done = json.load(f)

    chkpt_list_already_done = [i for i in chkpt_list if (i in dic_already_done.keys())]
    chkpt_list = [i for i in chkpt_list if (i not in dic_already_done.keys())]

    print(
        "{} chkpt_list_already_done : {}".format(
            len(chkpt_list_already_done), chkpt_list_already_done
        )
    )
    print("evaluating {} checkpoint(s): {}".format(len(chkpt_list), chkpt_list))

    return chkpt_list


def validate_checkpoint(t_logger, test_loader, chkpt, CFG, val_res):
    # given a single checkpoint path, update the results json
    CFG["resume_path"] = chkpt
    print("test_split", CFG["test_split"])
    print("resume_path", CFG["resume_path"])
    model = CountingAnything(CFG)

    val_trainer = Trainer(
        gpus=-1,
        logger=t_logger,
        max_epochs=CFG["max_epochs"],
        max_steps=CFG["max_steps"],
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        overfit_batches=CFG["overfit_batches"],
        check_val_every_n_epoch=CFG["val_every"],
        log_every_n_steps=1,
    )
    val_results = val_trainer.validate(model, test_loader)
    val_results = val_results[0]
    val_results.update(CFG)

    if not CFG["save_ims"]:
        with open(val_res, "r") as f:
            dic = json.load(f)

        dic[chkpt] = val_results

        with open(val_res, "w") as f:
            json.dump(dic, f, indent=4, sort_keys=True)
