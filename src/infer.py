from typing import Any, Dict, List, Tuple

import os
import hydra
import rootutils
import torch
import shutil

from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)
from src.data.components import Var_Dataset
from src.data.constants import LABEL_TO_INT

INT_TO_LABEL = {v: k.replace(" ", "_") for k, v in LABEL_TO_INT.items()}

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def infer(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Initialing dataset")
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((cfg.data.image_size[0], cfg.data.image_size[1])),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    dataset = Var_Dataset(
        data_dir="src/data/image",
        transform=transform,
        train=False
    )
    log.info(f"Length of dataset: {len(dataset)}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64
    )

    log.info(f"Instantiating model <{cfg.model.net._target_}>")
    model: nn.Module = hydra.utils.instantiate(cfg.model.net)
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")

    model_weights = checkpoint["state_dict"]

    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("net.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()
    model = model.to("cpu")
    for dataloader in dataloader:
        predicts = model(dataloader[0].to("cpu"))
        predicts = torch.argmax(predicts, dim=1)
        for l, path in zip(predicts, dataloader[-1]):
            dest = os.path.join("data/my_data", INT_TO_LABEL[l.item()])
            if not os.path.exists(dest):
                os.makedirs(dest)
            
            shutil.copy(path, dest)

    log.info("Starting testing!")

    return None, None



@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    infer(cfg)


if __name__ == "__main__":
    main()
