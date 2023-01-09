#!/usr/bin/env python

import argparse, os, datetime
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from utils import instantiate_from_config
from callbacks import SetupCallback


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default="configs/basic.yaml")
    p.add_argument('--devices', type=str, default="0,")
    p.add_argument('--seed', type=int, default=22)
    return p


if __name__ == "__main__":

    # LOAD ARGUMENTS AND CONFIGS
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config)
    trainer_kwargs = dict()
    assert config.project
    assert config.name

    # SET SEED
    seed_everything(opt.seed)

    # DEFINE NAMES
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + "_" + config.name
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # DEFINE CALLBACK
    trainer_kwargs["callbacks"] = [
        ModelCheckpoint(dirpath=ckptdir,
                        filename="checkpoint_{epoch:03d}",
                        verbose=True,
                        save_last=False),
        SetupCallback(
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
        )
    ]

    # DEFINE LOGGER
    trainer_kwargs["logger"] = WandbLogger(project=config.project,
                                           name=nowname)

    # DEFINE Trainer
    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    trainer_kwargs["devices"] = opt.devices
    for key in config.trainer:
        trainer_kwargs[key] = config.trainer[key]
    trainer = Trainer(**trainer_kwargs)

    # DEFINE MODEL
    model = instantiate_from_config(config.model)

    # DEFINE DATASET
    data = instantiate_from_config(config.data)

    try:
        trainer.fit(model, data)
    except Exception:
        raise
