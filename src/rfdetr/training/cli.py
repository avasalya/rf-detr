# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningCLI entry point for RF-DETR training and evaluation.

Provides the ``rfdetr`` command with auto-generated subcommands::

    rfdetr fit     --config configs/rfdetr_base.yaml
    rfdetr validate --ckpt_path output/best.ckpt
    rfdetr test    --ckpt_path output/best.ckpt
    rfdetr predict --ckpt_path output/best.ckpt

Both ``RFDETRModelModule`` and ``RFDETRDataModule`` share the same ``(model_config, train_config)`` constructor
signature.  ``link_arguments`` eliminates the duplication so the user specifies each config group once; the datamodule
receives the same values automatically at parse time.
"""

import inspect
from typing import Any

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from jsonargparse import set_parsing_settings

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer

_PARSER_KWARGS = {
    "prog": "rfdetr",
    "description": "RF-DETR command line interface.",
}


class RFDETRCli(LightningCLI):
    """LightningCLI subclass for RF-DETR training and evaluation.

    Wires ``RFDETRModelModule`` and ``RFDETRDataModule`` under a unified CLI, with argument linking that shares
    ``model_config`` and ``train_config`` between module and datamodule so the user only specifies them once.

    Auto-generated subcommands: ``fit``, ``validate``, ``test``, ``predict``.
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register argument links that share configs between module and datamodule.

        Linking ``model.model_config`` → ``data.model_config`` and ``model.train_config`` → ``data.train_config`` means
        the user specifies both config groups once; the datamodule receives the same values automatically.

        Args:
            parser: The jsonargparse ``LightningArgumentParser`` provided by
                ``LightningCLI``.
        """
        parser.link_arguments("model.model_config", "data.model_config", apply_on="parse")
        parser.link_arguments("model.train_config", "data.train_config", apply_on="parse")

    def _instantiate_trainer(self, config: dict[str, Any], callbacks: list[Callback]) -> Trainer:
        """Build the ``Trainer`` through :func:`build_trainer` so ``rfdetr fit`` matches ``RFDETR.train()``.

        The stock LightningCLI path constructs a bare ``Trainer`` from the ``trainer:`` section, which drops the whole
        RF-DETR training stack: per-epoch archive checkpoints (Lightning's implicit ``ModelCheckpoint`` keeps only the
        latest file under ``lightning_logs/`` and deletes the previous one), EMA, COCO evaluation, best-model selection,
        early stopping, ``max_epochs=train_config.epochs``, the resolved AMP precision, and
        ``default_root_dir=train_config.output_dir``.

        Only ``trainer:`` values that differ from the ``Trainer`` signature defaults are forwarded as overrides, so unset
        keys fall back to what ``build_trainer`` derives from ``train_config``. Callbacks given on the CLI (and the
        ``SaveConfigCallback``) are appended to the RF-DETR callback stack instead of replacing it.

        Args:
            config: Parsed ``trainer:`` section.
            callbacks: Extra callbacks registered through the parser's callback keys.

        Returns:
            The configured ``Trainer``.
        """
        defaults = inspect.signature(Trainer.__init__).parameters
        overrides = {key: value for key, value in config.items() if key not in defaults or value != defaults[key].default}

        user_callbacks = overrides.pop("callbacks", None) or []
        extra_callbacks = list(user_callbacks) if isinstance(user_callbacks, list) else [user_callbacks]
        extra_callbacks.extend(callbacks)
        default_callbacks = self.trainer_defaults.get("callbacks", [])
        extra_callbacks += default_callbacks if isinstance(default_callbacks, list) else [default_callbacks]
        if self.save_config_callback and not config.get("fast_dev_run", False):
            extra_callbacks.append(
                self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
            )

        trainer = build_trainer(
            self.model.train_config,
            self.model.model_config,
            include_training_callbacks=self.subcommand in (None, "fit"),
            **overrides,
        )
        # PTL re-sorts the callback list at fit time (checkpoint callbacks last), so appending here is safe.
        trainer.callbacks.extend(extra_callbacks)
        return trainer


def main() -> None:
    """Entry point for the ``rfdetr`` CLI.

    Constructs and runs ``RFDETRCli`` with ``RFDETRModelModule`` and ``RFDETRDataModule``.  Subcommands are auto-
    generated by ``LightningCLI``: ``fit``, ``validate``, ``test``, ``predict``.
    """
    set_parsing_settings(subclasses_enabled=[ModelConfig, TrainConfig])
    # build_trainer logs into train_config.output_dir, which is reused across resumes, so the saved config.yaml must be
    # allowed to overwrite the previous run's copy instead of aborting the job.
    RFDETRCli(
        RFDETRModelModule,
        RFDETRDataModule,
        parser_kwargs=_PARSER_KWARGS,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
