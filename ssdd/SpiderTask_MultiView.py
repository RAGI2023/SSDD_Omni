# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SpiderTask for Multi-View 360° panorama reconstruction.

Key differences from original SpiderTask:
1. Uses EquiDataset instead of ImageNet
2. Uses SSDDMultiView instead of SSDD
3. Handles multi-view inputs [B, N_views, 3, H, W] and panorama outputs [B, 3, H_pano, W_pano]
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch._dynamo.config
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm

from .dataset_equi import load_equirect
from .log.loggers import MetricLogger
from .models.blocks.ema import EMA, EMAWrapper
from .models.ssdd.losses import GanLoss, SSDDLosses
from .models.ssdd.ssdd import SSDD
from .models.ssdd.ssdd_multiview import SSDDMultiView
from .mutils.main_utils import (
    TaskState,
    UpAccelerator,
    ensure_path,
    split_dict,
)
from .mutils.torch_utils import count_parameters, freeze_model, reproducible_rand, unwrap
from .mutils.train_utils import (
    aggregate_losses,
    auto_compile,
    build_optimizer,
    load_training_state,
    save_training_state,
)

####################################################################
# Common initialization
####################################################################


class SpiderTasksMultiView:
    SHOW_MODEL_PARTS = ["encoder", "decoder", "fusion"]

    def __init__(self, cfg):
        self.state = TaskState(cfg=cfg)
        self.optimizer = None
        self.gan_optimizer = None
        self.training = False

        self.setup_job_env()
        self.setup()

        self.load_data()
        self.load_models()
        self.show_model()

        if self.training:
            self.task_train_prepare()
            # Load training state once model & optimizer are ready
            load_training_state(self.state, self.cfg.checkpoint_path)

        self.accelerator.wait_for_everyone()

    @property
    def cfg(self):
        return self.state.cfg

    @property
    def accelerator(self):
        return self.state.accelerator

    @property
    def models(self):
        return self.state.models

    @property
    def logger(self):
        return self.state.logger

    def print(self, *args, **kwargs):
        self.state.accelerator.print(*args, **kwargs)

    def run(self, task_name=None):
        task_name = task_name or self.cfg.task
        method_name = f"task_{task_name}"
        if hasattr(self, method_name):
            return self.__getattribute__(method_name)()
        else:
            raise ValueError(f"Run function {method_name} for task {task_name} inside {self.__class__.__name__} not found")

    def __call__(self, task_name=None):
        # Run task
        with self.logger.on_task_run() as task_log:
            set_seed(self.cfg.seed)
            task_result = self.run(task_name)

            # End task
            task_log.results = task_result
        self.accelerator.end_training()

        return task_result

    ##### Setup #####

    def setup_job_env(self):
        """Setup the job environment"""
        # Ensure working inside the run directory
        ensure_path(self.cfg.run_dir)
        os.chdir(self.cfg.run_dir)
        ensure_path(self.cfg.cache_dir)
        ensure_path(self.cfg.checkpoint_path)

        # Set seed
        set_seed(self.cfg.seed)

        # Set torch configuration
        torch.set_default_dtype(torch.float32)
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.optimize_ddp = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def setup(self):
        # Make working directory
        Path(self.cfg.checkpoint_path).mkdir(parents=True, exist_ok=True)

        # Accelerator
        self.state.accelerator = UpAccelerator(
            kwargs_handlers=[DistributedDataParallelKwargs()],
            gradient_accumulation_steps=self.cfg.training.grad_accumulate,
            step_scheduler_with_optimizer=False,
            mixed_precision=(self.cfg.training.mixed_precision or "no"),
        )

        self.init_state()

        # Logger
        self.state.logger = self.build_task_logger()

        # Accelerate configuration
        get_logger("accelerate.accelerator").setLevel("WARNING")
        get_logger("accelerate.checkpointing").setLevel("WARNING")

    def init_state(self):
        """Initialize the configuration & state with job-specific settings"""
        cfg = self.cfg

        # Initialize state variables
        self.state.num_processes = self.accelerator.num_processes
        self.accelerator.register_for_checkpointing(self.state)
        self.state.cur_epoch = 0
        self.state.cur_steps = 0

        self.opti_models = []  # pylint: disable=W0201
        self.training = cfg.task.endswith("train")  # pylint: disable=W0201
        self.checkpoint_path = Path(cfg.checkpoint_path)  # pylint: disable=W0201

    ##### Build modules #####

    def load_data(self):
        # Load multi-view dataset
        cfg_copy = deepcopy(self.cfg.dataset)
        cfg_copy['return_all_views'] = True  # Enable multi-view mode

        (train_dataset, test_dataset), (self.train_loader, self.test_loader) = load_equirect(cfg_copy)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare_test_data(self.test_loader)

        self.print("Loaded EquiDataset (multi-view):", {"train": train_dataset, "test": test_dataset})

    def prepare_model(
        self,
        model: torch,
        *,
        name: str,
        compile: bool = False,
        ema: Optional[Mapping] = None,
        freeze: Optional[bool] = None,
        remove_from_checkpointing: bool = False,
        **kwargs,
    ):
        self.state.models[name] = model

        # EMA
        if ema is not None:
            model = EMAWrapper(model, **ema)
            self.state.models[name] = model

        # Freeze
        if freeze:
            freeze_model(model)

        # Register checkpointing
        if not remove_from_checkpointing:
            self.accelerator.register_for_checkpointing(model)

        # Torch.compile
        if compile and self.training:
            model = auto_compile(compile, model)
            self.state.models[name] = model

        # Accelerate prepare (distributed)
        model = self.accelerator.prepare(model)
        self.state.models[name] = model

        return model

    def build_model(self, maker, **kwargs):
        prep_args = ["name", "compile", "ema", "freeze", "model_init", "remove_from_checkpointing"]
        prep_kwargs, module_kwargs = split_dict(kwargs, prep_args)
        model = maker(**module_kwargs)
        return self.prepare_model(model, **prep_kwargs)

    def load_models(self):
        # Build multi-view SSDD model
        self.build_model(SSDDMultiView, name="ae", **self.cfg.ssdd)

        if self.training:
            # Build auxiliary losses (uses base_ssdd from multi-view model)
            base_ssdd = unwrap(self.models["ae"], unw_ema=True).base_ssdd
            self.build_model(
                SSDDLosses,
                name="aux_losses",
                **self.cfg.aux_losses,
                ae=base_ssdd,  # Use base SSDD for aux losses
                accelerator=self.accelerator,
                checkpoint=self.cfg.ssdd.checkpoint
            )

            if self.cfg.get("gan", None) is not None:
                model_gan = GanLoss(**self.cfg.gan)
                self.prepare_model(model_gan, name="gan")

            if self.cfg.distill_teacher:
                # Build teacher model (standard SSDD, not multi-view)
                # Teacher works in panorama space directly
                # Filter out multi-view specific parameters that SSDD doesn't support
                multiview_params = ['n_views', 'fusion_type', 'fusion_hidden_dim',
                                    'use_view_encoding', 'view_encoding_type']
                teacher_cfg = {k: v for k, v in self.cfg.ssdd.items() if k not in multiview_params}
                self.build_model(SSDD, name="teacher", **teacher_cfg, remove_from_checkpointing=True)
                freeze_model(self.models["teacher"])
                self.models["teacher"].eval()

    def task_train_prepare(self):
        self.optimizer = build_optimizer([self.models["ae"], self.models["aux_losses"]], self.cfg.training.lr, self.cfg.training.weight_decay)
        self.print(f"Optimizer for autoencoder: {self.optimizer}")
        self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.cfg.get("gan", None) is not None:
            self.gan_optimizer = build_optimizer([self.models["gan"]], self.cfg.training.lr, self.cfg.training.weight_decay)
            self.print(f"Optimizer for GAN: {self.gan_optimizer}")
            self.gan_optimizer = self.accelerator.prepare(self.gan_optimizer)

    ##### Logging #####

    def build_task_logger(self):
        return MetricLogger(self.state, train=self.training)

    def _show_model_subparameter_count(self, model, recursive_on=None, depth=-1, name=None):
        if depth == -1:
            model = unwrap(model, unw_ema=True)
            name = name or "model"
            self.print(f"{name} parameters count:")
            self.print(f"Total: #{count_parameters(model)}   (trainable: #{count_parameters(model, trainable=True)})")
            self._show_model_subparameter_count(model, recursive_on, depth=0, name=name)
        else:
            for name, m in unwrap(model).named_children():
                self.print("     " * depth + f"- {name}: #{count_parameters(m)}   (trainable: #{count_parameters(m, trainable=True)})")
                if recursive_on and name in recursive_on:
                    sub_rec = [n[len(name) + 1 :] if n.startswith(name + ".") else n for n in recursive_on]
                    self._show_model_subparameter_count(m, sub_rec, depth=depth + 1, name=name)

    def show_model(self):
        for m_name, m in self.models.items():
            if self.SHOW_MODEL_PARTS not in [None, False]:
                self._show_model_subparameter_count(m, recursive_on=self.SHOW_MODEL_PARTS, name=m_name)
            self.print(f"{m_name}:", m)

    ##### Training (task_train) #####

    def set_train_state(self, is_training=True):
        for m in self.models.values():
            m.train(is_training)

        optimizers = [self.optimizer, self.gan_optimizer]
        for opt in optimizers:
            if opt is not None and hasattr(opt, "train"):
                if is_training:
                    opt.train()
                else:
                    opt.eval()

    def _compute_train_loss(self, batch, train_ctx):
        views, panorama = batch  # views: [B, N_views, 3, H, W], panorama: [B, 3, H_pano, W_pano]
        target_panorama = None

        # Train SSDD: main loss & predict panorama
        if "teacher" in self.models:
            # Distillation mode: Teacher performs multi-step inference, Student learns 1-step
            with torch.no_grad():
                self.models["teacher"].eval()
                # Teacher (pre-trained SSDDMultiView) encodes views and performs multi-step decoding
                target_panorama, z, noise = self.models["teacher"](
                    gt_views=views,          # Encode content from views
                    gt_panorama=panorama,    # Only for determining output shape
                    as_teacher=True
                )
                # target_panorama: Teacher's multi-step reconstruction
                # z: Teacher's fused latent from views
                # noise: Noise used by Teacher

            # Student learns to achieve same result in 1 step
            ssdd_out = self.models["ae"](
                gt_views=views,              # Same views input
                gt_panorama=target_panorama, # Learn to match Teacher's output
                z=z,                         # Use Teacher's fused latent
                noise=noise,                 # Share Teacher's noise
                from_noise=True              # Train with t=1.0 (one-step distillation)
            )
        else:
            # Normal training: learn to reconstruct GT panorama
            ssdd_out = self.models["ae"](views, gt_panorama=panorama)

        losses = ssdd_out.losses

        # Add auxiliary losses (computed on panorama output)
        aux_losses = self.models["aux_losses"](panorama, ssdd_out.x0_pred, target_x=target_panorama)
        losses.update(aux_losses)

        # Add GAN losses
        if "gan" in self.models:
            losses.update(
                self.models["gan"](
                    x_gt=panorama if target_panorama is None else target_panorama,
                    x_pred=ssdd_out.x0_pred,
                    xt=ssdd_out.xt,
                    t=ssdd_out.t,
                    existing_losses=losses,
                    n_train_steps=train_ctx["cur_steps"],
                    step="disc_loss",
                )
            )
            train_ctx["gan_ctx"] = {
                "x_pred": ssdd_out.x0_pred,
                "xt": ssdd_out.xt,
                "t": ssdd_out.t,
            }

        return losses

    def _compute_train_gan_loss(self, batch, train_ctx):
        views, panorama = batch
        return self.models["gan"](
            x_gt=panorama,
            **train_ctx["gan_ctx"],
            n_train_steps=train_ctx["cur_steps"],
            step="train",
        )

    def _train_do_step(self, optimizer: torch.optim.Optimizer, batch: Any, train_ctx: Dict[str, Any], step_gan=False):
        acc = self.accelerator

        if step_gan:
            models = [self.models["gan"]]
            with acc.autocast():  # pylint: disable=no-member
                losses = self._compute_train_gan_loss(batch, train_ctx)
        else:
            models = [self.models["ae"], self.models["aux_losses"]]
            with acc.autocast():  # pylint: disable=no-member
                losses = self._compute_train_loss(batch, train_ctx)

        assert isinstance(losses, dict) and all(isinstance(v, torch.Tensor) for v in losses.values()), f"Losses should be a dict of tensors, got {losses}"
        assert len(losses) > 0, "No loss returned"

        sum_loss, losses = aggregate_losses(self.cfg, losses)

        acc.backward(sum_loss)

        if self.cfg.training.grad_clip and acc.sync_gradients:
            for m in models:
                acc.clip_grad_norm_(m.parameters(), self.cfg.training.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return {k: v.detach() for k, v in losses.items()}

    def task_train(self):
        cfg = self.cfg
        self.set_train_state(True)

        for cur_epoch in range(self.state.cur_epoch, cfg.training.epochs):
            self.state.cur_epoch = cur_epoch

            # Train epoch
            with self.logger.on_epoch(self.train_loader):
                for i_batch, batch in enumerate(self.train_loader):
                    with self.logger.on_batch(i_batch) as batch_log:
                        train_ctx = {
                            "losses": batch_log.losses,
                            "i_batch": i_batch,
                            "cur_epoch": self.state.cur_epoch,
                            "cur_steps": self.state.cur_steps,
                        }

                        # AE step
                        with self.accelerator.accumulate(*[self.models["ae"], self.models["aux_losses"]]):
                            batch_log.losses = self._train_do_step(self.optimizer, batch, train_ctx)

                        # EMA update
                        if EMA.uses_ema(self.models["ae"]):
                            self.set_train_state(False)
                            EMA.update_ema_modules(self.models["ae"])
                            self.set_train_state(True)

                        # GAN step
                        if "gan" in self.models:
                            with self.accelerator.accumulate(self.models["gan"]):
                                gan_losses = self._train_do_step(self.gan_optimizer, batch, train_ctx, step_gan=True)
                                batch_log.losses.update(gan_losses)

                        self.accelerator.wait_for_everyone()
                        self.state.cur_steps += 1

            # Eval & checkpoint
            if (cur_epoch + 1) % cfg.training.eval_freq == 0:
                self.task_eval()
                save_training_state(self.state, self.checkpoint_path)

            self.accelerator.wait_for_everyone()

    ##### Evaluation (task_eval) #####

    def task_eval(self):
        acc = self.accelerator
        self.set_train_state(False)
        self.generator = torch.Generator(device=acc.device)
        self.generator.manual_seed(self.cfg.seed)

        with self.logger.on_eval(self.test_loader) as eval_log:
            last_views = None
            last_panorama = None
            last_rec = None

            for batch in tqdm(self.test_loader, desc="Eval", disable=not acc.is_main_process):
                views, panorama = batch  # [B, N_views, 3, H, W], [B, 3, H_pano, W_pano]

                with torch.no_grad(), acc.autocast():
                    # Generate reproducible noise for evaluation consistency (panorama shape, not view shape!)
                    noise = reproducible_rand(acc, self.generator, panorama.shape)
                    # Determine steps based on teacher mode
                    steps = 1 if "teacher" in self.models else self.cfg.ssdd.fm_sampler.steps
                    # Forward through multi-view model
                    rec_panorama = self.models["ae"](views, gt_panorama=panorama, noise=noise, steps=steps)

                # Update metrics (normalize to [0, 1] from [-1, 1])
                rgb_panorama = self.to_rgb(panorama)
                rgb_rec_panorama = self.to_rgb(rec_panorama)
                self.logger.metrics.update(x_gt=rgb_panorama, x_pred=rgb_rec_panorama)

                # Keep last batch for visualization
                last_views = views
                last_panorama = panorama
                last_rec = rec_panorama

            # Store samples for visualization (input views + GT panorama + predicted panorama)
            if self.cfg.show_samples and acc.is_main_process:
                n_samples = self.cfg.show_samples
                eval_log.input_views = last_views[:n_samples]  # [N, N_views, 3, H, W]
                eval_log.gt_samples = self.to_rgb(last_panorama[:n_samples])  # [N, 3, H_pano, W_pano]
                eval_log.rec_samples = self.to_rgb(last_rec[:n_samples])  # [N, 3, H_pano, W_pano]

        if self.training:
            self.set_train_state(True)

        return deepcopy(self.logger.metrics.last_m_vals)

    ##### Utils #####

    def to_rgb(self, x):  # x in [-1;1] ; output will be in [0;1]
        assert x.ndim == 4
        return torch.clamp(255 * (x + 1) / 2, 0, 255).round() / 255
