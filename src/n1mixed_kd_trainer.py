"""Train N1_MIXED segmentation with relational knowledge distillation.

Training noise is sigma=50/255; sigma=0.3 is used for evaluation.
"""

import argparse
import csv
import glob
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "epochs": 50,
    "batch": 8,
    "imgsz": 640,
    # Historical low learning rate for resumed runs; retain the original value.
    "lr0": 5e-5,
    "lambda_kd": 0.5,
    "mixed_noise_ratio": 0.5,
    "train_noise_sigma": 50 / 255,
}
DINOV2_SIZE = 448
LREF = 0.1
MOMENTUM = 0.937
OPTIMIZER = "auto"
_teacher_cache = None
_runtime_settings = DEFAULTS.copy()
_active_trainer = None


def _capture_sppf_features(_module, _inputs, output):
    """Store SPPF features without a closure that would make checkpoints unpicklable."""
    if _active_trainer is not None:
        _active_trainer._s_feat = output.mean(dim=[-1, -2])


def _repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(config_path: str | None) -> dict:
    settings = DEFAULTS.copy()
    if not config_path:
        return settings
    path = _repository_path(config_path)
    if not path.exists():
        return settings
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    for key in DEFAULTS:
        if key in config:
            settings[key] = config[key]
    return settings


def _find_data_yaml() -> str:
    """Find a data YAML from ISAID_DATA or Kaggle candidates."""
    candidates = []
    if os.environ.get("ISAID_DATA"):
        candidates.append(Path(os.environ["ISAID_DATA"]).expanduser())
    kaggle_input = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("**/data.yaml"))
    kagglehub = Path.home() / ".cache" / "kagglehub" / "datasets"
    if kagglehub.exists():
        candidates.extend(Path(path) for path in glob.glob(str(kagglehub / "**/data.yaml"), recursive=True))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("data.yaml was not found. Pass --data, set ISAID_DATA, or provide a Kaggle dataset.")


def _ensure_nc16(yaml_path: str, project: str) -> str:
    """Write an nc=16 correction below the selected project directory."""
    source = Path(yaml_path).expanduser().resolve()
    with source.open() as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("nc") == 16:
        return str(source)
    data["nc"] = 16
    if data.get("path") and not Path(data["path"]).expanduser().exists():
        data["path"] = str(source.parent)
    fixed = Path(project).expanduser() / "data_nc16.yaml"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    with fixed.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f"[DataYAML] {source} -> {fixed} (nc=16)")
    return str(fixed)


def _reject_completed_resume_checkpoint(weights: str) -> None:
    """Reject final checkpoints, whose resume state Ultralytics has stripped."""
    path = Path(weights).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch", -1) < 0 and checkpoint.get("optimizer") is None:
        raise RuntimeError(
            f"{path} is a completed (stripped) checkpoint and cannot be resumed. "
            "To train further, start a new run with --weights <ckpt> (without --resume)."
        )


def get_teacher(device="cuda"):
    """Load and cache the frozen DINOv2 ViT-B/14 teacher on the training device."""
    global _teacher_cache
    if _teacher_cache is None:
        print("[Teacher] Loading DINOv2 ViT-B/14...")
        _teacher_cache = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        _teacher_cache.eval().to(device)
        for parameter in _teacher_cache.parameters():
            parameter.requires_grad_(False)
    return _teacher_cache


def extract_teacher_cls(teacher, clean_images, block=11):
    """Extract one DINOv2 CLS token per clean image."""
    features = {}
    hook = teacher.blocks[block].register_forward_hook(
        lambda _module, _inputs, output: features.update({block: output[:, 0].detach()})
    )
    with torch.no_grad():
        resized = F.interpolate(clean_images, size=(DINOV2_SIZE, DINOV2_SIZE), mode="bilinear", align_corners=False)
        teacher(resized)
    hook.remove()
    return features[block]


def rkd_loss(student, teacher):
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    student_distances = torch.cdist(student, student)
    teacher_distances = torch.cdist(teacher, teacher)
    return F.smooth_l1_loss(
        student_distances / (student_distances.mean() + 1e-8),
        teacher_distances / (teacher_distances.mean() + 1e-8),
    )


class N1MixedKDTrainer(SegmentationTrainer):
    """Segmentation trainer with mixed input noise and an RKD adapter."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self._teacher = self._adapter = self._s_feat = self._t_feat = self._orig_loss = None
        self._kd_calls = self._kd_applied = 0
        self._epoch_rkd_sum = 0.0
        self._epoch_rkd_applied = 0
        self._sppf_ch = 512
        self._sppf_hook = None
        self.add_callback("on_train_epoch_start", self._reset_epoch_rkd)
        self.add_callback("on_train_epoch_end", self._log_epoch_rkd)

    @staticmethod
    def _reset_epoch_rkd(trainer):
        trainer._epoch_rkd_sum = 0.0
        trainer._epoch_rkd_applied = 0

    @staticmethod
    def _log_epoch_rkd(trainer):
        """Report RKD before lambda scaling without changing Ultralytics loss_items."""
        if getattr(trainer, "rank", -1) not in {-1, 0}:
            return
        applied = trainer._epoch_rkd_applied
        mean = trainer._epoch_rkd_sum / applied if applied else 0.0
        epoch = trainer.epoch + 1
        print(f"[N1Mixed] epoch={epoch} RKD mean (pre-lambda)={mean:.6f}; applied_batches={applied}")
        log_path = Path(trainer.save_dir) / "rkd_log.csv"
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["epoch", "rkd_mean", "applied_batches"])
            writer.writerow([epoch, f"{mean:.10f}", applied])

    def _layers(self):
        """Return the layer list whether or not the model is wrapped (DDP/EMA) — differs across Ultralytics versions."""
        model = getattr(self.model, "module", self.model)
        layers = model.model
        return layers if isinstance(layers, (nn.Sequential, nn.ModuleList)) else layers.model

    def check_resume(self, overrides):
        """Allow a resumed interrupted run to set its new total epoch count."""
        requested_epochs = overrides.get("epochs")
        super().check_resume(overrides)
        if self.resume and requested_epochs is not None:
            self.args.epochs = requested_epochs

    def _find_sppf(self):
        for module in self._layers():
            if module.__class__.__name__ == "SPPF":
                return module
        for index in (9, 10):
            try:
                module = self._layers()[index]
                if hasattr(module, "cv2"):
                    return module
            except (IndexError, AttributeError):
                continue
        raise RuntimeError("N1Mixed could not find the SPPF layer.")

    def _setup_distillation(self):
        """Create the adapter before the optimizer and scheduler are constructed."""
        if self._adapter is not None:
            return
        global _active_trainer
        _active_trainer = self
        self._teacher = get_teacher(self.device)
        sppf = self._find_sppf()
        if hasattr(sppf, "cv2"):
            self._sppf_ch = sppf.cv2.conv.out_channels
        self._sppf_hook = sppf.register_forward_hook(_capture_sppf_features)
        self._adapter = nn.Sequential(
            nn.Linear(self._sppf_ch, 256, bias=False), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Linear(256, 768, bias=False)
        ).to(self.device)
        sidecar = str(self.args.model).replace(".pt", "_kd_adapter.pt")
        if self.args.resume:
            try:
                self._adapter.load_state_dict(torch.load(sidecar, map_location="cpu", weights_only=False), strict=True)
                print(f"[N1Mixed] adapter loaded from {sidecar}")
            except Exception as error:
                raise RuntimeError(f"Resume requires KD adapter sidecar: {sidecar}. It could not be loaded: {error}") from error
        else:
            print("[N1Mixed] adapter: random initialization")

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Add the KD adapter before Ultralytics creates its scheduler.

        The base trainer calls this while assembling its training pipeline, immediately
        before scheduler creation.  Adding a parameter group afterwards leaves the
        scheduler's per-group lists too short on current Ultralytics/PyTorch.
        """
        self._setup_distillation()
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        # Keep the adapter learning rate identical to the previous post-setup group.
        optimizer.add_param_group({"params": list(self._adapter.parameters()), "lr": self.args.lr0})
        return optimizer

    def _setup_train(self, world_size=0):
        if self.args.resume:
            checkpoint_path = str(self.args.model)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint.get("optimizer") is not None:
                    checkpoint.pop("optimizer", None)
                    checkpoint.pop("opt", None)
                    torch.save(checkpoint, checkpoint_path)
                    print("[N1Mixed] optimizer stripped from checkpoint")
        try:
            super()._setup_train(world_size)
        except TypeError:
            super()._setup_train()
        self._orig_loss = self.model.loss

        def kd_loss(batch, predictions=None):
            loss, loss_items = self._orig_loss(batch, predictions)
            self._kd_calls += 1
            if self._s_feat is not None and self._t_feat is not None:
                term = rkd_loss(self._adapter(self._s_feat), self._t_feat)
                loss = loss + _runtime_settings["lambda_kd"] * term
                self._kd_applied += 1
                self._epoch_rkd_sum += float(term.detach())
                self._epoch_rkd_applied += 1
                if self._kd_applied == 1:
                    print(f"[N1Mixed] RKD loss applied (first batch): {float(term.detach()):.4f} x lambda {_runtime_settings['lambda_kd']}")
            return loss, loss_items

        self.model.loss = kd_loss

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self._teacher is not None:
            clean_images = batch["img"].clone()
            count = max(1, int(clean_images.shape[0] * _runtime_settings["mixed_noise_ratio"]))
            noisy_images = clean_images.clone()
            indexes = torch.randperm(clean_images.shape[0], device=clean_images.device)[:count]
            noise = torch.randn(count, *clean_images.shape[1:], device=clean_images.device) * _runtime_settings["train_noise_sigma"]
            noisy_images[indexes] = torch.clamp(noisy_images[indexes] + noise, 0, 1)
            batch["img"] = noisy_images
            with torch.no_grad():
                self._t_feat = extract_teacher_cls(self._teacher, clean_images)
        return batch

    def save_model(self):
        # Hooks are needed only on the live training model.  Remove them from
        # the live and EMA copies before Ultralytics pickles an EMA checkpoint.
        if self._sppf_hook is not None:
            self._sppf_hook.remove()
            self._sppf_hook = None
        if self.ema is not None:
            for module in self.ema.ema.modules():
                for key, hook in list(module._forward_hooks.items()):
                    if hook is _capture_sppf_features:
                        del module._forward_hooks[key]
        try:
            super().save_model()
            names = ["last.pt", "best.pt"]
            if self.save_period > 0 and self.epoch % self.save_period == 0:
                names.append(f"epoch{self.epoch}.pt")
            for name in names:
                checkpoint = self.save_dir / "weights" / name
                if checkpoint.exists() and self._adapter is not None:
                    torch.save(self._adapter.state_dict(), str(checkpoint).replace(".pt", "_kd_adapter.pt"))
        except Exception as error:
            print(f"[N1Mixed] adapter sidecar could not be saved: {error}")
        finally:
            # Reattach for the next epoch after every checkpoint save.
            self._sppf_hook = self._find_sppf().register_forward_hook(_capture_sppf_features)


def run_n1mixed(**overrides):
    """Run training from a notebook or another Python caller."""
    settings = _load_config(overrides.pop("config", "configs/n1mixed.yaml"))
    explicit_epochs = overrides.get("epochs") is not None
    settings.update({key: value for key, value in overrides.items() if value is not None})
    global _runtime_settings
    _runtime_settings = {key: settings[key] for key in DEFAULTS if key in settings}
    project = str(settings.pop("project", "runs/n1mixed"))
    data_yaml = settings.pop("data", None) or _find_data_yaml()
    data_yaml = _ensure_nc16(data_yaml, project)
    weights = settings.pop("weights", "yolov8s-seg.pt")
    resume = bool(settings.pop("resume", False))
    name = settings.pop("name", "N1_MIXED")
    backup_dir = settings.pop("backup_dir", None)
    print(f"[N1Mixed] data={data_yaml} resume={resume} epochs={settings['epochs']} batch={settings['batch']}")
    if resume and not Path(weights).expanduser().exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {weights}")
    if resume:
        _reject_completed_resume_checkpoint(weights)
        sidecar = str(weights).replace(".pt", "_kd_adapter.pt")
        if not Path(sidecar).exists():
            raise FileNotFoundError(f"Resume requires KD adapter sidecar; expected path: {sidecar}")
    if _runtime_settings["lambda_kd"] > 0 and settings["batch"] < 3:
        print("[N1Mixed] WARNING: batch < 3 makes distance-wise RKD identically zero after mean-distance normalization.")
    trainer_settings = {key: value for key, value in settings.items() if key not in {"lambda_kd", "mixed_noise_ratio", "train_noise_sigma"}}
    if resume and not explicit_epochs:
        trainer_settings.pop("epochs", None)  # keep the checkpoint's target epochs unless --epochs is given
    model = YOLO(weights)
    results = model.train(
        # Keep the checkpoint path so Ultralytics restores the interrupted run's
        # model/epoch state; completed stripped checkpoints are rejected above.
        data=data_yaml, project=project, name=name, exist_ok=True, resume=weights if resume else False,
        lrf=LREF, momentum=MOMENTUM, optimizer=OPTIMIZER, trainer=N1MixedKDTrainer,
        verbose=True, plots=False, **trainer_settings,
    )
    trainer = model.trainer
    print(f"[N1Mixed] RKD applied on {trainer._kd_applied}/{trainer._kd_calls} loss calls")
    if _runtime_settings["lambda_kd"] > 0 and trainer._kd_applied == 0:
        raise RuntimeError("RKD loss was never applied (feature hooks did not fire); this run is plain YOLO training.")
    best_pt = str(results.save_dir / "weights" / "best.pt")
    if backup_dir:
        destination = Path(backup_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, destination / Path(best_pt).name)
        print(f"[N1Mixed] backup={destination / Path(best_pt).name}")
    return best_pt


def parse_args():
    parser = argparse.ArgumentParser(description="Train N1_MIXED RKD segmentation.")
    parser.add_argument("--data", help="Path to data.yaml (otherwise ISAID_DATA or dataset discovery).")
    parser.add_argument("--weights", default="yolov8s-seg.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from --weights.")
    parser.add_argument("--project", default="runs/n1mixed")
    parser.add_argument("--name", default="N1_MIXED")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lambda-kd", dest="lambda_kd", type=float)
    parser.add_argument("--mixed-ratio", dest="mixed_noise_ratio", type=float)
    parser.add_argument("--noise-sigma", dest="train_noise_sigma", type=float)
    parser.add_argument("--config", default="configs/n1mixed.yaml")
    parser.add_argument("--backup-dir")
    parser.add_argument("--device", help="Training device passed to Ultralytics, e.g. 0, cpu, mps (default: auto).")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without training.")
    return parser.parse_args()


def main():
    args = vars(parse_args())
    dry_run = args.pop("dry_run")
    config = args.get("config")
    settings = _load_config(config)
    settings.update({key: value for key, value in args.items() if value is not None and key != "config"})
    data_yaml = settings.get("data") or _find_data_yaml()
    if dry_run:
        print(yaml.safe_dump(settings, sort_keys=True).strip())
        print(f"data: {Path(data_yaml).expanduser().resolve()}")
        return
    run_n1mixed(**args)


if __name__ == "__main__":
    main()
