"""Shared reproducible denoising validation, DDIM previews, and training state.

Full-volume sampling/export remains in seg2med_ddpm.py for explicit evaluation.
Validation and previews use separate persisted random experiments.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from itertools import islice

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.networks.schedulers import DDIMScheduler
from PIL import Image

logger = logging.getLogger(__name__)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@contextmanager
def evaluation(model):
    state, was_training = rng_state(), model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)
        restore_rng(state)


def unwrap(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def load_checkpoint(path, device="cpu"):
    # Checkpoints contain optimizer/RNG Python state; load only trusted files.
    return torch.load(path, map_location=device, weights_only=False)


def _item(batch, key, index, default=None):
    value = batch.get(key)
    if value is None:
        return default
    value = value[index]
    return value.item() if torch.is_tensor(value) and value.numel() == 1 else value


def samples(loader, keys):
    """Yield stable slices without an eager full-validation-set tensor cache."""
    for volume in loader:
        source, target = volume[keys[0]], volume[keys[1]]
        if source.ndim not in (4, 5) or source.shape != target.shape:
            raise ValueError("DDPM2D expects aligned (B,C,H,W[,D]) source/target tensors.")
        for index in range(source.shape[0]):
            base = _item(volume, "sample_id", index)
            if base is None:
                raise ValueError("DDPM training requires stable sample_id metadata from the dataset loader.")
            depth = source.shape[-1] if source.ndim == 5 else 1
            for z in range(depth):
                yield {"source": source[index, ..., z] if source.ndim == 5 else source[index],
                       "target": target[index, ..., z] if target.ndim == 5 else target[index],
                       "sample_id": f"{base}:z{z:06d}",
                       "patient_key": str(_item(volume, "patient_key", index, _item(volume, "patient_ID", index))),
                       "modality": int(_item(volume, "modality", index)),
                       "Aorta_diss": float(_item(volume, "Aorta_diss", index, 0))}


def batches(loader, keys, batch_size, *, shuffle=False):
    # Preserve the existing four-volume training groups, allowing multimodal
    # auxiliary losses to see mixed modalities without caching the full dataset.
    iterator = iter(loader)
    while True:
        volumes = list(islice(iterator, 4 if shuffle else 1))
        if not volumes:
            return
        slices = list(samples(volumes, keys))
        if shuffle:
            order = torch.randperm(len(slices)).tolist()
            slices = [slices[index] for index in order]
        for start in range(0, len(slices), batch_size):
            yield slices[start:start + batch_size]


def tensors(rows, owner):
    source = torch.stack([torch.as_tensor(row["source"]) for row in rows]).to(owner.device)
    target = torch.stack([torch.as_tensor(row["target"]) for row in rows]).to(owner.device)
    # Avoid propagating MONAI metadata into scheduler operations/checkpoints.
    source = source.as_tensor() if hasattr(source, "as_tensor") else source
    target = target.as_tensor() if hasattr(target, "as_tensor") else target
    batch = {"modality": torch.tensor([row["modality"] for row in rows], device=owner.device),
             "Aorta_diss": torch.tensor([row["Aorta_diss"] for row in rows], device=owner.device)}
    condition = owner._get_modality_condition(batch) if owner.with_conditioning else None
    return owner.prepare_model_input(source), target, batch, condition


def training_batches(owner):
    """Synchronize uneven slice counts without divergent DDP optimizer states.

    Exhausted ranks perform a zero-weight backward pass on their last batch;
    every optimizer takes the same steps. Real gradients are sample-weighted.
    """
    iterator = iter(batches(owner.train_loader, owner.keys, owner.config.dataset.batch_size, shuffle=True))
    previous = None
    while True:
        rows = next(iterator, None)
        local_count = len(rows) if rows else 0
        if not owner.is_ddp:
            if rows is None:
                return
            yield rows, local_count, 1.0
            continue
        total_count = torch.tensor(local_count, device=owner.device)
        dist.all_reduce(total_count)
        if not int(total_count):
            return
        if rows:
            previous = rows
        if previous is None:
            raise ValueError("Every distributed rank needs at least one training volume.")
        yield previous, local_count, local_count * owner.world_size / int(total_count)


def noise_target(scheduler, target, noise, timesteps):
    prediction_type = scheduler.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return target
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(target, noise, timesteps)
    raise ValueError(f"Unsupported diffusion prediction_type: {prediction_type}")


class DiffusionTrainer:
    def __init__(self, owner):
        self.owner, self.config = owner, owner.config
        self.logs = Path(owner.paths["saved_logs_folder"])
        self.models = Path(owner.paths["saved_model_folder"])
        self.rank = owner.rank
        self.logs.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.monitor = self.config.validation.monitor
        if self.config.ddpm.noise_type not in {"normal", "uniform"}:
            raise ValueError("ddpm.noise_type must be normal or uniform.")
        valid_metrics = {"val_denoising_mse", "val_patient_denoising_mse", "val_modality_patient_denoising_mse"}
        if self.monitor not in valid_metrics:
            raise ValueError(f"validation.monitor must be one of {sorted(valid_metrics)}")
        self.min_delta = float(self.config.train.earlystopping_delta)
        self.patience = int(self.config.train.earlystopping_patience)
        if self.min_delta < 0 or self.patience < 1 or self.config.train.val_epoch_interval < 1:
            raise ValueError("Invalid early stopping or validation interval settings.")
        options = self.config.train.scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            owner.optimizer, mode="min", factor=options.factor, patience=options.patience,
            threshold=self.min_delta, threshold_mode="abs", min_lr=options.min_lr, cooldown=options.cooldown)
        self.scaler = torch.amp.GradScaler("cuda", enabled=owner.device.type == "cuda")
        self.best, self.bad_checks, self.global_step, self.start_epoch = math.inf, 0, 0, 0
        self.history = []
        self.resume_state = None
        if self.config.train.resume:
            self.resume_state = load_checkpoint(self.config.ckpt_path)
            self._restore()
        self.run_metadata = self._metadata()
        metadata_path = self.logs / "run_metadata.json"
        if metadata_path.exists():
            if not self.config.train.resume:
                raise ValueError("Run directory already contains training metadata; choose a new train.run_dir or resume explicitly.")
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("diffusion", "preprocessing", "validation", "source_manifests"):
                if previous[key] != self.run_metadata[key]:
                    raise ValueError(f"Run metadata changed ({key}); start a fresh run rather than reusing validation experiments.")
        elif self.config.train.resume:
            raise ValueError("Resume requires the original run_metadata.json and validation sampling records.")
        elif self.rank == 0:
            self._json(metadata_path, self.run_metadata)
        with evaluation(owner.model):
            self.validation_records = self._validation_records()
            self.preview_records = self._preview_records()
        if self.resume_state:
            states = self.resume_state["rng_states"]
            if owner.is_ddp and len(states) != owner.world_size:
                raise ValueError("Resume must use the same distributed world size.")
            restore_rng(states[self.rank])
        logger.info("Monitor=%s; minimum absolute improvement=%g; early stopping patience=%d validation checks",
                    self.monitor, self.min_delta, self.patience)

    @staticmethod
    def _json(path, content):
        path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")

    def _metadata(self):
        owner = self.owner
        versions = {}
        for name in ("torch", "monai", "monai-generative", "numpy", "pillow"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unknown"
        import yaml
        config = yaml.safe_load(self.config.dump())
        manifests = {}
        for key in ("train_csv", "val_csv"):
            path = Path(getattr(self.config.dataset, key))
            manifests[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "in-memory-test"
        preprocessing = {key: value for key, value in config["dataset"].items()
                         if key not in {"train_csv", "val_csv", "test_csv", "data_dir", "batch_size", "val_batch_size",
                                        "num_workers", "random_shuffle", "train_number", "val_number"}}
        from ...Dataloader import NormalizePriorTargetd
        normalizer = NormalizePriorTargetd(self.owner.keys[0], self.owner.keys[1])
        return {"diffusion": {"num_train_timesteps": owner.scheduler.num_train_timesteps,
                              "prediction_type": str(owner.scheduler.prediction_type),
                              "noise_type": self.config.ddpm.noise_type,
                              "betas": owner.scheduler.betas.cpu().tolist()},
                "preprocessing": {"dataset": preprocessing, "dataset_loader": config["dataset_name"],
                                  "normalization": {"source": {str(k): list(v) for k, v in normalizer.source_ranges.items()},
                                                    "target": {str(k): list(v) for k, v in normalizer.target_ranges.items()}}},
                "validation": config["validation"], "source_manifests": manifests,
                "software_versions": versions, "configuration": config}

    def _validation_records(self):
        path = self.logs / "validation_sampling.csv"
        repeats = int(self.config.validation.repeats)
        if repeats < 1:
            raise ValueError("validation.repeats must be positive")
        records, ids = [], set()
        for row in samples(self.owner.val_loader, self.owner.keys):
            sample_id = row["sample_id"]
            if sample_id in ids:
                raise ValueError(f"Duplicate validation sample ID: {sample_id}")
            ids.add(sample_id)
            for repeat in range(repeats):
                step_seed = stable_seed(self.config.validation.seed, sample_id, repeat, "timestep")
                timestep = int(torch.randint(self.owner.scheduler.num_train_timesteps, (1,),
                                            generator=torch.Generator().manual_seed(step_seed)).item())
                records.append({"sample_id": sample_id, "patient_key": row["patient_key"],
                                "modality": row["modality"], "timestep": timestep,
                                "noise_seed": stable_seed(self.config.validation.seed, sample_id, repeat, "denoising-noise"),
                                "repeat_id": repeat, "shape": "x".join(map(str, row["target"].shape))})
        if not records:
            raise ValueError("Validation set contains no slices.")
        records.sort(key=lambda record: (record["sample_id"], record["repeat_id"]))
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            if saved != [{key: str(value) for key, value in record.items()} for record in records]:
                raise ValueError("validation_sampling.csv does not match validation IDs, shapes, seed, or diffusion schedule.")
        elif self.config.train.resume:
            raise ValueError("Resume requires validation_sampling.csv from the original run.")
        elif self.rank == 0:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        return {(record["sample_id"], record["repeat_id"]): record for record in records}

    def _preview_records(self):
        options = self.config.validation.preview
        if not options.enabled:
            return []
        if options.sampler != "ddim" or not 1 <= options.num_inference_steps <= self.owner.scheduler.num_train_timesteps:
            raise ValueError("Previews require sampler: ddim and 1..num_train_timesteps inference steps.")
        if options.epoch_interval < 1 or options.num_samples < 1 or options.eta < 0:
            raise ValueError("Invalid preview interval, sample count, or eta.")
        if len(options.display_range) != 2 or options.display_range[0] >= options.display_range[1]:
            raise ValueError("Preview display_range must contain increasing [minimum, maximum].")
        available = [row for (sample, repeat), row in self.validation_records.items() if repeat == 0]
        by_id = {row["sample_id"]: row for row in available}
        if options.sample_ids:
            if len(set(options.sample_ids)) != len(options.sample_ids) or set(options.sample_ids) - set(by_id):
                raise ValueError("Preview sample_ids must be unique IDs from validation_sampling.csv.")
            selected = [by_id[key] for key in options.sample_ids]
        else:
            # Round-robin modalities, preferring unseen patients; use middle
            # slices per patient/modality instead of arbitrary adjacent slices.
            candidates = defaultdict(list)
            for row in available:
                candidates[(row["modality"], row["patient_key"])].append(row)
            pools = defaultdict(list)
            for (modality, patient), rows in sorted(candidates.items()):
                rows = sorted(rows, key=lambda row: row["sample_id"])
                pools[modality].append(rows[len(rows) // 2])
            selected, seen = [], set()
            while pools and len(selected) < options.num_samples:
                for modality in sorted(list(pools)):
                    pool = pools[modality]
                    index = next((i for i, row in enumerate(pool) if row["patient_key"] not in seen), 0)
                    chosen = pool.pop(index)
                    selected.append(chosen)
                    seen.add(chosen["patient_key"])
                    if not pool:
                        del pools[modality]
                    if len(selected) == options.num_samples:
                        break
            selected_ids = {row["sample_id"] for row in selected}
            for row in sorted(available, key=lambda row: (row["modality"], row["sample_id"])):
                if len(selected) >= options.num_samples:
                    break
                if row["sample_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["sample_id"])
        if options.seeds and len(options.seeds) != len(selected):
            raise ValueError("preview.seeds must match the selected preview sample count.")
        records = [{"sample_id": row["sample_id"], "patient_key": row["patient_key"], "modality": row["modality"],
                    "seed": int(options.seeds[index]) if options.seeds else stable_seed(options.seed, row["sample_id"], "preview")}
                   for index, row in enumerate(selected)]
        path = self.logs / "preview_samples.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != records:
                raise ValueError("Fixed preview sample selection changed; use the original run settings.")
        elif self.config.train.resume:
            raise ValueError("Resume requires preview_samples.json from the original run.")
        elif self.rank == 0:
            self._json(path, records)
        return records

    def validate(self):
        owner = self.owner
        values = []
        # All ranks evaluate the complete validation set with an unwrapped
        # model (no DDP buffer collectives or padded validation duplicates).
        with evaluation(owner.model):
            for rows in batches(owner.val_loader, owner.keys, self.config.dataset.val_batch_size):
                source, target, batch, condition = tensors(rows, owner)
                for repeat in range(self.config.validation.repeats):
                    records = [self.validation_records[(row["sample_id"], repeat)] for row in rows]
                    noises = []
                    for row, record in zip(rows, records):
                        generator = torch.Generator().manual_seed(record["noise_seed"])
                        create = torch.rand if self.config.ddpm.noise_type == "uniform" else torch.randn
                        noises.append(create(tuple(row["target"].shape), generator=generator))
                    noise = torch.stack(noises).to(owner.device)
                    timesteps = torch.tensor([record["timestep"] for record in records], device=owner.device)
                    prediction = owner.model_inferer_predict(target, source, unwrap(owner.model), noise, timesteps, condition)
                    expected = noise_target(owner.scheduler, target, noise, timesteps)
                    losses = F.mse_loss(prediction.float(), expected.float(), reduction="none").flatten(1).mean(1)
                    for row, loss in zip(rows, losses.cpu().tolist()):
                        values.append((row["patient_key"], row["modality"], loss))
        patients, modality_patients = defaultdict(list), defaultdict(list)
        for patient, modality, loss in values:
            patients[patient].append(loss)
            modality_patients[(modality, patient)].append(loss)
        modalities = defaultdict(list)
        for (modality, patient), losses in modality_patients.items():
            modalities[modality].append(float(np.mean(losses)))
        metrics = {"val_denoising_mse": float(np.mean([row[2] for row in values])),
                   "val_patient_denoising_mse": float(np.mean([np.mean(losses) for losses in patients.values()])),
                   "val_modality_patient_denoising_mse": float(np.mean([np.mean(losses) for losses in modalities.values()])),
                   "val_samples": len(values), "val_patients": len(patients)}
        for modality, losses in sorted(modalities.items()):
            metrics[f"val_modality_{modality}_patient_denoising_mse"] = float(np.mean(losses))
            rows = [row[2] for row in values if row[1] == modality]
            metrics[f"val_modality_{modality}_denoising_mse"] = float(np.mean(rows))
            metrics[f"val_modality_{modality}_samples"] = len(rows)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite validation metric; refusing checkpoint selection.")
        return metrics

    def _ddim_scheduler(self):
        owner = self.owner
        scheduler = DDIMScheduler(num_train_timesteps=owner.scheduler.num_train_timesteps,
                                  prediction_type=owner.scheduler.prediction_type,
                                  clip_sample=owner.scheduler.clip_sample)
        # Copy the actual training schedule, including any nondefault beta
        # schedule. Never mutate the training DDPM timestep sequence.
        for key in ("betas", "alphas", "alphas_cumprod"):
            setattr(scheduler, key, getattr(owner.scheduler, key).detach().clone().to(owner.device))
        scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(owner.device)
        scheduler.set_timesteps(self.config.validation.preview.num_inference_steps, device=owner.device)
        return scheduler

    def preview(self, epoch):
        if self.rank != 0 or not self.preview_records:
            return {}
        owner, options = self.owner, self.config.validation.preview
        selected = {record["sample_id"]: record for record in self.preview_records}
        folder = Path(owner.paths["saved_img_folder"]) / "previews" / f"epoch_{epoch:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        generated_metrics, metadata = [], []
        with evaluation(owner.model):
            scheduler = self._ddim_scheduler()
            for row in samples(owner.val_loader, owner.keys):
                if row["sample_id"] not in selected:
                    continue
                record = selected[row["sample_id"]]
                source, target, batch, condition = tensors([row], owner)
                generator = torch.Generator().manual_seed(record["seed"])
                image = torch.randn(tuple(target.shape), generator=generator).to(owner.device)
                for timestep in scheduler.timesteps:
                    step = torch.tensor([int(timestep)], device=owner.device)
                    # Prediction adapters add training noise, so call the model
                    # directly here on an already noisy DDIM sample + prior.
                    arguments = {"x": torch.cat([image, source], dim=1), "timesteps": step}
                    if getattr(owner, "class_embed_condition", False):
                        arguments["context" if owner.guide_by_context else "class_labels"] = condition
                    else:
                        arguments["context"] = condition
                    predicted = unwrap(owner.model)(**arguments)
                    image, _ = scheduler.step(predicted, int(timestep), image,
                                              eta=options.eta, generator=generator)
                minimum, maximum = options.display_range
                panels = []
                for tensor in (source, image, target):
                    array = tensor[0, 0].detach().float().cpu().numpy().T
                    panels.append(((np.clip(array, minimum, maximum) - minimum) * 255 / (maximum - minimum)).astype(np.uint8))
                filename = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16] + ".png"
                Image.fromarray(np.concatenate(panels, axis=1)).save(folder / filename)
                mse = float(F.mse_loss(image.float(), target.float()).item())
                generated_metrics.append(mse)
                metadata.append({**record, "png": filename, "preview_generated_image_mse": mse})
        self._json(folder / "preview_metadata.json", {
            "epoch": epoch, "global_step": self.global_step, "sampler": options.sampler,
            "num_inference_steps": options.num_inference_steps, "eta": options.eta,
            "timesteps": scheduler.timesteps.cpu().tolist(), "display_range": list(options.display_range),
            "panel_order": ["conditioning_prior", "generated_image", "real_target"], "samples": metadata})
        return {"preview_generated_image_mse": float(np.mean(generated_metrics))}

    def _restore(self):
        state = self.resume_state
        required = {"optimizer", "lr_scheduler", "scaler", "early_stopping", "rng_states", "configuration", "history"}
        if not required.issubset(state):
            raise ValueError("Checkpoint lacks full resume state. Use train.resume: false for weights-only fine-tuning.")
        old = state["configuration"]
        if old["model_name"] != self.config.model_name or old["validation"]["monitor"] != self.monitor:
            raise ValueError("Cannot change model or monitoring metric during resume.")
        if old["train"]["earlystopping_delta"] != self.min_delta:
            raise ValueError("Cannot change the minimum improvement during resume.")
        if old["train"]["loss"] != self.config.train.loss or old["train"]["scheduler"] != dict(self.config.train.scheduler):
            raise ValueError("Cannot change the training loss or LR scheduler settings during resume.")
        self.owner.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if hasattr(self.owner, "classifier"):
            unwrap(self.owner.classifier).load_state_dict(state["classifier"])
            self.owner.classifier_optimizer.load_state_dict(state["classifier_optimizer"])
        self.start_epoch, self.global_step = state["epoch"], state["global_step"]
        self.best = state["early_stopping"]["best_metric"]
        self.bad_checks = state["early_stopping"]["bad_checks"]
        self.history = state["history"]
        self.resume_state = state

    def checkpoint(self, epoch, *, best=False, periodic=False):
        owner = self.owner
        states = [rng_state()]
        if owner.is_ddp:
            states = [None] * owner.world_size
            dist.all_gather_object(states, rng_state())
        if self.rank != 0:
            return
        import yaml
        state = {"format_version": 2, "model": unwrap(owner.model).state_dict(),
                 "optimizer": owner.optimizer.state_dict(), "opt": owner.optimizer.state_dict(),
                 "lr_scheduler": self.lr_scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": epoch, "global_step": self.global_step, "best_metric": self.best,
                 "early_stopping": {"best_metric": self.best, "bad_checks": self.bad_checks,
                                    "patience": self.patience, "min_delta": self.min_delta},
                 "rng_states": states, "configuration": yaml.safe_load(self.config.dump()),
                 "history": self.history}
        if hasattr(owner, "classifier"):
            state["classifier"] = unwrap(owner.classifier).state_dict()
            state["classifier_optimizer"] = owner.classifier_optimizer.state_dict()
        for filename in ["last.pt"] + (["best.pt"] if best else []) + ([f"model_{epoch}.pt"] if periodic else []):
            temporary = self.models / (filename + ".tmp")
            torch.save(state, temporary)
            temporary.replace(self.models / filename)

    def run(self):
        owner = self.owner
        if self.bad_checks >= self.patience:
            logger.info("Checkpoint has already reached early stopping; start a fresh run to fine-tune.")
            return
        for epoch in range(self.start_epoch + 1, self.config.train.num_epochs + 1):
            owner.epoch = epoch
            owner.model.train()
            if owner.is_ddp and hasattr(owner.train_loader.sampler, "set_epoch"):
                owner.train_loader.sampler.set_epoch(epoch)
            sums, count = np.zeros(2, dtype=np.float64), 0
            for epoch_step, (rows, real_count, gradient_weight) in enumerate(training_batches(owner), start=1):
                source, target, batch, condition = tensors(rows, owner)
                create = torch.rand_like if self.config.ddpm.noise_type == "uniform" else torch.randn_like
                noise = create(target)
                timesteps = torch.randint(owner.scheduler.num_train_timesteps, (len(rows),), device=owner.device)
                expected = noise_target(owner.scheduler, target, noise, timesteps)
                owner.optimizer.zero_grad(set_to_none=True)
                if hasattr(owner, "classifier_optimizer"):
                    owner.classifier_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=owner.device.type, enabled=owner.device.type == "cuda"):
                    predicted = owner.model_inferer_predict(target, source, owner.model, noise, timesteps, condition)
                    loss, mse, auxiliary = owner.loss_calc(predicted, expected, modality=batch["modality"],
                        mode=self.config.train.loss, classifier=getattr(owner, "classifier", None))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; check batch composition for contrastive losses.")
                self.scaler.scale(loss * gradient_weight).backward()
                self.scaler.step(owner.optimizer)
                if hasattr(owner, "classifier_optimizer"):
                    self.scaler.step(owner.classifier_optimizer)
                self.scaler.update()
                count += real_count
                sums += [loss.item() * real_count, mse.item() * real_count]
                self.global_step += 1
                if self.rank == 0:
                    logger.info(
                        "Train | epoch=%d/%d | step=%d | global_step=%d | samples=%d | loss=%.6f | denoising_mse=%.6f | lr=%.3e%s",
                        epoch, self.config.train.num_epochs, epoch_step, self.global_step,
                        real_count, loss.item(), mse.item(), owner.optimizer.param_groups[0]["lr"],
                        " | padding-only batch (other ranks still training)" if not real_count else "",
                    )
            if owner.is_ddp:
                aggregate = torch.tensor([*sums, count], dtype=torch.float64, device=owner.device)
                dist.all_reduce(aggregate)
                sums, count = aggregate[:2].cpu().numpy(), int(aggregate[2])
                step = torch.tensor(self.global_step, device=owner.device)
                dist.all_reduce(step, op=dist.ReduceOp.MAX)
                self.global_step = int(step)
            if count == 0:
                raise ValueError("Training loader produced no samples.")
            record = {"epoch": epoch, "global_step": self.global_step, "train_loss": float(sums[0] / count),
                      "train_denoising_mse": float(sums[1] / count), "train_samples": count,
                      "lr_used": owner.optimizer.param_groups[0]["lr"], "monitor": self.monitor,
                      "minimum_improvement": self.min_delta}
            improved = False
            if epoch % self.config.train.val_epoch_interval == 0:
                record.update(self.validate())
                metric = record[self.monitor]
                improved = metric < self.best - self.min_delta
                if improved:
                    self.best, self.bad_checks = metric, 0
                else:
                    self.bad_checks += 1
                if self.config.train.scheduler.enabled:
                    self.lr_scheduler.step(metric)
            if self.config.validation.preview.enabled and epoch % self.config.validation.preview.epoch_interval == 0:
                record.update(self.preview(epoch))
            record.update(lr=owner.optimizer.param_groups[0]["lr"], bad_validation_checks=self.bad_checks,
                          best_metric=None if not math.isfinite(self.best) else self.best)
            self.history.append(record)
            if self.rank == 0:
                self._json(self.logs / "epoch_metrics.json", self.history)
                fields = list(dict.fromkeys(key for item in self.history for key in item))
                with (self.logs / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(self.history)
                logger.info("Epoch summary | %s", json.dumps(record))
            interval = self.config.train.save_ckpt_interval
            self.checkpoint(epoch, best=improved, periodic=interval > 0 and epoch % interval == 0)
            if self.bad_checks >= self.patience:
                logger.info("Early stopping triggered by validation metric.")
                break
