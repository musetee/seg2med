"""Supported model runners for the train/inference entry points."""

from importlib import import_module


MODEL_RUNNERS = {
    "AttentionUnet": (".networks.unet.unet", "UNETRunner"),
    "pix2pix": (".networks.gan.cyclegan_and_pix2pix_runner", "CycleGAN_pix2pix_Runner"),
    "cycle_gan": (".networks.gan.cyclegan_and_pix2pix_runner", "CycleGAN_pix2pix_Runner"),
    "ddpm2d_seg2med": (".networks.ddpm.seg2med_ddpm", "DDPM2DSeg2MedRunner"),
    "ddpm2d_seg2med_multimodal": (
        ".networks.ddpm.seg2med_ddpm", "DDPM2DSeg2Med_multimodal_Runner"
    ),
}


def available_model_names():
    return sorted(MODEL_RUNNERS)


def validate_model_name(name):
    if name not in MODEL_RUNNERS:
        available = ", ".join(available_model_names())
        raise ValueError(
            f"Model '{name}' is not supported. Available models: {available}. "
            "Run 'python -m seg2med_train.train --list-models' to inspect supported models."
        )


def get_model_runner(name):
    validate_model_name(name)
    module_name, class_name = MODEL_RUNNERS[name]
    return getattr(import_module(module_name, package=__package__), class_name)
