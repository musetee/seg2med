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


def _check_name(value, what):
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}")
    if not value or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"{what} must be a non-empty name without whitespace: {value!r}")


def register_model(name, module_name, class_name):
    """Register a model runner from another package, e.g. ``register_model("my_ddpm", "my_pkg.runner", "Runner")``.

    The module is imported only when the model is selected. Re-registering the same target is a no-op;
    a different target for an existing name, or an invalid name/target, is rejected and nothing is registered.
    """
    _check_name(name, "model name")
    _check_name(module_name, "module name")
    _check_name(class_name, "class name")
    if not all(part.isidentifier() for part in module_name.split(".")) or not class_name.isidentifier():
        raise ValueError(f"invalid runner target {module_name}.{class_name}")
    target = (module_name, class_name)
    if MODEL_RUNNERS.get(name, target) != target:
        raise ValueError(f"Model '{name}' is already registered as {MODEL_RUNNERS[name]}.")
    MODEL_RUNNERS[name] = target


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
