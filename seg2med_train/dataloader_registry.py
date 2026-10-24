"""Dataset registration and lightweight discovery for the generation runner."""

import ast
from pathlib import Path

DATASET_REGISTRY = {}


def available_dataset_names():
    """Read active loader registrations without importing MONAI or PyTorch.

    Dataloader.py is the active loader module used by the generation runner; loaders registered by
    extension packages that are already imported are listed too.
    """
    source = Path(__file__).with_name("Dataloader.py")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    names = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "register_dataset"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                names.add(decorator.args[0].value)
    return sorted(names | set(DATASET_REGISTRY))


def get_dataset_loader(name):
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "(none registered)"
        raise NotImplementedError(
            f"Dataset loader '{name}' not found. Available dataset loaders: {available}. "
            "Run 'python -m seg2med_train.train --list-datasets' to inspect the active loaders."
        ) from None


def register_dataset(name):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator
