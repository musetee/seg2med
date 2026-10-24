"""Dataset registration and lightweight discovery for the generation runner."""

import ast
from pathlib import Path

DATASET_REGISTRY = {}
_REPLACED = set()  # names explicitly overridden with replace=True
_CORE_MODULE = "seg2med_train.Dataloader"


def available_dataset_names():
    """Read active loader registrations without importing MONAI or PyTorch.

    Dataloader.py is the active loader module used by the generation runner; loaders registered by
    extension packages that are already imported are listed too.
    """
    return sorted(set(_declared_in_dataloader()) | set(DATASET_REGISTRY))


def _declared_in_dataloader():
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
    return names


def get_dataset_loader(name):
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "(none registered)"
        raise NotImplementedError(
            f"Dataset loader '{name}' not found. Available dataset loaders: {available}. "
            "Run 'python -m seg2med_train.train --list-datasets' to inspect the active loaders."
        ) from None


def register_dataset(name, *, replace=False):
    """Decorator registering a dataset loader ``fn(opt, paths)`` under ``name`` (``dataset_name`` in the YAML).

    Registering the same function again (also after a module reload: same module and qualified name) is a
    no-op that keeps the newest object. A different function for a registered name, or for a name declared
    by seg2med_train/Dataloader.py that is not imported yet, is rejected unless ``replace=True`` is given
    explicitly, so the import order of extension packages cannot silently change which loader a
    configuration uses. An explicit replacement of a core loader is kept when Dataloader.py is imported later.
    """
    if not isinstance(name, str):
        raise TypeError(f"dataset name must be a string, got {type(name).__name__}")
    if not name or name != name.strip() or any(c.isspace() for c in name):
        raise ValueError(f"dataset name must be a non-empty name without whitespace: {name!r}")

    def decorator(fn):
        if not callable(fn):
            raise TypeError(f"dataset loader for '{name}' must be callable")
        current = DATASET_REGISTRY.get(name)
        module = getattr(fn, "__module__", None)
        same = current is fn or (current is not None and
                                 (getattr(current, "__module__", None), getattr(current, "__qualname__", None)) ==
                                 (module, getattr(fn, "__qualname__", None)))
        if replace:
            _REPLACED.add(name)
        elif current is not None and not same:
            if module == _CORE_MODULE and name in _REPLACED:
                return fn  # the core default yields to an explicit replacement registered earlier
            raise ValueError(f"Dataset loader '{name}' is already registered by "
                             f"{current.__module__}.{current.__qualname__}; pass replace=True to override it.")
        elif current is None and module != _CORE_MODULE and name in _declared_in_dataloader():
            raise ValueError(f"Dataset loader '{name}' is provided by seg2med_train.Dataloader; "
                             "pass replace=True to override it.")
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator
