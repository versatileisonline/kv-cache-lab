"""KV-cache method registry: parameter validation and cache construction per method."""


def _number(value, low, high):
    return type(value) in (int, float) and low < value <= high


def _no_params(params):
    if params:
        raise ValueError("fullkv takes no parameters.")


def _h2o_params(params):
    if set(params) != {"budget_fraction", "recent_fraction"}:
        raise ValueError("h2o requires exactly budget_fraction and recent_fraction.")
    if not _number(params["budget_fraction"], 0, 1):
        raise ValueError("h2o budget_fraction must be in (0, 1].")
    if not _number(params["recent_fraction"], 0, 1):
        raise ValueError("h2o recent_fraction must be in (0, 1].")


def _kivi_params(params):
    if set(params) != {"bits", "group_size", "residual_length"}:
        raise ValueError("kivi requires exactly bits, group_size, and residual_length.")
    if params["bits"] not in (2, 4) or type(params["bits"]) is not int:
        raise ValueError("kivi bits must be 2 or 4.")
    for key in ("group_size", "residual_length"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"kivi {key} must be a positive integer.")


def _dynamic_cache(model, params):
    from transformers import DynamicCache
    return DynamicCache()


# new_cache is None until a method is implemented; its parameters still validate.
METHODS = {
    "fullkv": {"validate": _no_params, "new_cache": _dynamic_cache},
    "h2o": {"validate": _h2o_params, "new_cache": None},
    "kivi": {"validate": _kivi_params, "new_cache": None},
}


def validate_method(method):
    if not isinstance(method, dict) or set(method) != {"name", "params"}:
        raise ValueError("method must be an object with name and params.")
    entry = METHODS.get(method["name"])
    if entry is None:
        raise ValueError(f"Unknown method {method['name']!r}; choose from {sorted(METHODS)}.")
    if not isinstance(method["params"], dict):
        raise ValueError("method.params must be an object.")
    entry["validate"](method["params"])
    if entry["new_cache"] is None:
        raise ValueError(f"Method {method['name']!r} is specified but not implemented yet.")


def cache_factory(method, model):
    """Return a zero-argument function creating a fresh cache for one request."""
    new_cache = METHODS[method["name"]]["new_cache"]
    return lambda: new_cache(model, method["params"])
