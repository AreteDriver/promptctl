"""Feature gating decorators for promptctl."""

from __future__ import annotations

import functools
from typing import Any

from promptctl.exceptions import LicenseError
from promptctl.licensing import Tier, get_license

_ENV_VAR = "PROMPTCTL_LICENSE"


def require_pro(feature: str):
    """Decorator that gates a function behind Pro tier."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            info = get_license()
            if info.tier != Tier.PRO:
                raise LicenseError(
                    f"'{feature}' requires a Pro license. Set {_ENV_VAR} environment variable."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
