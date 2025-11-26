from __future__ import annotations

import os
from enum import Enum


class HostRole(str, Enum):
    """Logical role of the current host.

    Even though everything initially runs on a *single* machine, we keep the
    roles explicit so that later, when you scale out, you can adjust behavior
    based on an environment variable instead of rewriting logic.
    """

    SINGLE = "single"
    BLENDER = "blender"
    ORCHESTRATOR = "orchestrator"
    CUBB = "cubb"
    RU_EMU = "ru_emu"


def current_role() -> HostRole:
    """Return the current host role based on the UAV_ACAR_HOST_ROLE env var.

    Defaults to HostRole.SINGLE.
    """
    value = os.getenv("UAV_ACAR_HOST_ROLE", HostRole.SINGLE.value)
    try:
        return HostRole(value)
    except ValueError:
        return HostRole.SINGLE
