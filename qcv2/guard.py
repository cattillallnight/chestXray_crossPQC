"""VinDr guard.

Any function that *fits* on data (training, temperature, Platt, prior
correction, BBSE, Saerens-EM) must call assert_not_vindr() on the tensor it
fits. The only exception is the labelled ORACLE rung, which passes
allow_vindr=True explicitly.
"""
from __future__ import annotations

import numpy as np


class VinDrLeakError(RuntimeError):
    pass


def assert_not_vindr(origin: str, allow_vindr: bool = False) -> None:
    """Call with a string tag describing the data origin.

    Every fit passes a literal origin string ('chexpert_val', 'chexpert_train',
    'vindr_bal', ...). If it names vindr and the caller has not opted in, raise.
    """
    if allow_vindr:
        return
    if origin is None:
        raise VinDrLeakError("origin is None; refusing to fit without a data origin tag")
    if "vindr" in str(origin).lower():
        raise VinDrLeakError(
            f"refusing to fit/evaluate-select on data tagged {origin!r}; "
            "only the labelled ORACLE rung may touch VinDr"
        )