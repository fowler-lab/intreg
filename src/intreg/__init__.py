from importlib.metadata import version, PackageNotFoundError

from .intreg import IntReg
from .meintreg import MeIntReg

__all__ = ["IntReg", "MeIntReg"]

try:
    __version__ = version("intreg")
except PackageNotFoundError:
    __version__ = "0.0.0"