"""TorchCode — PyTorch practice engine. Used in Jupyter Notebooks.

Example:
    from torch_judge import status, check

    # View progress for all tasks
    status()

    # After implementing the function, run the judge
    check("relu")
"""

from torch_judge._version import __version__
from torch_judge.engine import check, hint
from torch_judge.progress import status, reset_progress

__all__ = ["__version__", "check", "hint", "status", "reset_progress"]
