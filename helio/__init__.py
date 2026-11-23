__version__ = "2.6.0"
__app_name__ = "Helio"

from helio.utils.file_utils import _helio_excepthook
import sys
sys.excepthook = _helio_excepthook

__all__ = ["__version__", "__app_name__"]
