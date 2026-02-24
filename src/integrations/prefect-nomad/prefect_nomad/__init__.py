from . import _version

__version__ = _version.__version__


from .credentials import NomadCredentials
from .worker import NomadWorker
