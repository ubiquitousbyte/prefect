class NomadError(Exception):
    """Base exception for Nomad-related errors."""

    pass


class NomadJobRegistrationError(NomadError):
    """Raised when a Nomad job fails to register with the cluster."""

    pass


class NomadJobTimeoutError(NomadError):
    """Raised when a Nomad job exceeds the configured timeout."""

    pass


class NomadJobSchedulingError(NomadError):
    """Raised when a Nomad job fails to schedule due to resource constraints."""

    pass


class NomadEvaluationError(NomadError):
    """Raised when a Nomad evaluation fails or is canceled."""

    pass


class NomadJobStopError(NomadError):
    """Raised when a Nomad job fails to stop."""

    pass
