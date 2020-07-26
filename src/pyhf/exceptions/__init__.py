import sys


class Unsupported(Exception):
    """
    Unsupported exceptions are raised when something is requested, that is not supported by the current configuration.
    """


class InvalidMeasurement(Exception):
    """
    InvalidMeasurement is raised when a specified measurement is invalid given the specification.
    """


class InvalidNameReuse(Exception):
    pass


class InvalidSpecification(Exception):
    """
    InvalidSpecification is raised when a specification does not validate against the given schema.
    """

    def __init__(self, ValidationError):
        self.exc_info = sys.exc_info()
        self.parent = ValidationError
        self.path = ''
        for item in ValidationError.path:
            if isinstance(item, int):
                self.path += '[{}]'.format(item)
            else:
                self.path += '.{}'.format(item)
        self.path = self.path.lstrip('.')
        self.instance = ValidationError.instance
        message = '{0}.\n\tPath: {1}\n\tInstance: {2}'.format(
            ValidationError.message, self.path, self.instance
        )
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class InvalidPatchSet(Exception):
    """InvalidPatchSet is raised when a given patchset object does not have the right configuration, even though it validates correctly against the schema."""


class InvalidPatchLookup(Exception):
    """InvalidPatchLookup is raised when the patch lookup from a patchset object has failed"""


class PatchSetVerificationError(Exception):
    """PatchSetVerificationError is raised when the workspace digest does not match the patchset digests as part of the verification procedure"""


class InvalidWorkspaceOperation(Exception):
    """InvalidWorkspaceOperation is raised when an operation on a workspace fails."""


class UnspecifiedPOI(Exception):
    """
    UnspecifiedPOI is raised when a given model does not have POI(s) defined but is used in contexts that need it.

    This can occur when e.g. trying to calculate CLs on a POI-less model.
    """


class InvalidModel(Exception):
    """
    InvalidModel is raised when a given model does not have the right configuration, even though it validates correctly against the schema.

    This can occur, for example, when the provided parameter of interest to fit against does not get declared in the specification provided.
    """


class InvalidModifier(Exception):
    """
    InvalidModifier is raised when an invalid modifier is requested. This includes:

        - creating a custom modifier with the wrong structure
        - initializing a modifier that does not exist, or has not been loaded

    """


class InvalidInterpCode(Exception):
    """
    InvalidInterpCode is raised when an invalid/unimplemented interpolation code is requested.
    """


class ImportBackendError(Exception):
    """
    MissingLibraries is raised when something is imported by sustained an import error due to missing additional, non-default libraries.
    """


class InvalidBackend(Exception):
    """
    InvalidBackend is raised when trying to set a backend that does not exist.
    """


class InvalidOptimizer(Exception):
    """
    InvalidOptimizer is raised when trying to set an optimizer that does not exist.
    """


class InvalidPdfParameters(Exception):
    """
    InvalidPdfParameters is raised when trying to evaluate a pdf with invalid parameters.
    """


class InvalidPdfData(Exception):
    """
    InvalidPdfData is raised when trying to evaluate a pdf with invalid data.
    """


class FailedMinimization(Exception):
    """
    FailedMinimization is raised when a minimization did not succeed.
    """

    def __init__(self, result):
        self.result = result
        message = getattr(
            result, 'message', "Unknown failure. See fit result for more details."
        )
        super().__init__(message)
