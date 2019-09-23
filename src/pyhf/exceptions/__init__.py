import sys


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
        super(InvalidSpecification, self).__init__(message)


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


class InvalidOptimizer(Exception):
    """
    InvalidOptimizer is raised when trying to set using an optimizer that does not exist.
    """


class InvalidPdfParameters(Exception):
    """
    InvalidPdfParameters is raised when trying to evaluate a pdf with invalid parameters.
    """


class InvalidPdfData(Exception):
    """
    InvalidPdfData is raised when trying to evaluate a pdf with invalid data.
    """
