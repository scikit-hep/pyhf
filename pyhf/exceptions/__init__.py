import sys

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
        message = '{0}.\n\tPath: {1}\n\tInstance: {2}'.format(ValidationError.message, self.path, self.instance)
        # Call the base class constructor with the parameters it needs
        super(InvalidSpecification, self).__init__(message)


class InvalidModifier(Exception):
    """
    InvalidModifier is raised when an invalid modifier is requested. This includes:

        - creating a custom modifier with the wrong structure
        - initializing a modifier that does not exist, or has not been loaded

    """
    pass

class InvalidInterpCode(Exception):
    """
    InvalidInterpCode is raised when an invalid/unimplemented interpolation code is requested.
    """
    pass
