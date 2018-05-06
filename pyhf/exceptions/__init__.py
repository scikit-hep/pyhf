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
