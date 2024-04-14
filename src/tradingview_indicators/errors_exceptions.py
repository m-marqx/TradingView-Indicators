"""
Custom Exceptions Module

This module defines custom exceptions to handle specific error
conditions.

Classes
-------
InvalidArgumentError(Exception)
    Exception raised for invalid function arguments.

"""

class InvalidArgumentError(Exception):
    """
    Exception raised for invalid arguments.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """
