"""
Example module showing how to format the comments in your code to show up nicely in autodocs.

Overview
-----------

Classes:

* :class:`DocsExampleClass`

Functions:

* :func:`docs_example_func`

"""


class DocsExampleClass:
    """
    Example class showing you how to format the comments in your code to show up nicely in autodocs.

    Attributes
    --------------
    hello_flag : bool
        Set to True by initialisation

    Methods
    --------------
    __init__ :
        Initialises an instance of the class.

    """

    def __init__(self, **kwargs):
        """
        Initialise a new instance of the class.

        Parameters
        --------------
        **kwargs : dict
            Catch all rubbish.

        Returns
        --------------
        DocsExampleClass
            A new instance of the class.
        """
        self.hello_flag = True


def docs_example_func(val_in):
    """
    Adds one to the input value.

    Parameters
    --------------
    val_in : int
        A number that goes in.

    Returns
    --------------
    int :
        Input value incremented by one.
    """
    return val_in + 1
