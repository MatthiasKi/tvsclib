from enum import Enum

class CanonicalForm(Enum):
    """ Canonical (characterizing) Forms of a state space system. """
    INPUT = 1
    OUTPUT = 2
    BALANCED = 3