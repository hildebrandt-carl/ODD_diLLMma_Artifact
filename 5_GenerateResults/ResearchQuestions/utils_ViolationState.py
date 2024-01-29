from enum import Enum

# Define the violation state
class ViolationState(Enum):
    NO_VIOLATION    = (1, 'C2')
    UNDETERMINED    = (2, 'C0')
    MIXED_RESPONSE  = (3, 'C1')
    YES_VIOLATION   = (4, 'C3')

    def __init__(self, num, color):
        self.num = num
        self.color = color
