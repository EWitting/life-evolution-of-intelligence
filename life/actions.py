"""Fixed action enum (ADR-005). Append only; never reorder or renumber."""

NOOP = 0
FORWARD = 1        # move one cell in the facing direction, if not blocked
TURN_LEFT = 2
TURN_RIGHT = 3
USE = 4            # held object on the faced cell: transition, else pick up, else drop
EAT = 5            # consume the held object if it is edible
VOCALIZE = 6       # emit a sound heard by agents within hear_radius next tick

NUM_ACTIONS = 7
NAMES = ["NOOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT", "USE", "EAT", "VOCALIZE"]
