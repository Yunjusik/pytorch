import numpy as np
import gym


class ActionSpace(gym.Space):
    """
        1) Block_Size(shard)   :   Discrete 4  - 2MB[0], 4MB[1], 6MB[2], 8MB[3],   - params: min: 2, max: 8  (megabytes)
        2) Time Interval       :   Discrete 4  - 2[0]  ,   4[1],   6[2],   8[3]    - params: min: 2, max: 8  (seconds)
        3) number of shard (K) :   Discrete 4  - 1[0],     2[1],   4[2],   8[3]    - params: min: 1, max: 8

    - Can be initialized as
        ([ 4, 4, 4 ]) Quaternary   -> 4x4x4 = 64 action space in discrete type action


    """

    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(ActionSpace, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, ActionSpace) and self.n == other.n
