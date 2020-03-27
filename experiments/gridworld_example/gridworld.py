import numpy as np

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step
from tf_agents.environments import utils
from tf_agents.specs import array_spec


class GridWorldEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,),
            dtype=np.int32,
            minimum=[0, 0, 0, 0],
            maximum=[5, 5, 5, 5],
            name="observation",
        )
        self._state = [0, 0, 5, 5]
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, 5, 5]  # [x, y, win_x, win_y]
        self.episode_ended = False
        return time_step.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.move(action)

        if self.game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.game_over():
                reward = 100
            else:
                reward = 0
            return time_step.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return time_step.transition(
                np.array(self._state, dtype=np.int32), reward=0, discount=0.9
            )

    def move(self, action):
        row, col, _, _ = self._state
        if action == 0:
            if row - 1 >= 0:
                self._state[0] -= 1
        if action == 1:
            if row + 1 < 6:
                self._state[0] += 1
        if action == 2:
            if col - 1 >= 0:
                self._state[1] -= 1
        if action == 3:
            if col + 1 < 6:
                self._state[1] += 1

    def game_over(self):
        row, col, frow, fcol = self._state
        return row == frow and col == fcol


if __name__ == "__main__":
    env = GridWorldEnv()
    utils.validate_py_environment(env, episodes=5)
