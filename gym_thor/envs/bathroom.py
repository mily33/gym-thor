from gym_thor.envs.thor_env import ThorEnv

import logging
logger = logging.getLogger(__name__)


class BathroomEnv(ThorEnv):
    def __init__(self):
        super(BathroomEnv, self).__init__()

    def init(self):
        self.scene_name = 'bathroom_02'
        self.terminal_list = [26, 37, 43, 53, 69]
