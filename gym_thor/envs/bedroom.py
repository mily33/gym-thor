from gym_thor.envs.thor_env import ThorEnv

import logging
logger = logging.getLogger(__name__)


class BedroomEnv(ThorEnv):
    def __init__(self):
        super(BedroomEnv, self).__init__()

    def init(self):
        self.scene_name = 'bedroom_04'
        self.terminal_list = [134, 264, 320, 384, 387]
