from gym_thor.envs.thor_env import ThorEnv

import logging
logger = logging.getLogger(__name__)


class KitchenEnv(ThorEnv):
    def __init__(self):
        super(KitchenEnv, self).__init__()

    def init(self):
        self.scene_name = 'kitchen_02'
        self.terminal_list = [90, 136, 157, 207, 329]
