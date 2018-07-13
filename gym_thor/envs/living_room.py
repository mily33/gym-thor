from gym_thor.envs.thor_env import ThorEnv

import logging
logger = logging.getLogger(__name__)


class LivingroomEnv(ThorEnv):
    def __init__(self):
        super(LivingroomEnv, self).__init__()

    def init(self):
        self.scene_name = 'livinig_room_08'
        self.terminal_list = [92, 135, 193, 228, 254]
