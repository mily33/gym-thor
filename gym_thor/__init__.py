from gym.envs.registration import register


register(id='thor-v0', entry_point='gym_thor.envs:ThorEnv')
register(id='bedroom-v0', entry_point='gym_thor.envs:BedroomEnv')
register(id='livingroom-v0', entry_point='gym_thor.envs:LivingroomEnv')
register(id='bathroom-v0', entry_point='gym_thor.envs:BathroomEnv')
register(id='kitchen-v0', entry_point='gym_thor.envs:KitchenEnv')
