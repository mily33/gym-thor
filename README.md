# gym-thor

Before install gym-thor, please run 'download_scene_dumps.sh' to download scene data.
```
cd gym-thor
./download_scene_dumps
```
After finishing downloading, run
```
pip install -e .
```
## test 
```python
import gym
import gym_thor
env = gym.make('bedroom-v0')
```
