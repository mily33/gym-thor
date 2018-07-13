from setuptools import setup


setup(name='gym-thor',
      version='0.0.1',
      install_requires=['gym'],
      package_data={"gym-thor": ["gym_thor/envs/data/*.h5"]})
