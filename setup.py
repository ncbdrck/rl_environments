from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages

# fetch values from package.xml
setup_args = generate_distutils_setup(
    name="rl_environments",
    packages=find_packages('src'),
    package_dir={'': 'src'},

    description="The rl_environments package",
    url="https://github.com/ncbdrck/rl_environments.git",
    keywords=['ROS', 'reinforcement learning', 'machine-learning', 'gym', 'gymnasium', 'robotics', 'openai', 'gazebo',
              'realros', 'multiros', 'reactorx200', 'viperx300s', 'VX300S', 'Trossen', 'UR5e', 'Niryo'],

    author='Jayasekara Kapukotuwa',
    author_email='j.kapukotuwa@research.ait.ie',
)

setup(**setup_args)
