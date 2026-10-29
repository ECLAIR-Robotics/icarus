import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'icarus'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # rl_nav_node runs the skill pack's policy runner and policy files, and serves web/rl_nav.
        (os.path.join('share', package_name, 'rl_nav'), ['skills/rl_nav_utils.py']),
        (os.path.join('share', package_name, 'rl_nav', 'rl_nav_policy'), glob('skills/rl_nav_policy/*')),
        (os.path.join('share', package_name, 'rl_nav', 'web'), glob('web/rl_nav/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manas',
    maintainer_email='manas.agrawal2@gmail.com',
    description='Icarus ROS 2 package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'icarus_node = icarus.icarus_node:main',
            'rl_nav_node = icarus.rl_nav_node:main',
        ],
    },
)
