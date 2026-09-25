"""Puts the mars_sim_driver ROS package dir on sys.path so sandbox scripts
can `from mars_sim_driver import world` without a ROS install."""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1] / "innate-os" / "ros2_ws" / "src" / "mars_bot" / "mars_sim_driver"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
