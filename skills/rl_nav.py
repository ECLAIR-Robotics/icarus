import math
import time
from pathlib import Path

from pydantic import BaseModel

from innate import Lidar, Mobility, Odometry, Skill, SkillOutput, SkillReturn, resource

from .rl_nav_utils import NavController, NavSpec, Policy, goal_in_base, load_policy, relative_to_odom

POLICY_DIR = Path(__file__).resolve().parent / "rl_nav_policy"

MAX_GOAL_DISTANCE = 5.0  # m; the policy trained on rooms up to 8 x 8 m
TIME_LIMIT_PER_METER = 10.0  # s
MIN_TIME_LIMIT = 20.0
# cmd_vel deadman: the base stops by itself if this loop dies.
CMD_DURATION = 0.3


class RlNavResult(BaseModel):
    reached: bool
    remaining_m: float
    safety_stops: int


class RlNavigate(Skill):
    """Drive to a point given relative to where the robot is now (forward_m ahead, left_m to the
    left, metres) with the learned lidar obstacle-avoidance policy. Goals up to 5 m away."""

    mobility: Mobility
    odom: Odometry
    lidar: Lidar

    @resource
    def _spec(self) -> NavSpec:
        return NavSpec.load(POLICY_DIR / "policy_io.json")

    @resource
    def _policy(self) -> Policy:
        return load_policy(POLICY_DIR)

    def guidelines(self) -> str:
        return (
            "Drives to a point relative to the robot's current pose (forward_m ahead, left_m to the "
            "left) while avoiding obstacles seen by the lidar, using an RL policy trained in simulation. "
            "It can't see obstacles below ~17 cm (the lidar height) or overhangs. With dry_run=True it "
            "only reports what it would do. Speed is capped at 0.4 m/s and 1 rad/s."
        )

    def execute(self, forward_m: float, left_m: float, dry_run: bool = False) -> SkillReturn:
        if math.hypot(forward_m, left_m) > MAX_GOAL_DISTANCE:
            self.fail(f"Goal is {math.hypot(forward_m, left_m):.1f} m away; the policy handles up to {MAX_GOAL_DISTANCE} m")
        spec = self._spec
        try:
            controller = NavController(spec, self._policy)
        except ValueError as e:
            self.fail(str(e))
        if self.wait_for(lambda: self.odom, timeout=3.0) is None or self.wait_for(lambda: self.lidar, timeout=3.0) is None:
            self.fail("No odometry or lidar")

        odom = self.odom
        goal = relative_to_odom((odom.x, odom.y, odom.theta), forward_m, left_m)
        deadline = time.monotonic() + max(MIN_TIME_LIMIT, TIME_LIMIT_PER_METER * math.hypot(forward_m, left_m))
        period = 1.0 / spec.control_hz
        controller.reset(time.monotonic())
        next_tick = time.monotonic()
        try:
            while True:
                odom, scan = self.odom, self.lidar
                pose = (odom.x, odom.y, odom.theta)
                goal_body = goal_in_base(goal, pose)
                remaining = math.hypot(*goal_body)
                result = RlNavResult(reached=remaining < spec.goal_radius, remaining_m=round(remaining, 2), safety_stops=controller.safety_stops)
                if result.reached:
                    self.mobility.stop()
                    return SkillOutput(f"Reached the goal ({remaining:.2f} m off)", result)
                if time.monotonic() > deadline:
                    self.mobility.stop()
                    return SkillOutput(f"Timed out {remaining:.2f} m from the goal", result)

                cmd = controller.step(goal_body, pose, odom.stamp, scan, time.monotonic())
                if dry_run:
                    nearest = cmd.bins.min() * spec.lidar_max_range
                    self.feedback(f"goal ({goal_body[0]:.2f}, {goal_body[1]:.2f}) m -> cmd {cmd.linear:.2f} m/s, {cmd.angular:.2f} rad/s; nearest {nearest:.2f} m")
                else:
                    self.mobility.send_cmd_vel(linear_x=cmd.linear, angular_z=cmd.angular, duration=CMD_DURATION)

                next_tick += period
                self.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.mobility.stop()
