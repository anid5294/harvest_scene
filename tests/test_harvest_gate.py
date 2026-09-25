import unittest

import numpy as np

from contract import ContractError, FPS, validate_state_action_trace
from g1_harvest import (
    HarvestPolicy,
    apple_in_tray,
    choose_apple,
    default_home,
    local_to_world,
    robot_placement,
    world_to_local,
)


class HarvestGateTests(unittest.TestCase):
    def test_selected_apple_is_in_reach_band(self):
        positions = np.array([[0.4, 0.1, 1.75], [0.8, -0.2, 1.10], [1.2, 0.0, 0.55]])
        self.assertEqual(choose_apple(positions), 1)

    def test_robot_placement_preserves_target_in_local_frame(self):
        apple = np.array([1.25, -0.70, 1.12])
        base, yaw = robot_placement(apple)
        local = world_to_local(apple, base, yaw)
        np.testing.assert_allclose(local, [0.50, -0.20, 1.12], atol=1e-8)
        np.testing.assert_allclose(local_to_world(local, base, yaw), apple, atol=1e-8)

    def test_tray_requires_position_and_settling(self):
        tray = np.array([0.3, -0.4, 0.88])
        self.assertTrue(apple_in_tray(tray + [0.0, 0.0, 0.06], tray, 0.03))
        self.assertFalse(apple_in_tray(tray + [0.0, 0.0, 0.06], tray, 0.2))

    def test_contract_accepts_contiguous_control_trace(self):
        states = np.zeros((4, 43), dtype=np.float32)
        actions = np.ones((4, 43), dtype=np.float32)
        timestamps = np.arange(4, dtype=np.float32) / np.float32(FPS)
        validate_state_action_trace(states, actions, timestamps)
        with self.assertRaises(ContractError):
            validate_state_action_trace(states, actions, timestamps + np.float32(0.01))

    def test_seed42_nominal_waypoints_have_ik_solutions(self):
        policy = HarvestPolicy(default_home(), np.array([0.50, -0.20, 1.12]))
        self.assertEqual(policy.phases[-1].name, "SETTLE")
        self.assertTrue(all(np.isfinite(phase.target).all() for phase in policy.phases))


if __name__ == "__main__":
    unittest.main()
