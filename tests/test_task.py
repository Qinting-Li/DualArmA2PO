from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.task import InsertionMetrics, InsertionPhase, InsertionStateMachine


def metrics(**overrides) -> InsertionMetrics:
    values = {
        "approach_distance_m": 0.1,
        "lateral_error_m": 0.01,
        "angle_error_deg": 5.0,
        "insertion_depth_m": 0.0,
        "contact_force_N": 0.0,
        "sync_error_m": 0.0,
    }
    values.update(overrides)
    return InsertionMetrics(**values)


class TaskTests(unittest.TestCase):
    def test_successful_phase_sequence(self):
        task = InsertionStateMachine()
        self.assertEqual(task.update(metrics(approach_distance_m=0.01)), InsertionPhase.ALIGN)
        self.assertEqual(
            task.update(metrics(lateral_error_m=0.0005, angle_error_deg=1.0)),
            InsertionPhase.INSERT,
        )
        self.assertEqual(
            task.update(metrics(insertion_depth_m=0.04)), InsertionPhase.HOLD
        )
        for _ in range(25):
            phase = task.update(metrics(insertion_depth_m=0.04))
        self.assertEqual(phase, InsertionPhase.DONE)

    def test_force_limit_aborts(self):
        task = InsertionStateMachine()
        self.assertEqual(
            task.update(metrics(contact_force_N=45.1)), InsertionPhase.ABORT
        )


if __name__ == "__main__":
    unittest.main()

