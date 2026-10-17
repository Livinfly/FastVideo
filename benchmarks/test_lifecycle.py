"""Regression checks for process trees and failed groups with complete metrics."""

import os
import signal
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common import HERE, schedule, write_json
from lifecycle import stop_worker
from report import collect


class LifecycleTests(unittest.TestCase):
    def test_cleanup_stops_descendants_after_client_exit(self):
        # A native worker can survive its API client and ignore a graceful stop.
        # Use real CPU processes, so a mock cannot hide an orphaned process group.
        script = """
import subprocess, sys
code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(300)'
subprocess.Popen([sys.executable, '-u', '-c', code])
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script], start_new_session=True,
            stdout=subprocess.PIPE, text=True,
        )
        try:
            self.assertTrue(select.select([child.stdout], [], [], 10)[0])
            self.assertEqual(child.stdout.readline(), "ready\n")
            child.wait(timeout=10)
            self.assertEqual(child.returncode, 0)
            # The descendant still holds the pipe open after the API client exits.
            # Pipe EOF works even when /proc uses a different PID namespace.
            self.assertFalse(select.select([child.stdout], [], [], 0)[0])
            stop_worker(child, timeout=0.1)
            self.assertTrue(select.select([child.stdout], [], [], 5)[0])
            self.assertEqual(child.stdout.readline(), "")
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            child.stdout.close()

    def test_failed_or_unreaped_group_cannot_supply_speedup(self):
        named = (HERE / "configs/fasth3_v2.json").exists()
        groups = [
            dict(model="test", gpus=n, directory=f"test/g{n}" if named else f"offload_off/g{n}")
            for n in [1, 2]
        ]
        plan = dict(groups=groups, gpu_counts=[1, 2], offload=["off"], modes=["dit"], warmup=1, repeats=1)
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder)
            for group in groups:
                directory = suite / group["directory"]
                directory.mkdir(parents=True)
                seconds = 10 / group["gpus"]
                records = [
                    dict(mode=mode, warmup=warmup, repeat=repeat, output_shape=[1, 3, 1, 8, 8],
                         response_e2e_s=seconds, distributed_e2e_s=seconds,
                         dit_forward_sum_max_rank_s=seconds, dit_forward_step_max_sum_s=seconds)
                    for mode, warmup, repeat in schedule(["dit"], 1, 1)
                ]
                write_json(directory / "metrics.json", dict(status="complete", records=records))
            # Both metrics files are complete, but G1 failed during teardown.
            for first_status in ["failed", "running"]:
                completed = [dict(groups[1], status="complete")]
                if first_status == "failed":
                    completed.append(dict(groups[0], status="failed"))
                write_json(suite / "status.json", dict(status="running", completed=completed))
                self.assertTrue(all("e2e_speedup" not in row for row in collect(suite, plan)))
            write_json(suite / "status.json", dict(
                status="complete", completed=[dict(group, status="complete") for group in groups]
            ))
            self.assertEqual(collect(suite, plan)[1]["e2e_speedup"], 2)


if __name__ == "__main__":
    unittest.main()
