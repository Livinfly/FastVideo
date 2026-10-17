"""Host-side process cleanup and coordinator admission; never used during timing."""

import json
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path


def stop_worker(child, timeout=30):
    """Stop the entire dedicated worker group, even if its leader already exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader can exit while a native scheduler/rank ignores SIGTERM.
        # Do not release the GPU locks with live descendants left behind.
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def completed_groups(suite):
    """Only the coordinator can accept a group after checking its exit status."""
    path = Path(suite) / "status.json"
    if not path.exists():
        return set()
    state = json.loads(path.read_text())
    return {group["directory"] for group in state.get("completed", [])
            if group.get("status") == "complete"}
