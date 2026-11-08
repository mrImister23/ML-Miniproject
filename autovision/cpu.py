"""Match CPU thread counts to the CPU share the server really gives us.

Hosted containers (e.g. Streamlit Community Cloud) often see all the host's cores but may only
use ~2 of them. PyTorch and OpenCV then start one thread per host core, use up the share in a
burst and get paused by the OS, which shows up as low FPS with second-long freezes.
"""

import os
from pathlib import Path


def cpu_quota() -> float | None:
    """CPUs allowed by the container's cgroup limit (Linux), or None if there is no limit."""
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def usable_cpus() -> tuple[int, int, float | None]:
    """(cores the OS reports, cores we may use, cgroup quota)."""
    reported = os.cpu_count() or 1
    usable = reported
    if hasattr(os, "sched_getaffinity"):
        usable = len(os.sched_getaffinity(0))
    quota = cpu_quota()
    if quota is not None:
        usable = min(usable, max(1, int(quota)))
    return reported, usable, quota


def limit_cpu_threads() -> tuple[int, int, float | None]:
    """Cap PyTorch and OpenCV threads at the usable core count. Returns usable_cpus()."""
    import cv2
    import torch

    reported, usable, quota = usable_cpus()
    if not torch.cuda.is_available():
        # Shared cloud hosts report many cores but hand out ~2 in practice; one thread per
        # reported core only makes the threads fight. AUTOVISION_THREADS overrides.
        usable = int(os.environ.get("AUTOVISION_THREADS", min(usable, 2)))
    if usable < reported:
        torch.set_num_threads(usable)
        cv2.setNumThreads(usable)
    return reported, usable, quota
