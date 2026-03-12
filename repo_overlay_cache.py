"""Reusable overlayfs and tmpfs repo cache helpers.

The primary use case is to pull a MooseFS-backed repo into node-local tmpfs once
per worker process, then run git-heavy operations against the local copy. When a
mutable workspace is required (for example to fetch missing SHAs), mount an
overlayfs layer over the tmpfs lowerdir so writes remain copy-on-write and can be
discarded cheaply.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

DEFAULT_REPO_CACHE_BASE = os.environ.get("REPO_RAMDISK_CACHE_BASE", "/dev/shm/repo_cache")
DEFAULT_OVERLAY_MERGED_BASE = os.environ.get("REPO_OVERLAY_MERGED_BASE", "/dev/shm/repo_overlay_merged")
DEFAULT_OVERLAY_TMP_BASE = os.environ.get("REPO_OVERLAY_TMP_BASE", "/dev/shm/repo_overlay_tmp")
DEFAULT_COPY_TIMEOUT_SEC = int(os.environ.get("REPO_CACHE_COPY_TIMEOUT_SEC", "3600"))
DEFAULT_MOUNT_TIMEOUT_SEC = int(os.environ.get("REPO_OVERLAY_MOUNT_TIMEOUT_SEC", "120"))


@dataclass
class OverlayMount:
    merged: str
    upper: str
    work: str
    lower: str
    tag: str


def _safe_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(tag))[:180]


def _remaining_timeout(deadline_monotonic: float | None, default_timeout_sec: int) -> int:
    if deadline_monotonic is None:
        return default_timeout_sec
    remaining = max(0.0, deadline_monotonic - time.monotonic())
    if remaining <= 0.0:
        raise TimeoutError("deadline exceeded")
    return max(1, min(default_timeout_sec, int(remaining)))


def mount_overlay(
    repo_path: str,
    tag: str,
    merged_base: str = DEFAULT_OVERLAY_MERGED_BASE,
    scratch_base: str = DEFAULT_OVERLAY_TMP_BASE,
    mount_timeout_sec: int = DEFAULT_MOUNT_TIMEOUT_SEC,
    deadline_monotonic: float | None = None,
) -> OverlayMount:
    safe_tag = _safe_tag(tag)
    merged = os.path.join(merged_base, safe_tag)
    upper = os.path.join(scratch_base, f"ovl-upper-{safe_tag}")
    work = os.path.join(scratch_base, f"ovl-work-{safe_tag}")
    for d in (merged_base, scratch_base, merged, upper, work):
        os.makedirs(d, exist_ok=True)
    subprocess.run(
        [
            "fuse-overlayfs",
            "-o",
            f"lowerdir={repo_path},upperdir={upper},workdir={work}",
            merged,
        ],
        check=True,
        timeout=_remaining_timeout(deadline_monotonic, mount_timeout_sec),
    )
    return OverlayMount(merged=merged, upper=upper, work=work, lower=repo_path, tag=safe_tag)


def unmount_overlay(mount: OverlayMount | tuple[str, str, str] | None) -> None:
    if mount is None:
        return
    if isinstance(mount, tuple):
        merged, upper, work = mount
    else:
        merged, upper, work = mount.merged, mount.upper, mount.work
    for cmd in (
        ["fusermount3", "-u", merged],
        ["fusermount3", "-u", "-z", merged],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception:
            continue
        if r.returncode == 0:
            break
    for d in (upper, work, merged):
        shutil.rmtree(d, ignore_errors=True)


class RepoRamdiskCache:
    """Process-scoped tmpfs repo cache.

    Each worker process copies a source repo into tmpfs lazily on first use.
    Copy-on-write overlays can then be mounted over the cached repo when a
    mutable workspace is required.
    """

    def __init__(
        self,
        cache_root: str | None = None,
        overlay_merged_base: str | None = None,
        overlay_tmp_base: str | None = None,
        copy_timeout_sec: int = DEFAULT_COPY_TIMEOUT_SEC,
        cleanup_on_exit: bool = True,
    ) -> None:
        pid_tag = f"pid-{os.getpid()}"
        self.cache_root = cache_root or os.path.join(DEFAULT_REPO_CACHE_BASE, pid_tag)
        self.overlay_merged_base = overlay_merged_base or os.path.join(DEFAULT_OVERLAY_MERGED_BASE, pid_tag)
        self.overlay_tmp_base = overlay_tmp_base or os.path.join(DEFAULT_OVERLAY_TMP_BASE, pid_tag)
        self.copy_timeout_sec = copy_timeout_sec
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        os.makedirs(self.cache_root, exist_ok=True)
        os.makedirs(self.overlay_merged_base, exist_ok=True)
        os.makedirs(self.overlay_tmp_base, exist_ok=True)
        if cleanup_on_exit:
            atexit.register(self.cleanup)

    def _repo_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _ready_marker(self, path: str) -> str:
        return os.path.join(path, ".repo_cache_ready")

    def ensure_local_repo(self, source_repo_dir: str, deadline_monotonic: float | None = None) -> str:
        if not source_repo_dir or not os.path.isdir(source_repo_dir):
            raise FileNotFoundError(source_repo_dir)
        repo_name = os.path.basename(os.path.normpath(source_repo_dir))
        dest = os.path.join(self.cache_root, repo_name)
        marker = self._ready_marker(dest)
        if os.path.exists(marker):
            return dest
        lock = self._repo_lock(repo_name)
        with lock:
            if os.path.exists(marker):
                return dest
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)
            staging = os.path.join(
                self.cache_root,
                f".staging-{repo_name}-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}",
            )
            os.makedirs(staging, exist_ok=False)
            try:
                subprocess.run(
                    ["cp", "-a", source_repo_dir + "/.", staging + "/"],
                    check=True,
                    timeout=_remaining_timeout(deadline_monotonic, self.copy_timeout_sec),
                )
                with open(self._ready_marker(staging), "w") as f:
                    f.write(f"source={source_repo_dir}\n")
                    f.write(f"copied_at_unix={int(time.time())}\n")
                os.rename(staging, dest)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return dest

    def mount_overlay(
        self,
        source_repo_dir: str,
        tag: str,
        deadline_monotonic: float | None = None,
    ) -> OverlayMount:
        lower = self.ensure_local_repo(source_repo_dir, deadline_monotonic=deadline_monotonic)
        return mount_overlay(
            lower,
            tag,
            merged_base=self.overlay_merged_base,
            scratch_base=self.overlay_tmp_base,
            deadline_monotonic=deadline_monotonic,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.overlay_merged_base, ignore_errors=True)
        shutil.rmtree(self.overlay_tmp_base, ignore_errors=True)
        shutil.rmtree(self.cache_root, ignore_errors=True)
