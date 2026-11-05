"""OS parent-exit protection for molecule workers, without third-party packages."""

import ctypes
import os
import signal
import sys
from contextlib import contextmanager


def arm_parent_exit(parent_pid: int) -> None:
    """Make a Linux worker die when its spawning supervisor thread exits.

    Check the expected parent PID after arming SIGKILL to cover a parent that
    exited during setup. Windows protection is installed by the parent instead;
    other platforms fail before chemistry starts. Call before importing RDKit.
    """
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_PDEATHSIG refers to the spawning thread, which the supervisor
        # keeps alive until its child has been reaped. Check the setup race too.
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
        if os.getppid() != parent_pid:
            os._exit(1)
    elif os.name != "nt":
        raise RuntimeError("molecule process isolation requires Windows or Linux")


@contextmanager
def child_job(pid: int):
    """Guard a Windows worker and wait for job members when the context exits.

    Assign the child before its input handshake enables chemistry. Kill-on-close
    covers server crashes; normal cleanup retains process handles, terminates
    the job, waits for launcher descendants, and closes every retained handle.
    Job setup failures propagate to the supervisor, which reaps its Popen child.
    On Linux this context is a no-op; the worker arms its own parent-death signal.
    """
    if os.name != "nt":
        yield
        return

    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        """ctypes layout of JOBOBJECT_BASIC_LIMIT_INFORMATION."""

        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t),
            ("active_processes", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        """ctypes layout used to install the Windows kill-on-close flag."""

        _fields_ = [
            ("basic", BasicLimits),
            ("io_counters", ctypes.c_uint64 * 6),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    job = kernel.CreateJobObjectW(None, None)  # non-inheritable handle
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.AssignProcessToJobObject(job, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(process)
        try:
            yield
        finally:
            # A Windows venv launcher may spawn the actual interpreter. Waiting
            # only for Popen's PID does not wait for that descendant. Termination
            # is asynchronous; retain member handles BEFORE terminating the job
            # and wait for them to signal, even after the job reports zero active
            # processes. Native workers do not spawn further processes.
            handles = []
            try:
                count = 4
                while True:

                    class ProcessList(ctypes.Structure):
                        """Variable-length job membership buffer, grown on overflow."""

                        _fields_ = [
                            ("assigned", wintypes.DWORD),
                            ("listed", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * count),
                        ]

                    members = ProcessList()
                    if kernel.QueryInformationJobObject(
                        job, 3, ctypes.byref(members), ctypes.sizeof(members), None
                    ):
                        break
                    code = ctypes.get_last_error()
                    if code != 234:  # ERROR_MORE_DATA: grow rather than truncate
                        raise ctypes.WinError(code)
                    count = max(count * 2, members.assigned)
                for member_pid in members.pids[: members.listed]:
                    handle = kernel.OpenProcess(
                        0x100000, False, member_pid
                    )  # SYNCHRONIZE
                    if handle:
                        handles.append(handle)
                    elif ctypes.get_last_error() != 87:  # already exited
                        raise ctypes.WinError(ctypes.get_last_error())
                if not kernel.TerminateJobObject(job, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                for handle in handles:
                    if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
                        raise ctypes.WinError(ctypes.get_last_error())
            finally:
                for handle in handles:
                    kernel.CloseHandle(handle)
    finally:
        kernel.CloseHandle(job)
