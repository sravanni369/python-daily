"""Day 15 - the GIL, and why threads don't speed up CPU work in Python.

Source on my own shelf:
  "Dive into Deep Learning" (Zhang, Lipton, Li, Smola), PDF p. 270:
    "it is now subject to the dreaded global interpreter lock that makes everything
     wait for Python to complete."
  Same book, ch. 13 Computational Performance: 13.2 Asynchronous Computation (p. 552),
  13.3 Automatic Parallelism (p. 555).

The book names the GIL as a cost and moves on. This measures it.

Only one thread can execute Python bytecode at a time. So for CPU-bound work, a thread
pool buys you nothing and usually costs a little. Processes each get their own
interpreter and their own GIL, so they actually scale.

The inversion is the interesting part: for I/O-bound work the same thread pool helps a
lot, because a thread waiting on I/O releases the GIL and lets another run.

Standard library only.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List

WORKERS = 4
TASKS = 8


def cpu_work(n: int) -> int:
    """Burn CPU. Pure Python arithmetic, so it holds the GIL the whole time."""
    total = 0
    for i in range(3_000_000):
        total += i * i % 7
    return total


def io_work(n: int) -> int:
    """Simulate waiting on a file or a network call. sleep() releases the GIL."""
    time.sleep(0.25)
    return n


def run_serial(fn: Callable[[int], int], items: List[int]) -> float:
    start = time.perf_counter()
    for i in items:
        fn(i)
    return time.perf_counter() - start


def run_pool(pool_cls, fn: Callable[[int], int], items: List[int]) -> float:
    start = time.perf_counter()
    with pool_cls(max_workers=WORKERS) as pool:
        list(pool.map(fn, items))
    return time.perf_counter() - start


def report(label: str, serial: float, threads: float, procs: float) -> None:
    print(f"\n{label}")
    print(f"  serial     {serial:6.2f}s   baseline")
    print(f"  threads    {threads:6.2f}s   {serial / threads:4.2f}x")
    print(f"  processes  {procs:6.2f}s   {serial / procs:4.2f}x")


if __name__ == "__main__":
    # This guard is not optional. Without it, ProcessPoolExecutor on Windows re-imports
    # this module in every child process and spawns processes forever.
    print(f"CPU count: {os.cpu_count()}   workers: {WORKERS}   tasks: {TASKS}")
    items = list(range(TASKS))

    print("\nRunning CPU-bound work three ways...")
    c_serial = run_serial(cpu_work, items)
    c_threads = run_pool(ThreadPoolExecutor, cpu_work, items)
    c_procs = run_pool(ProcessPoolExecutor, cpu_work, items)
    report("CPU-BOUND: counting in a Python loop", c_serial, c_threads, c_procs)

    if c_serial / c_threads < 1.2:
        print("  -> threads bought nothing. The GIL let only one run Python at a time.")
    else:
        print(f"  -> threads gave {c_serial / c_threads:.2f}x here, which is more than the")
        print("     GIL should allow. Worth checking what this machine is doing.")

    print("\nRunning I/O-bound work three ways...")
    i_serial = run_serial(io_work, items)
    i_threads = run_pool(ThreadPoolExecutor, io_work, items)
    i_procs = run_pool(ProcessPoolExecutor, io_work, items)
    report("I/O-BOUND: sleeping, the way a file or network read waits", i_serial, i_threads, i_procs)
    print("  -> threads win now. A thread waiting on I/O releases the GIL.")

    print("\nThe rule")
    print("  CPU-bound  -> processes. Each gets its own interpreter and its own GIL.")
    print("  I/O-bound  -> threads. They are cheaper than processes and the waiting is free.")
    print("  Reaching for threads to make slow Python maths faster is the common mistake.")
