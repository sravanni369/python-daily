"""Independent check of day15. Re-times the same work and recomputes every speedup,
rather than trusting the numbers day15 printed about itself."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from day15_gil_threads_vs_processes import cpu_work, io_work

ITEMS = list(range(8))


def timed(fn, items, pool_cls=None):
    start = time.perf_counter()
    if pool_cls is None:
        for i in items:
            fn(i)
    else:
        with pool_cls(max_workers=4) as pool:
            list(pool.map(fn, items))
    return time.perf_counter() - start


if __name__ == "__main__":
    print(f"cores = {os.cpu_count()}, workers = 4, tasks = {len(ITEMS)}\n")
    verdicts = []

    for name, fn, claim, test in [
        ("CPU-bound", cpu_work, "threads <= 1.2x  (GIL blocks parallel bytecode)",
         lambda s: s <= 1.2),
        ("I/O-bound", io_work, "threads >  2.0x  (waiting thread releases the GIL)",
         lambda s: s > 2.0),
    ]:
        ser = timed(fn, ITEMS)
        thr = timed(fn, ITEMS, ThreadPoolExecutor)
        prc = timed(fn, ITEMS, ProcessPoolExecutor)
        s_thr, s_prc = ser / thr, ser / prc
        ok = test(s_thr)
        verdicts.append(ok)

        print(f"{name}")
        print(f"  serial     {ser:5.2f}s")
        print(f"  threads    {thr:5.2f}s   recomputed speedup {s_thr:.2f}x")
        print(f"  processes  {prc:5.2f}s   recomputed speedup {s_prc:.2f}x")
        print(f"  claim: {claim}")
        print(f"  -> {'HOLDS' if ok else 'FAILS'}\n")

    print("ALL CLAIMS HOLD" if all(verdicts) else "AT LEAST ONE CLAIM FAILED")
