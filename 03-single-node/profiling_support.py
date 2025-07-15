# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified from torchtitan/profiling.py by Alex Rybchuk

import contextlib
import os
import pickle
import time
from pathlib import Path

import torch
# the number of warmup steps before the active step in each profiling cycle
WAIT = 1
WARMUP = 3
ACTIVE = 5

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(enable_profiling: bool = True,
                           with_memory: bool = True,
                           log_dir = Path):
    # get user defined profiler settings

    if enable_profiling:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if not log_dir.exists(): log_dir.mkdir()

        if with_memory:
            def trace_handler(prof):
                # Performance profiling
                prof.export_chrome_trace(str(Path(log_dir,f"rank{rank}_trace.json")))

                # Memory timeline by different categories
                if rank == 0:
                    prof.export_memory_timeline(str(Path(log_dir,f"rank{rank}_memory.html")), device="cuda:0")
        else:
            def trace_handler(prof):
                # Performance profiling
                prof.export_chrome_trace(str(Path(log_dir,f"rank{rank}_trace.json")))

        wait, warmup, active = WAIT, WARMUP, ACTIVE
        if (with_memory) and (rank==0):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=with_memory,
                with_stack=with_memory
            ) as torch_profiler:
                yield torch_profiler
        else:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
                on_trace_ready=trace_handler,
                record_shapes=True
            ) as torch_profiler:
                yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(*, 
                                 enable_snapshot: bool = True,
                                 log_dir_parent = Path):
    if enable_snapshot:
        if not log_dir_parent.exists(): log_dir_parent.mkdir()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = Path(log_dir_parent, dir_name)
                if not curr_snapshot_dir.exists(): curr_snapshot_dir.mkdir()
                with open(
                    Path(curr_snapshot_dir,f"rank{rank}_memory_snapshot.pickle"), "wb"
                ) as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)


        profiler = MemoryProfiler(0, 1)
        try:
            yield profiler
        except torch.OutOfMemoryError as e:
            print("OutOfMemory error!")
            profiler.step(exit_ctx=True)
    else:
        yield None
