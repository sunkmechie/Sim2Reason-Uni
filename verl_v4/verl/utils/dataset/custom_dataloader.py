import torch.multiprocessing as mp
from queue import Empty
from typing import Generator
import sys
import time

import multiprocessing as mp
import time
import sys
from queue import Empty
from typing import List, Generator

class ParallelDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        collate_fn=None,
        max_batches_prefetch=10,
        max_index_queue_size=20,
        infinite=False,
        max_wait_time=60,
        max_restarts_per_worker=3,
        max_steps=None,
    ):
        '''
        Note: increase number of open files limit if you encounter too many open files error with many workers:
            ulimit -n 4096

        max_batches_prefetch: the maximum number of batches to prefetch
        max_index_queue_size: the maximum number of indices (batch_size * max_index_queue_size) to keep in the index queue
        infinite: whether to run in infinite mode
        max_wait_time: the maximum time to wait for a batch to be filled
        max_restarts_per_worker: the maximum number of times to restart a dead worker
        max_steps: the maximum number of iterations to run. The max steps has priority over infinite mode
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.max_batches_prefetch = max_batches_prefetch
        self.max_index_queue_size = max_index_queue_size
        self.infinite = infinite
        self.max_wait_time = max_wait_time
        self.max_restarts_per_worker = max_restarts_per_worker
        self.dataset_size = len(dataset)
        self.index_queue_size = max_index_queue_size * batch_size
        self.chunk_size = self.index_queue_size // 2

        print(f"Index queue size: {self.index_queue_size}")
        print(f"Max batches prefetch: {self.max_batches_prefetch * self.batch_size}")

        self.index_queue = None # reset the queues in self._start_workers()
        self.output_queue = None    # reset the queues in self._start_workers()
        self.workers = []
        self._next_index = 0
        self.worker_restart_count = [0] * self.num_workers
        self.max_steps = max_steps  # max steps has priority over infinite mode
    
    def __len__(self):
        if self.max_steps is not None:
            return self.max_steps
        else:
            return self.dataset_size // self.batch_size

    def _worker_loop(self, dataset, index_queue, output_queue):
        while True:
            try:
                index = index_queue.get(timeout=1)
            except Empty:
                continue
            if index is None:
                break
            # try:
            sample = dataset[index]
            if sample is not None:
                output_queue.put(sample)
            # except Exception as e:
            #     print(f"[Worker] Error at index {index}: {e}", file=sys.stderr)
            #     continue

    def _fill_index_queue(self, start_idx):
        end_idx = start_idx + self.chunk_size
        for idx in range(start_idx, end_idx):
            self.index_queue.put(idx % self.dataset_size)
        return end_idx

    def _start_workers(self):
        # reset the queues and workers
        self.index_queue = mp.Queue(maxsize=self.index_queue_size)
        self.output_queue = mp.Queue(maxsize=self.max_batches_prefetch * self.batch_size)
        self.workers = []
        self._next_index = 0
        self.worker_restart_count = [0] * self.num_workers

        for i in range(self.num_workers):
            p = mp.Process(target=self._worker_loop, args=(self.dataset, self.index_queue, self.output_queue))
            p.daemon = True # set daemon to True so the child process will be killed when the main process is killed
            p.name = f"Worker-{i}"
            p.start()
            self.workers.append(p)
        # Cancel join to prevent slow shutdown
        self.index_queue.cancel_join_thread()
        self.output_queue.cancel_join_thread()

    def __iter__(self) -> Generator[List[dict], None, None]:
        self._start_workers()
        self._next_index = self._fill_index_queue(0)
        

        batch = []
        step = 0
        batch_start_time = time.time()

        while True:
            try:
                sample = self.output_queue.get(timeout=5)
                if sample is not None:
                    batch.append(sample)
            
                if self.index_queue.qsize() < self.chunk_size:
                    # fill the index queue all the time
                    self._next_index = self._fill_index_queue(self._next_index)

                if len(batch) == self.batch_size:
                    print(f"[Step {step}] Batch ready | Output queue: {self.output_queue.qsize()/self.batch_size} | Index queue: {self.index_queue.qsize()/self.batch_size}")
                    if self.collate_fn is not None:
                        batch = self.collate_fn(batch)
                    yield batch
                    batch = []
                    step += 1
                    batch_start_time = time.time()
                    
                    # max steps has priority over infinite mode
                    if self.max_steps is not None and step >= self.max_steps:
                        print(f"[Main] ✅ Dataloader reached max steps ({self.max_steps}). Stopping.")
                        break
                    
                    if not self.infinite:
                        if (self.max_steps is None and self.dataset_size // self.batch_size <= step):
                            print(f"[Main] ✅ Dataloader reached max steps ({self.max_steps}). Stopping.")
                            break

            except Empty:
                if time.time() - batch_start_time > self.max_wait_time:
                    print(f"[Main] ⏱️ Timeout: waited >{self.max_wait_time}s for batch {step}. Stopping.")
                    break
                else:
                    print(f"[Main] ⚠️ Waiting for batch {step}... Collected data points: {self.output_queue.qsize()}")
                    # check and restart dead workers
                    for i, p in enumerate(self.workers):
                        if not p.is_alive():
                            if self.worker_restart_count[i] < self.max_restarts_per_worker:
                                print(f"[Main] 🔁 Worker {p.name} died. Restarting...")
                                new_p = mp.Process(target=self._worker_loop, args=(self.dataset, self.index_queue, self.output_queue))
                                new_p.daemon = True
                                new_p.name = f"Worker-{i}"
                                new_p.start()
                                self.workers[i] = new_p
                                self.worker_restart_count[i] += 1
                            else:
                                print(f"[Main] ❌ Worker {p.name} exceeded max restarts. Not restarting.")
                    continue

        self._cleanup()

    def _cleanup(self):
        # clear the index queue to avoid old tasks blocking workers from getting None
        self._drain_queue(self.index_queue)
        self._drain_queue(self.output_queue)
        print(f"[Main] ⚠️ Cleaning up... Index queue: {self.index_queue.qsize()}")
        print(f"[Main] ⚠️ Cleaning up... Output queue: {self.output_queue.qsize()}")
        for _ in range(self.num_workers):
            self.index_queue.put(None)

        for p in self.workers:
            p.join(timeout=1)

        self.workers.clear()
    
    def _drain_queue(self, q):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs

    from sim.qa_gen_rule import data_gen
    import datetime
    import random
    import sys
    import psutil
    import torch.multiprocessing as mp
    from queue import Empty
    from dataclasses import dataclass
    from typing import Any, Generator, Optional
    from verl.utils.dataset.online_rl_dataset import OnlineRLHFDataset
    from verl.utils.dataset.online_rl_dataset import collate_fn

    cfg = OmegaConf.load('config/config.yaml')
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs('Qwen/Qwen2.5-3B')

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    dataset = OnlineRLHFDataset(tokenizer, main_cfg=cfg)
    

    loader = ParallelDataLoader(dataset, batch_size=32, num_workers=16, 
                           max_index_queue_size=3, max_batches_prefetch=5, max_wait_time=120, infinite=True,
                           collate_fn=collate_fn, max_steps=2)

    start_time = time.time()
    for i, batch in enumerate(loader):
        print(f"[Step {i}] Training")
        # print(batch)
        # simulate computing source heavy task
        def compute_heavy_task():
            for _ in range(10000):
                a = 1*100*200 /(100^2)
        compute_heavy_task()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
    
    # for i, batch in enumerate(loader):
    #     print(f"[Step {i}] Training")
    #     # print(batch)
    #     # simulate computing source heavy task
    #     def compute_heavy_task():
    #         for _ in range(10000):
    #             a = 1*100*200 /(100^2)
    #     compute_heavy_task()
    #     end_time = time.time()
    #     print(f"Time taken: {end_time - start_time} seconds")