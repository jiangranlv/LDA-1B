from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

class DistributedTaskBatchSampler(
    torch.utils.data.Sampler[List[Tuple[int, str]]]
):
    """
    Multi-task batch sampler with:
    - hard constraint: each batch contains at least one sample for each task
    - soft constraint: remaining slots are sampled according to task_weights

    The dataset is expected to support ``dataset[(index, task)]``.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        tasks: List[str],
        task_weights: Dict[str, float],
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert batch_size >= len(tasks), (
            "batch_size must be >= number of tasks to satisfy hard constraint"
        )

        if not dist.is_available() or not dist.is_initialized():
            num_replicas = 1
        else:
            num_replicas = dist.get_world_size()

        if not dist.is_available() or not dist.is_initialized():
            rank = 0
        else:
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.tasks = list(tasks)
        self.task_weights = np.array([task_weights[t] for t in self.tasks], dtype=np.float64)
        self.task_weights /= self.task_weights.sum()

        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.num_replicas = num_replicas

        self.batch_rng: np.random.Generator | None = None
        self.task_counters: dict[str, int] = {}
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.batch_rng = np.random.default_rng(self.seed + epoch * 1009 + self.rank * 13)
        # Rank-unique dummy index streams. Each task uses an independent counter.
        self.task_counters = {task: self.rank for task in self.tasks}

        # Keep dataset epoch synchronized when dataset supports it.
        if callable(getattr(self.dataset, "set_epoch", None)):
            self.dataset.set_epoch(epoch)

    def _sample_task_by_weight(self) -> str:
        assert self.batch_rng is not None
        return self.batch_rng.choice(self.tasks, p=self.task_weights)

    def _next_index(self, task: str) -> int:
        idx = self.task_counters[task]
        self.task_counters[task] += self.num_replicas
        return idx

    def __iter__(self) -> Iterator[List[Tuple[int, str]]]:
        for _ in range(len(self)):
            batch: List[Tuple[int, str]] = []

            for task in self.tasks:
                batch.append((self._next_index(task), task))

            while len(batch) < self.batch_size:
                task = self._sample_task_by_weight()
                batch.append((self._next_index(task), task))

            assert self.batch_rng is not None
            self.batch_rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        denom = self.batch_size * self.num_replicas
        if self.drop_last:
            return len(self.dataset) // denom
        return int(np.ceil(len(self.dataset) / denom))
