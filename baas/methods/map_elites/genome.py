"""MAP-Elites genome spec for discrete adversary action sequences.

Multi-adversary: genome shape is (n_blocks, n_adv), each column being an
independent action sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ActionSeqGenomeSpec:
    """Genome specification for a discrete adversary action sequence.

    Single adversary: genome shape (n_blocks,).
    Multi-adversary: genome shape (n_blocks, n_adv).
    """

    horizon_steps: int
    n_adversaries: int = 1
    n_actions: int = 5
    n_blocks: int = 20
    block_size: int = 3
    pad_action: int = 1    # IDLE

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")
        if self.n_actions <= 1:
            raise ValueError("n_actions must be >= 2")
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if not (0 <= self.pad_action < self.n_actions):
            raise ValueError("pad_action out of range")

    @property
    def genome_shape(self) -> tuple:
        """Shape of a single genome array."""
        return (self.n_blocks,) if self.n_adversaries == 1 else (self.n_blocks, self.n_adversaries)

    @property
    def solution_dim(self) -> int:
        """Flattened genome length used by pyribs emitters."""
        return int(np.prod(self.genome_shape))

    def decode(self, genome: np.ndarray) -> List[List[int]]:
        """Decode a genome array to per-adversary action sequences.

        Returns n_adversaries lists, each of length horizon_steps.
        Float genomes are rounded then clamped to [0, n_actions-1].
        """
        g = np.asarray(genome).reshape(self.genome_shape)
        g_int = np.clip(np.rint(g).astype(np.int32), 0, self.n_actions - 1)

        seqs: List[List[int]] = []
        for adv_idx in range(self.n_adversaries):
            col = g_int[:, adv_idx] if self.n_adversaries > 1 else g_int
            seq: List[int] = []
            for gene in col:
                seq.extend([int(gene)] * self.block_size)
            if len(seq) < self.horizon_steps:
                seq.extend([self.pad_action] * (self.horizon_steps - len(seq)))
            else:
                seq = seq[: self.horizon_steps]
            seqs.append(seq)
        return seqs

    def sample_random(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random integer genome."""
        return rng.integers(low=0, high=self.n_actions, size=self.genome_shape, dtype=np.int32)

    def to_continuous(self, genome: np.ndarray) -> np.ndarray:
        """Cast integer genome to float32 for pyribs emitters."""
        return np.asarray(genome, dtype=np.float32).reshape(-1)

    def from_continuous(self, x: np.ndarray) -> np.ndarray:
        """Round a continuous pyribs solution back to integer genome."""
        return np.clip(np.rint(x), 0, self.n_actions - 1).astype(np.int32).reshape(self.genome_shape)
