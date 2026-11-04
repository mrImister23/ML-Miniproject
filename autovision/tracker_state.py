"""Per-track memory: smooths classifier output over time so labels don't flicker."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackMemory:
    probs: deque = field(default_factory=lambda: deque(maxlen=10))
    last_classified: int = -10**9
    last_seen: int = 0

    def mean_proba(self) -> np.ndarray | None:
        if not self.probs:
            return None
        return np.mean(self.probs, axis=0)


class TrackState:
    def __init__(self, history: int = 10, stale_after: int = 60):
        self.history = history
        self.stale_after = stale_after
        self.tracks: dict[int, TrackMemory] = {}

    def get(self, track_id: int, frame_idx: int) -> TrackMemory:
        mem = self.tracks.get(track_id)
        if mem is None:
            mem = TrackMemory(probs=deque(maxlen=self.history))
            self.tracks[track_id] = mem
        mem.last_seen = frame_idx
        return mem

    def needs_classification(self, track_id: int, frame_idx: int, every_n: int) -> bool:
        mem = self.tracks.get(track_id)
        return mem is None or frame_idx - mem.last_classified >= every_n

    def add(self, track_id: int, frame_idx: int, proba: np.ndarray):
        mem = self.get(track_id, frame_idx)
        mem.probs.append(proba)
        mem.last_classified = frame_idx

    def cleanup(self, frame_idx: int):
        stale = [t for t, m in self.tracks.items() if frame_idx - m.last_seen > self.stale_after]
        for t in stale:
            del self.tracks[t]

    def reset(self):
        self.tracks.clear()
