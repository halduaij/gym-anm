"""
HDF5 + RLDS writer for scalable offline-RL datasets.

Layout
------
root/
    transitions/          # 1 dataset per field
        obs:        float32 [N, obs_dim]
        action:     float32 [N, act_dim]
        reward:     float32 [N]
        next_obs:   float32 [N, obs_dim]
        done:       uint8   [N]
        beh_id:     int32   [N]     (index into behaviour table)
    behaviour/
        id:         int32   [M]
        name:       string  [M]
        params:     string  [M]  (JSON)
    metadata/attrs        dataset-level attrs
"""
from __future__ import annotations
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class RLDatasetWriter:
    def __init__(self, path: str | Path, obs_dim: int, act_dim: int,
                 est_n_transitions: int, compress: bool = True):
        self.file = h5py.File(path, "w")
        c = "gzip" if compress else None
        self.dset = {
            "obs": self.file.create_dataset(
                "transitions/obs", (0, obs_dim), maxshape=(None, obs_dim),
                dtype="f4", compression=c
            ),
            "action": self.file.create_dataset(
                "transitions/action", (0, act_dim), maxshape=(None, act_dim),
                dtype="f4", compression=c
            ),
            "reward": self.file.create_dataset(
                "transitions/reward", (0,), maxshape=(None,), dtype="f4", compression=c
            ),
            "next_obs": self.file.create_dataset(
                "transitions/next_obs", (0, obs_dim), maxshape=(None, obs_dim),
                dtype="f4", compression=c
            ),
            "done": self.file.create_dataset(
                "transitions/done", (0,), maxshape=(None,), dtype="u1", compression=c
            ),
            "beh_id": self.file.create_dataset(
                "transitions/beh_id", (0,), maxshape=(None,), dtype="i4", compression=c
            ),
        }
        self._size = 0
        self.beh_table_grp = self.file.create_group("behaviour")
        self.beh_idx = {}

    def _grow(self, n: int):
        for ds in self.dset.values():
            ds.resize(self._size + n, axis=0)

    def add_behaviour(self, name: str, params: Dict[str, Any]) -> int:
        if name in self.beh_idx:
            return self.beh_idx[name]
        idx = len(self.beh_idx)
        self.beh_idx[name] = idx
        id_ds = self.beh_table_grp.require_dataset("id", (idx + 1,), dtype="i4", maxshape=(None,), exact=False)
        if id_ds.shape[0] <= idx:
            id_ds.resize((idx + 1,))
        id_ds[idx] = idx
        
        name_ds = self.beh_table_grp.require_dataset("name", (idx + 1,), dtype=h5py.string_dtype(), maxshape=(None,), exact=False)
        if name_ds.shape[0] <= idx:
            name_ds.resize((idx + 1,))
        name_ds[idx] = name
        
        params_ds = self.beh_table_grp.require_dataset("params", (idx + 1,), dtype=h5py.string_dtype(), maxshape=(None,), exact=False)
        if params_ds.shape[0] <= idx:
            params_ds.resize((idx + 1,))
        params_ds[idx] = json.dumps(params)
        return idx

    def append_transition(self, obs, action, reward, next_obs, done, beh_id):
        self._grow(1)
        i = self._size
        self.dset["obs"][i] = obs
        self.dset["action"][i] = action
        self.dset["reward"][i] = reward
        self.dset["next_obs"][i] = next_obs
        self.dset["done"][i] = done
        self.dset["beh_id"][i] = beh_id
        self._size += 1

    def close(self):
        self.file.attrs["n_transitions"] = self._size
        self.file.flush()
        self.file.close()