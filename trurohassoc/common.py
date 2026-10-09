"""Shared I/O helpers. Identifiers are always read as strings (leading zeros and literal 'NA' IDs kept)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {"AMR": "NAM", "America": "NAM", "American": "NAM", "Native_American": "NAM",
           "Europe": "EUR", "European": "EUR", "Africa": "AFR", "African": "AFR",
           "East_Asia": "EAS", "Central_South_Asia": "SAS", "Middle_East": "MDE", "Oceania": "OCE"}
ANCESTRY_NAMES = {"NAM": "Native American", "EUR": "European", "AFR": "African", "EAS": "East Asian",
                  "SAS": "South Asian", "MDE": "Middle Eastern", "OCE": "Oceanian"}


def unique(items):
    return list(dict.fromkeys(items))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=strict_object)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, set)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def numeric(frame, columns):
    for col in columns:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def write_tsv(path, frame):
    frame.to_csv(path, sep="\t", index=False, na_rep="NA",
                 compression="gzip" if str(path).endswith(".gz") else None)


def validate_ids(frame, label, keys=("IID",)):
    if not set(keys).issubset(frame):
        raise ValueError(f"{label} lacks {keys}")
    if frame.duplicated(list(keys)).any() or any(frame[k].eq("").any() for k in keys):
        raise ValueError(f"{label} has duplicate or empty IDs")


def collapse_ancestry(frame, prefix):
    """Harmonise ancestry labels (e.g. Q_American -> Q_NAM) and sum duplicated components."""
    out = frame[["IID"]].copy()
    for col in frame:
        if col.startswith(prefix):
            name = prefix + ALIASES.get(col[len(prefix):], col[len(prefix):])
            values = pd.to_numeric(frame[col], errors="coerce")
            out[name] = out[name] + values if name in out else values
    return out


def check_proportions(frame, columns, label):
    values = frame[columns].to_numpy(float)
    if (not columns or not np.isfinite(values).all() or (values < -1e-8).any()
            or (values > 1 + 1e-8).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-4)):
        raise ValueError(f"Invalid {label} ancestry proportions (must be finite, in [0,1] and sum to 1)")
