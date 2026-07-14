"""Load a CosmoSIS/Nautilus text chain and draw weighted posterior samples.

Ported from skyvar/tests/chains_tests.ipynb (read_chain_header / canonical_name /
load_chain) so the PPD driver reads chains identically to the presentation notebook.
"""
import numpy as np


def read_chain_header(path):
    raw_columns = None
    metadata = {}
    with open(path) as fp:
        for line in fp:
            if not line.startswith("#"):
                break
            stripped = line[1:].strip()
            if "\t" in stripped and "--" in stripped:
                raw_columns = stripped.split("\t")
            elif "=" in stripped:
                key, value = stripped.split("=", 1)
                metadata[key.strip()] = value.strip()
    if raw_columns is None:
        raise ValueError(f"No column header found in {path}")
    if "n_varied" not in metadata:
        raise ValueError(f"Missing #n_varied in {path}")
    return raw_columns, metadata


def canonical_name(raw_name):
    param = raw_name.split("--", 1)[-1].lower()
    return {
        "sigma_8": "sig8",
        "s_8": "S_8",
        "sigma_12": "sig12",
        "2pt_chi2": "chi2",
    }.get(param, param)


def load_chain(path, fixed=None):
    path = str(path)
    raw_cols, meta = read_chain_header(path)
    names = [canonical_name(c) for c in raw_cols]
    col = {name: i for i, name in enumerate(names)}
    for required in ("log_weight", "post"):
        if required not in col:
            raise ValueError(f"{path} is missing {required!r}")

    data = np.atleast_2d(np.loadtxt(path))
    log_weight = data[:, col["log_weight"]]
    weights = np.zeros_like(log_weight, dtype=float)
    ok = np.isfinite(log_weight)
    if not np.any(ok):
        raise ValueError(f"{path} has no finite log_weight values")
    weights[ok] = np.exp(log_weight[ok] - np.max(log_weight[ok]))

    return {
        "path": path,
        "data": data,
        "weights": weights,
        "col": col,
        "names": names,
        "varied": names[: int(meta["n_varied"])],
        "fixed": dict(fixed or {}),
        "meta": meta,
    }


def map_row(entry):
    """Maximum-posterior (MAP) sample row."""
    post = entry["data"][:, entry["col"]["post"]]
    good = np.isfinite(post)
    idx = np.flatnonzero(good)[np.argmax(post[good])]
    return entry["data"][idx], int(idx)


def sample_dict(entry, row, names):
    """Extract a short-name -> value dict for one chain row (fixed params filled in)."""
    out = {}
    for n in names:
        if n in entry["col"]:
            out[n] = float(row[entry["col"][n]])
        elif n in entry["fixed"]:
            out[n] = float(entry["fixed"][n])
        else:
            raise KeyError(f"{n} not in chain columns or fixed set")
    return out


def thin_weighted(entry, names, n_samples, seed=0):
    """Draw ``n_samples`` posterior draws by systematic weighted resampling.

    Returns (samples ndarray [n_samples, len(names)], list-of-dicts, chosen_row_indices).
    Rows with non-finite params or non-positive weight are excluded first.
    """
    data = entry["data"]
    w = entry["weights"].astype(float).copy()

    mask = np.isfinite(w) & (w > 0)
    for n in names:
        if n in entry["col"]:
            mask &= np.isfinite(data[:, entry["col"][n]])
    idx_pool = np.flatnonzero(mask)
    if idx_pool.size == 0:
        raise ValueError("no usable chain rows")

    wp = w[idx_pool]
    wp = wp / wp.sum()
    cdf = np.cumsum(wp)
    cdf[-1] = 1.0

    rng = np.random.default_rng(seed)
    u0 = rng.uniform(0.0, 1.0 / n_samples)
    positions = u0 + np.arange(n_samples) / n_samples
    chosen_local = np.searchsorted(cdf, positions, side="left")
    chosen_local = np.clip(chosen_local, 0, idx_pool.size - 1)
    chosen = idx_pool[chosen_local]

    dicts = [sample_dict(entry, data[i], names) for i in chosen]
    arr = np.array([[d[n] for n in names] for d in dicts], float)
    return arr, dicts, chosen
