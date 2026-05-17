"""
Targeted mixup → train LGBM (default) and record blind RMSE.

- random / exhaustive shared: **append one CSV row after each trial** (records survive interruption).
- random: random sampling of augmentation count and Y-interval selection.
- exhaustive: grid of total augmentation × nonempty Y-interval subsets × seed list.
- If there are 3+ intervals and priority indices (65–80, ≥80) are selected, apply extra budget
  only to those buckets (may exceed target sum / grid total). Disable with --no-priority-boost.

Usage (from src_tox):
  python MLM_LGBM_augmentation_trials.py --trials 50 --master-seed 0
  python MLM_LGBM_augmentation_trials.py --exhaustive
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Paths for training labels and embeddings (override via args or env if needed)
X_TRAIN = SCRIPT_DIR / "./data_tox/Embeddings/train_mlm.npy"
Y_TRAIN = SCRIPT_DIR / "./data_tox/Embeddings/train_labels.npy"
X_BLIND = SCRIPT_DIR / "./data_tox/Embeddings/blind_mlm.npy"
Y_BLIND = SCRIPT_DIR / "./data_tox/Embeddings/blind_labels.npy"

OUT_DIR = SCRIPT_DIR / "MLM_LGBM_augmentation_trials_out"

# Seven bins used for train vs blind distribution comparison (names are for logging only)
Y_INTERVALS: list[tuple[str, float, float]] = [
    ("very_neg", -45.0, -20.0),
    ("neg_light", -20.0, 0.0),
    ("low_0_15", 0.0, 15.0),
    ("mid_15_45", 15.0, 45.0),
    ("upper_mid_45_65", 45.0, 65.0),
    ("high_65_80", 65.0, 80.0),
    ("very_high_80p", 80.0, 115.0),
]

AUG_TOTAL_MIN_DEFAULT = 80
AUG_TOTAL_MAX_DEFAULT = 250
NUM_INTERVALS_POOL = len(Y_INTERVALS)
NEIGHBOR_DELTA_DEFAULT = 20.0
MAX_TRIES_PER_BUCKET_DEFAULT = 200_000

# Default exhaustive grid: 100–250 step 10 → 16 values; 11 default seeds → 16×127×11 = 22,352 runs
EXHAUSTIVE_AUG_MIN_DEFAULT = 100
EXHAUSTIVE_AUG_MAX_DEFAULT = 250
EXHAUSTIVE_AUG_STEP_DEFAULT = 10
EXHAUSTIVE_SEEDS_DEFAULT = [1, 10, 20, 30, 40, 42, 45, 50, 80, 100, 200]
LGBM_RANDOM_STATE_DEFAULT = 42

# high_65_80=5, very_high_80p=6 (priority interval indices for extra augmentation)
PRIORITY_INTERVAL_INDICES: frozenset[int] = frozenset({5, 6})
PRIORITY_BOOST_MIN_K = 3
PRIORITY_BOOST_MIN_EXTRA = 10
PRIORITY_BOOST_DIVISOR = 6


def equal_split_budgets(n_total: int, k: int) -> list[int]:
    """Split n_total evenly across k intervals as integers (pass remainder +1 from the front)."""
    if k <= 0:
        raise ValueError("k must be positive")
    base, rem = divmod(int(n_total), k)
    return [base + (j < rem) for j in range(k)]


def priority_boost_per_bucket(aug_nominal: int) -> int:
    """Extra augmentation target per selected priority bucket (always positive)."""
    return max(PRIORITY_BOOST_MIN_EXTRA, int(aug_nominal) // PRIORITY_BOOST_DIVISOR)


def apply_priority_bucket_boost(
    base_budgets: list[int],
    picked: tuple[int, ...],
    aug_nominal: int,
    *,
    enabled: bool,
) -> tuple[list[int], int]:
    """
    Add extra targets only when at least PRIORITY_BOOST_MIN_K intervals are selected and a
    priority index is among them (may amplify the grid target sum).
    """
    k = len(picked)
    base = [int(x) for x in base_budgets]
    if not enabled or k < PRIORITY_BOOST_MIN_K:
        return base, 0
    if len(base) != k:
        raise ValueError("base_budgets length must match picked")
    if not any(ii in PRIORITY_INTERVAL_INDICES for ii in picked):
        return base, 0
    bump = priority_boost_per_bucket(aug_nominal)
    extra_sum = 0
    out = base[:]
    for bi, ii in enumerate(picked):
        if ii in PRIORITY_INTERVAL_INDICES:
            out[bi] += bump
            extra_sum += bump
    return out, extra_sum


def nonempty_interval_subsets():
    """Yield all nonempty subsets of the 7 interval indices as sorted tuples (fixed case order)."""
    idx = range(NUM_INTERVALS_POOL)
    for kk in range(1, NUM_INTERVALS_POOL + 1):
        for comb in combinations(idx, kk):
            yield tuple(int(x) for x in comb)


def aug_total_grid(lo: int, hi: int, step: int) -> list[int]:
    if step <= 0 or lo > hi:
        raise ValueError("invalid aug grid")
    return list(range(lo, hi + 1, step))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mixup_numpy(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: float,
    y2: float,
    alpha: float = 0.5,
) -> tuple[np.ndarray, float]:
    x_mix = alpha * x1 + (1.0 - alpha) * x2
    y_mix = alpha * y1 + (1.0 - alpha) * y2
    return x_mix, float(y_mix)


def targeted_mixup_only(
    X: np.ndarray,
    y: np.ndarray,
    target_low: float,
    target_high: float,
    n_new: int,
    neighbor_delta: float = NEIGHBOR_DELTA_DEFAULT,
    max_tries: int = MAX_TRIES_PER_BUCKET_DEFAULT,
    random_state: int = 42,
    mix_alpha_fixed: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mixup from original X, y only → return only new samples to add (may be zero).
    Same acceptance rules as notebook augmentation_works_mlm.targeted_mixup.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    d = X.shape[1]

    target_idx = np.where((y >= target_low) & (y <= target_high))[0]
    if len(target_idx) < 2 or n_new <= 0:
        return np.empty((0, d), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    x_new_list: list[np.ndarray] = []
    y_new_list: list[float] = []
    tries = 0
    while len(y_new_list) < n_new and tries < max_tries:
        tries += 1
        i = int(rng.choice(target_idx))
        cand = np.where(np.abs(y - y[i]) <= neighbor_delta)[0]
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        j = int(rng.choice(cand))
        x_mix, y_mix = mixup_numpy(X[i], X[j], float(y[i]), float(y[j]), alpha=mix_alpha_fixed)
        if target_low <= y_mix <= target_high:
            x_new_list.append(x_mix)
            y_new_list.append(y_mix)

    if not y_new_list:
        return np.empty((0, d), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    X_new = np.stack(x_new_list, axis=0)
    y_new = np.asarray(y_new_list, dtype=y.dtype)
    return X_new, y_new


def sample_interval_allocations(
    rng: np.random.Generator,
    n_total: int,
    k: int,
) -> np.ndarray:
    """k positive integers summing to n_total (multinomial, random proportions)."""
    if k <= 0 or n_total <= 0:
        raise ValueError("n_total and k must be positive")
    p = rng.random(k)
    p = p / p.sum()
    return rng.multinomial(n_total, p)


def run_trial_with_picked(
    trial_id: int,
    *,
    aug_nominal_grid_total: int,
    picked: tuple[int, ...],
    budgets_final: list[int],
    budgets_base: list[int],
    priority_boost_extra_sum: int,
    mixup_rs_base: int,
    X: np.ndarray,
    y: np.ndarray,
    X_blind: np.ndarray,
    y_blind: np.ndarray,
    standardize: bool,
    neighbor_delta: float,
    trial_mode: str,
    master_seed_log: int | None = None,
    lgbm_random_state: int = LGBM_RANDOM_STATE_DEFAULT,
) -> dict:
    """Per picked interval, use budgets_final targets for mixup then LGBM (priority bucket bonus may apply)."""
    k_int = len(picked)

    aug_parts: list[tuple[np.ndarray, np.ndarray, str, float, float, int, int, int]] = []
    for bi, ii in enumerate(picked):
        name, lo, hi = Y_INTERVALS[ii]
        n_req = int(budgets_final[bi])
        seed_bucket = mixup_rs_base * 524_287 + bi * 10_009 + ii * 503 + 1_337
        seed_bucket &= 0x7FFFFFFF
        Xn, yn = targeted_mixup_only(
            X,
            y,
            lo,
            hi,
            n_new=n_req,
            neighbor_delta=neighbor_delta,
            random_state=seed_bucket,
        )
        n_act = len(yn)
        aug_parts.append((Xn, yn, name, lo, hi, n_req, n_act, seed_bucket))

    X_aug_list = [X] + [p[0] for p in aug_parts]
    y_aug_list = [y] + [p[1] for p in aug_parts]
    X_train_aug = np.concatenate(X_aug_list, axis=0)
    y_train_aug = np.concatenate(y_aug_list, axis=0)

    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_train_aug)
        X_bl = scaler.transform(np.asarray(X_blind, dtype=float))
    else:
        X_fit = np.asarray(X_train_aug, dtype=float)
        X_bl = np.asarray(X_blind, dtype=float)

    t0 = time.perf_counter()
    model = LGBMRegressor(random_state=lgbm_random_state)
    model.fit(X_fit, y_train_aug)
    fit_sec = time.perf_counter() - t0

    pred = model.predict(X_bl)
    blind_rmse = rmse(y_blind, pred)

    row: dict[str, object] = {
        "trial_mode": trial_mode,
        "trial_id": trial_id,
        "master_seed": master_seed_log if master_seed_log is not None else mixup_rs_base,
        "mixup_random_state_base": mixup_rs_base,
        "aug_total_target": aug_nominal_grid_total,
        "budget_requested_sum": int(sum(int(x) for x in budgets_final)),
        "budget_base_before_priority_boost": json.dumps([int(x) for x in budgets_base]),
        "priority_boost_extra_sum": int(priority_boost_extra_sum),
        "aug_total_actual": int(sum(p[6] for p in aug_parts)),
        "n_train_after_aug": int(X_train_aug.shape[0]),
        "k_intervals": k_int,
        "interval_names": json.dumps([Y_INTERVALS[i][0] for i in picked], ensure_ascii=False),
        "interval_indices": json.dumps(list(picked)),
        "interval_bounds": json.dumps(
            [[Y_INTERVALS[i][1], Y_INTERVALS[i][2]] for i in picked],
            ensure_ascii=False,
        ),
        "budget_per_interval": json.dumps([int(x) for x in budgets_final]),
        "actual_per_interval": json.dumps([int(p[6]) for p in aug_parts]),
        "bucket_seeds": json.dumps([int(p[7]) for p in aug_parts]),
        "blind_rmse": blind_rmse,
        "fit_time_sec": fit_sec,
        "standardize": standardize,
        "lgbm_random_state": int(lgbm_random_state),
    }
    return row


def run_one_trial(
    trial_id: int,
    master_seed: int,
    X: np.ndarray,
    y: np.ndarray,
    X_blind: np.ndarray,
    y_blind: np.ndarray,
    *,
    aug_total_bounds: tuple[int, int],
    k_interval_bounds: tuple[int, int],
    standardize: bool,
    neighbor_delta: float,
    lgbm_random_state: int = LGBM_RANDOM_STATE_DEFAULT,
    priority_boost_enabled: bool = True,
) -> dict:
    rng_trial = np.random.default_rng(master_seed + 10_003 * trial_id)

    aug_total = int(rng_trial.integers(aug_total_bounds[0], aug_total_bounds[1] + 1))
    k_int = int(
        rng_trial.integers(k_interval_bounds[0], k_interval_bounds[1] + 1),
    )

    picked = rng_trial.choice(NUM_INTERVALS_POOL, size=k_int, replace=False)
    picked = sorted(int(x) for x in picked)

    budgets = sample_interval_allocations(rng_trial, aug_total, k_int)
    budgets_base = [int(budgets[j]) for j in range(k_int)]
    budgets_fin, pb_sum = apply_priority_bucket_boost(
        budgets_base,
        tuple(picked),
        aug_total,
        enabled=priority_boost_enabled,
    )

    mixup_rs = int(rng_trial.integers(0, 2**31 - 1))

    return run_trial_with_picked(
        trial_id,
        aug_nominal_grid_total=aug_total,
        picked=tuple(picked),
        budgets_final=budgets_fin,
        budgets_base=budgets_base,
        priority_boost_extra_sum=pb_sum,
        mixup_rs_base=mixup_rs,
        X=X,
        y=y,
        X_blind=X_blind,
        y_blind=y_blind,
        standardize=standardize,
        neighbor_delta=neighbor_delta,
        trial_mode="random",
        master_seed_log=int(master_seed),
        lgbm_random_state=lgbm_random_state,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixup+LGBM blind RMSE: random or exhaustive search")
    p.add_argument(
        "--exhaustive",
        action="store_true",
        help="Exhaustive: aug total grid × nonempty Y-interval subsets × --exhaustive-seeds",
    )
    p.add_argument(
        "--exhaustive-aug-min",
        type=int,
        default=EXHAUSTIVE_AUG_MIN_DEFAULT,
        help="Exhaustive aug total grid lower bound (inclusive)",
    )
    p.add_argument(
        "--exhaustive-aug-max",
        type=int,
        default=EXHAUSTIVE_AUG_MAX_DEFAULT,
        help="Exhaustive aug total grid upper bound (inclusive)",
    )
    p.add_argument(
        "--exhaustive-aug-step",
        type=int,
        default=EXHAUSTIVE_AUG_STEP_DEFAULT,
        help="Exhaustive aug total grid step",
    )
    p.add_argument(
        "--exhaustive-seeds",
        type=str,
        default=",".join(str(s) for s in EXHAUSTIVE_SEEDS_DEFAULT),
        help="Comma-separated seed list (mixup RNG bases)",
    )
    p.add_argument("--trials", type=int, default=20, help="Used only without --exhaustive")
    p.add_argument("--master-seed", type=int, default=0)
    p.add_argument("--aug-min", type=int, default=AUG_TOTAL_MIN_DEFAULT)
    p.add_argument("--aug-max", type=int, default=AUG_TOTAL_MAX_DEFAULT)
    p.add_argument("--k-min", type=int, default=1, help="Min number of y-intervals per trial (1–7)")
    p.add_argument("--k-max", type=int, default=7)
    p.add_argument("--neighbor-delta", type=float, default=NEIGHBOR_DELTA_DEFAULT)
    p.add_argument(
        "--lgbm-random-state",
        type=int,
        default=LGBM_RANDOM_STATE_DEFAULT,
        help="LGBMRegressor random_state (default 42)",
    )
    p.add_argument(
        "--no-priority-boost",
        action="store_true",
        help="Disable extra targets when intervals≥3 and priority indices (65–80, ≥80) are included",
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not use StandardScaler (default: fit scaler on train+aug then LGBM)",
    )
    p.add_argument(
        "--x-train",
        type=str,
        default=str(X_TRAIN),
    )
    p.add_argument("--y-train", type=str, default=str(Y_TRAIN))
    p.add_argument("--x-blind", type=str, default=str(X_BLIND))
    p.add_argument("--y-blind", type=str, default=str(Y_BLIND))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def _ordered_fieldnames(keys: set[str]) -> list[str]:
    preferred = (
        "trial_mode",
        "trial_id",
        "master_seed",
        "mixup_random_state_base",
        "aug_total_target",
        "budget_requested_sum",
        "priority_boost_extra_sum",
        "blind_rmse",
    )
    fieldnames = sorted(keys)
    out: list[str] = []
    for k in preferred:
        if k in keys:
            out.append(k)
    for k in fieldnames:
        if k not in out:
            out.append(k)
    return out


def _append_csv_row(
    csv_path: Path,
    row: dict,
    csv_state: dict,
) -> None:
    """Called each step: on first row fix header and column order, then append."""
    keys = set(row.keys())
    if csv_state["fieldnames"] is None:
        csv_state["fieldnames"] = _ordered_fieldnames(keys)
    mode = "a" if csv_state["header_written"] else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(
            fp,
            fieldnames=csv_state["fieldnames"],
            extrasaction="ignore",
        )
        if not csv_state["header_written"]:
            w.writeheader()
            csv_state["header_written"] = True
        w.writerow(row)


def _write_summary_json(
    csv_path: Path,
    rows: list[dict],
    *,
    summary_extra: dict,
    ts: str,
) -> dict:
    if not rows:
        raise RuntimeError("No trial rows produced")
    summary_path = csv_path.with_name(csv_path.name.replace("augment_trials", "augment_trials_summary", 1))
    best = min(rows, key=lambda r: float(r["blind_rmse"]))
    payload = {"timestamp": ts, "csv": str(csv_path), "best_rmse_blind": best["blind_rmse"], "best_trial_json": best}
    payload.update(summary_extra)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def init_csv_incremental_state() -> dict:
    return {"fieldnames": None, "header_written": False}


def main() -> None:
    args = parse_args()
    k_min = max(1, min(args.k_min, NUM_INTERVALS_POOL))
    k_max = max(k_min, min(args.k_max, NUM_INTERVALS_POOL))
    aug_min = max(1, args.aug_min)
    aug_max = max(aug_min, args.aug_max)

    X = np.load(args.x_train)
    y = np.load(args.y_train).reshape(-1)
    X_blind = np.load(args.x_blind)
    y_blind = np.load(args.y_blind).reshape(-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    std = not args.no_scale
    nd = args.neighbor_delta

    rows: list[dict] = []
    loop_t0 = time.perf_counter()

    if args.exhaustive:
        seeds = [int(s.strip()) for s in args.exhaustive_seeds.split(",") if s.strip()]
        if not seeds:
            raise SystemExit("--exhaustive-seeds is empty.")

        grids = aug_total_grid(
            args.exhaustive_aug_min,
            args.exhaustive_aug_max,
            args.exhaustive_aug_step,
        )
        subsets = list(nonempty_interval_subsets())

        total_runs = len(grids) * len(subsets) * len(seeds)
        print(
            f"[exhaustive] aug_grid={len(grids)} steps subsets={len(subsets)} seeds={len(seeds)} "
            f"→ total {total_runs} runs",
            flush=True,
        )

        csv_path = out_dir / f"augment_trials_exhaustive_{ts}.csv"
        csv_state = init_csv_incremental_state()
        tid = 0

        for mixup_rs in seeds:
            for aug_total in grids:
                for picked in subsets:
                    budgets_base = equal_split_budgets(aug_total, len(picked))
                    budgets_fin, pb_sum = apply_priority_bucket_boost(
                        budgets_base,
                        picked,
                        int(aug_total),
                        enabled=not args.no_priority_boost,
                    )
                    row = run_trial_with_picked(
                        tid,
                        aug_nominal_grid_total=int(aug_total),
                        picked=picked,
                        budgets_final=budgets_fin,
                        budgets_base=budgets_base,
                        priority_boost_extra_sum=pb_sum,
                        mixup_rs_base=int(mixup_rs),
                        X=X,
                        y=y,
                        X_blind=X_blind,
                        y_blind=y_blind,
                        standardize=std,
                        neighbor_delta=nd,
                        trial_mode="exhaustive",
                        master_seed_log=None,
                        lgbm_random_state=args.lgbm_random_state,
                    )
                    rows.append(row)
                    _append_csv_row(csv_path, row, csv_state)
                    done = tid + 1
                    elapsed = time.perf_counter() - loop_t0
                    rem = total_runs - done
                    eta_s = (elapsed / done) * rem if done > 0 and rem > 0 else 0.
                    eta_m, eta_ss = divmod(int(eta_s), 60)
                    print(
                        f"[done {done}/{total_runs}] aug={aug_total} seed_base={mixup_rs} "
                        f"blind_rmse={row['blind_rmse']:.6f} elapsed={elapsed:.1f}s "
                        f"eta~{eta_m}m{eta_ss:02d}s\n"
                        f"         picked={picked}",
                        flush=True,
                    )
                    tid += 1

        best = _write_summary_json(
            csv_path,
            rows,
            ts=ts,
            summary_extra={
                "mode": "exhaustive",
                "n_trials": total_runs,
                "exhaustive_aug_grid": grids,
                "n_seeds": len(seeds),
                "equal_split_allocation": True,
                "priority_bucket_boost_enabled": not args.no_priority_boost,
                "priority_interval_indices": sorted(PRIORITY_INTERVAL_INDICES),
                "priority_boost_min_k": PRIORITY_BOOST_MIN_K,
                "lgbm_random_state": args.lgbm_random_state,
            },
        )
        print(f"\nSaved: {csv_path}")
        print(f"best blind_rmse: {best['blind_rmse']:.6f} (trial {best['trial_id']})")
        return

    csv_path = out_dir / f"augment_trials_random_{ts}.csv"
    csv_state = init_csv_incremental_state()
    n_tri = args.trials
    for tid in range(n_tri):
        row = run_one_trial(
            tid,
            args.master_seed,
            X,
            y,
            X_blind,
            y_blind,
            aug_total_bounds=(aug_min, aug_max),
            k_interval_bounds=(k_min, k_max),
            standardize=std,
            neighbor_delta=nd,
            lgbm_random_state=args.lgbm_random_state,
            priority_boost_enabled=not args.no_priority_boost,
        )
        rows.append(row)
        _append_csv_row(csv_path, row, csv_state)
        done = tid + 1
        elapsed = time.perf_counter() - loop_t0
        rem = n_tri - done
        eta_s = (elapsed / done) * rem if done > 0 and rem > 0 else 0.0
        eta_m, eta_ss = divmod(int(eta_s), 60)
        print(
            f"[done {done}/{n_tri}] trial_id={tid}  blind_rmse={row['blind_rmse']:.6f}  "
            f"aug_actual={row['aug_total_actual']}/{row['aug_total_target']}  "
            f"elapsed={elapsed:.1f}s  eta~{eta_m}m{eta_ss:02d}s\n"
            f"         intervals={row['interval_names']}",
            flush=True,
        )

    best = _write_summary_json(
        csv_path,
        rows,
        ts=ts,
        summary_extra={
            "mode": "random",
            "n_trials": args.trials,
            "master_seed": args.master_seed,
            "priority_bucket_boost_enabled": not args.no_priority_boost,
            "lgbm_random_state": args.lgbm_random_state,
        },
    )
    print(f"\nSaved: {csv_path}")
    print(f"best blind_rmse: {best['blind_rmse']:.6f} (trial {best['trial_id']})")


if __name__ == "__main__":
    main()
