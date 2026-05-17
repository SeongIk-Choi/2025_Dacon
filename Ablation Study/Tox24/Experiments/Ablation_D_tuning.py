"""
LGBM blind-guided search V3 for five augmented datasets.

Performs the same grid search over five augmented training files with aug_total = 120/140/160/190/200
Fixes n_estimators to 5000
Applies early stopping during fitting using the leaderboard eval_set
Saves an all-trials CSV and a summary CSV for each aug_total
Saves an integrated best-report JSON across all aug_total settings
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TAG = "MLM"
DATA_TYPE = "MIN"
EMBEDDING_TAG = TAG

AUG_TOTAL_LIST = [120, 140, 160, 190, 200]

X_LD = "../data_tox/Embeddings/leaderboard_mlm.npy"
Y_LD = "../data_tox/Embeddings/leaderboard_labels.npy"
X_BL = "../data_tox/Embeddings/blind_mlm.npy"
Y_BL = "../data_tox/Embeddings/blind_labels.npy"

OUT_DIR = "./MLM_LGBM_GRIDAUG5_gridsearch_LDBL_V3"
STANDARD_SCALE_X = True
LGBM_N_JOBS = 10
AUG_PARALLEL_WORKERS = len(AUG_TOTAL_LIST)

SELECTION_TARGET = "blind"  # "blind" or "leaderboard"
EARLY_STOPPING_ROUNDS = 200

# V3 핵심: n_estimators 고정
N_ESTIMATORS_FIXED = 3000

PARAM_GRID: dict = {
    "learning_rate": [0.1, 0.05, 0.01],
    "num_leaves": [31, 24, 50, 63],
    "min_child_samples": [20, 15, 30, 40],
    "subsample": [1.0, 0.8, 0.9],
    "colsample_bytree": [1.0, 0.9, 0.8],
    "reg_lambda": [0.0, 0.1, 0.2],
    "reg_alpha": [0.0, 0.1, 0.2],
    "max_depth": [-1, 10, 8]
}
# ---------------------------------------------------------------------------


def load_npy(path: str) -> np.ndarray:
    p = str((SCRIPT_DIR / path).resolve()) if not Path(path).is_absolute() else path
    if not Path(p).is_file():
        raise FileNotFoundError(p)
    return np.load(p)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _trial_sort_key(
    trial: tuple[float, float, dict, int],
    selection_target: str,
) -> tuple[float, float]:
    blind_rmse, ld_rmse, _, _ = trial
    return (blind_rmse, ld_rmse) if selection_target == "blind" else (ld_rmse, blind_rmse)


def _train_one_trial(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_ld: np.ndarray,
    y_ld: np.ndarray,
    X_bl: np.ndarray,
    y_bl: np.ndarray,
    random_state: int,
) -> tuple[float, float, dict, int]:
    es_cb = early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
    model = LGBMRegressor(
        random_state=random_state,
        n_jobs=LGBM_N_JOBS,
        verbose=-1,
        n_estimators=N_ESTIMATORS_FIXED,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_bl, y_bl)],
        eval_metric="rmse",
        callbacks=[
            es_cb,
        ],
    )

    pred_ld = model.predict(X_ld)
    pred_bl = model.predict(X_bl)
    ld_rmse = rmse(y_ld, pred_ld)
    blind_rmse = rmse(y_bl, pred_bl)
    best_iter = int(model.best_iteration_ or N_ESTIMATORS_FIXED)
    return blind_rmse, ld_rmse, params, best_iter


def _paths_for_aug_total(aug_total: int) -> tuple[str, str]:
    x_train = f"../data_tox/Embeddings/train_mlm_aug_trial_{aug_total}.npy"
    y_train = f"../data_tox/Embeddings/train_mlm_labels_aug_trial_{aug_total}.npy"
    return x_train, y_train


def run_for_one_aug_total(
    aug_total: int,
    *,
    out_dir: Path,
    selection_target: str,
    random_state: int,
) -> dict:
    X_ld = load_npy(X_LD)
    y_ld = load_npy(Y_LD).reshape(-1)
    X_bl = load_npy(X_BL)
    y_bl = load_npy(Y_BL).reshape(-1)

    x_train_path, y_train_path = _paths_for_aug_total(aug_total)
    X_train = load_npy(x_train_path)
    y_train = load_npy(y_train_path).reshape(-1)

    X_ld_local = X_ld
    X_bl_local = X_bl
    if STANDARD_SCALE_X:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_ld_local = scaler.transform(X_ld)
        X_bl_local = scaler.transform(X_bl)

    param_list = list(ParameterGrid(PARAM_GRID))
    print(f"[aug_total={aug_total}] grid trials={len(param_list)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stream_csv = out_dir / f"lgbm_blindsearch_v3_aug{aug_total}_stream_{ts}.csv"
    stream_fields = [
        "trial_idx",
        "n_trials_total",
        "aug_total",
        "blind_rmse",
        "leaderboard_rmse",
        "best_iteration",
        "selection_target",
        "selection_rmse",
        "best_blind_so_far",
        "best_leaderboard_so_far",
        "params_json",
    ]

    all_trials: list[tuple[float, float, dict, int]] = []
    best_so_far: tuple[float, float, dict, int] | None = None
    n_total = len(param_list)
    with open(stream_csv, "w", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(sf, fieldnames=stream_fields)
        writer.writeheader()
        sf.flush()

        for idx, params in enumerate(param_list, start=1):
            trial = _train_one_trial(
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_ld=X_ld_local,
                y_ld=y_ld,
                X_bl=X_bl_local,
                y_bl=y_bl,
                random_state=random_state,
            )
            all_trials.append(trial)
            b, l, p, bi = trial

            if best_so_far is None or _trial_sort_key(trial, selection_target) < _trial_sort_key(
                best_so_far, selection_target
            ):
                best_so_far = trial

            sel_rmse = b if selection_target == "blind" else l
            best_b, best_l, _, _ = best_so_far

            print(
                f"[aug={aug_total} trial {idx:>4}/{n_total}] "
                f"blind={b:.6f} ld={l:.6f} best_iter={bi} "
                f"| best_so_far blind={best_b:.6f} ld={best_l:.6f}",
                flush=True,
            )

            writer.writerow(
                {
                    "trial_idx": idx,
                    "n_trials_total": n_total,
                    "aug_total": aug_total,
                    "blind_rmse": f"{b:.8f}",
                    "leaderboard_rmse": f"{l:.8f}",
                    "best_iteration": bi,
                    "selection_target": selection_target,
                    "selection_rmse": f"{sel_rmse:.8f}",
                    "best_blind_so_far": f"{best_b:.8f}",
                    "best_leaderboard_so_far": f"{best_l:.8f}",
                    "params_json": json.dumps(p, sort_keys=True),
                }
            )
            sf.flush()

    all_trials = sorted(all_trials, key=lambda t: _trial_sort_key(t, selection_target))
    best_bl, best_ld, best_params, best_iter = all_trials[0]

    trials_df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "aug_total": aug_total,
                "blind_rmse": bl,
                "leaderboard_rmse": ld,
                "best_iteration": bi,
                "params_json": json.dumps(p, sort_keys=True),
            }
            for i, (bl, ld, p, bi) in enumerate(all_trials)
        ]
    )
    trials_csv = out_dir / f"lgbm_blindsearch_v3_aug{aug_total}_alltrials_{ts}.csv"
    trials_df.to_csv(trials_csv, index=False)

    best_model = LGBMRegressor(
        random_state=random_state,
        n_jobs=LGBM_N_JOBS,
        verbose=-1,
        n_estimators=N_ESTIMATORS_FIXED,
        **best_params,
    )
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_ld_local, y_ld)],
        eval_metric="rmse",
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    y_ld_pred = best_model.predict(X_ld_local)
    y_bl_pred = best_model.predict(X_bl_local)
    ld_r2 = r2_score(y_ld, y_ld_pred)
    bl_r2 = r2_score(y_bl, y_bl_pred)

    summary_row = {
        "aug_total": aug_total,
        "selection_target": selection_target,
        "blind_rmse_trial_best": float(best_bl),
        "leaderboard_rmse_trial_best": float(best_ld),
        "best_iteration_trial_best": int(best_iter),
        "leaderboard_r2_refit": float(ld_r2),
        "leaderboard_rmse_refit": float(rmse(y_ld, y_ld_pred)),
        "blind_r2_refit": float(bl_r2),
        "blind_rmse_refit": float(rmse(y_bl, y_bl_pred)),
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "stream_csv": str(stream_csv),
        "alltrials_csv": str(trials_csv),
    }
    return summary_row


def main() -> None:
    if SELECTION_TARGET not in ("blind", "leaderboard"):
        raise ValueError("SELECTION_TARGET must be 'blind' or 'leaderboard'")

    out_dir = (SCRIPT_DIR / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X_ld = load_npy(X_LD)
    y_ld = load_npy(Y_LD).reshape(-1)
    X_bl = load_npy(X_BL)
    y_bl = load_npy(Y_BL).reshape(-1)

    print(f"[info] X_ld={X_ld.shape}, y_ld={y_ld.shape}")
    print(f"[info] X_bl={X_bl.shape}, y_bl={y_bl.shape}")
    print(
        f"[info] n_estimators={N_ESTIMATORS_FIXED} fixed, early_stopping_rounds={EARLY_STOPPING_ROUNDS}"
    )

    print(f"[info] per-model n_jobs={LGBM_N_JOBS}, aug_workers={AUG_PARALLEL_WORKERS}")

    summary_rows = []
    max_workers = max(1, min(int(AUG_PARALLEL_WORKERS), len(AUG_TOTAL_LIST)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_for_one_aug_total,
                aug_total=aug_total,
                out_dir=out_dir,
                selection_target=SELECTION_TARGET,
                random_state=42,
            ): aug_total
            for aug_total in AUG_TOTAL_LIST
        }
        for fut in as_completed(futures):
            aug_total = futures[fut]
            row = fut.result()
            summary_rows.append(row)
            print(
                f"[done] aug_total={aug_total} "
                f"blind_best={row['blind_rmse_trial_best']:.6f} "
                f"ld_best={row['leaderboard_rmse_trial_best']:.6f}",
                flush=True,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="blind_rmse_trial_best" if SELECTION_TARGET == "blind" else "leaderboard_rmse_trial_best",
        ascending=True,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = out_dir / f"lgbm_blindsearch_v3_summary_{DATA_TYPE}_{EMBEDDING_TAG}_{ts}.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_row = summary_df.iloc[0].to_dict()
    report = {
        "version": "v3",
        "target": SELECTION_TARGET,
        "aug_totals": AUG_TOTAL_LIST,
        "n_estimators_fixed": N_ESTIMATORS_FIXED,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "param_grid": PARAM_GRID,
        "best_summary_row": best_row,
        "summary_csv": str(summary_csv),
    }
    report_json = out_dir / f"lgbm_blindsearch_v3_report_{DATA_TYPE}_{EMBEDDING_TAG}_{ts}.json"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[out] {summary_csv}")
    print(f"[out] {report_json}")
    print("[done] V3 grid search configuration file prepared.")


if __name__ == "__main__":
    main()
