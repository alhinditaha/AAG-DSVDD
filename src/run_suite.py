#!/usr/bin/env python3
"""
run_suite.py — Orchestrator for run_methods_simple.py

• Controls ALL parameters of your pipeline:
  - Dataset generation (banana / moons) + shapes & noise
  - Splits & semi-supervised labeling policy
  - Model toggles: AAG-DSVDD (proposed), DeepSVDD, DeepSAD, OC-SVM, SVDD-RBF
  - Full hyper-parameters for each model
  - Device/plotting/output & CSV append path
  - Multi-seed runs (single call to run_methods_simple.py per param combo)
  - Grid sweeps (Cartesian "explode" across lists)
  - Optional parallel execution + command logging

Usage (single one-shot config):
    python run_suite.py

Usage (grid over several values; set *_LISTS below and GRID_EXPLODE=True):
    python run_suite.py

Notes:
  - This runner does NOT parse CLI itself; you edit the CONFIG block below.
  - All flags passed here map 1:1 to run_methods_simple.py’s CLI.
"""

import os
import shlex
import subprocess
from pathlib import Path
from itertools import product
from typing import Iterable, List, Union, Optional, Tuple
from datetime import datetime
import concurrent.futures as fut

# =============================================================================
# PATHS
# =============================================================================

# Path to the main script you already have (this repo)
HERE = Path(__file__).resolve().parent
MAIN = HERE / "run_methods.py"   # << change if your file lives elsewhere
PY   = "python"                         # interpreter (e.g., "python3" or absolute)

# =============================================================================
# GLOBAL EXECUTION CONTROLS
# =============================================================================

EXECUTE: bool = True            # True -> actually run; False -> print commands only
PARALLEL: bool = False          # True -> parallelize when GRID_EXPLODE=True
N_WORKERS: int = 4              # number of parallel workers if PARALLEL=True
SAVE_CMDS_TO: Optional[str] = "run_commands.log"   # None -> don’t save

# =============================================================================
# HIGH-LEVEL RUN MODES
# =============================================================================

ONE_SHOT: bool      = True      # Run ONE configuration (uses FIRST value of each list below)
GRID_EXPLODE: bool  = False     # Cartesian product across any *_LISTS below; one command per combo

# =============================================================================
# COMMON OUTPUT / LOGGING / DEVICE / PLOTTING
# =============================================================================
DEVICE: str         = "cpu"                     # "cpu" or "cuda"
OUT_DIR: str        = "plots"                   # where plots are saved
CSV_PATH: str       = "logs/results_simple.csv" # appended by run_methods_simple
PLOT: bool          = True                      # True: plot decision boundaries
GRID_RES: int       = 300                       # plot heatmap resolution
PRINT_EVERY: int    = 1                         # training logs print interval inside methods

# =============================================================================
# DATASET & LABELING CONFIG (lists = sweep; first value used in ONE_SHOT)
# =============================================================================

# Which dataset(s) to run
DATASETS: List[str] = ["banana"]                # ["banana"], ["moons"], or ["banana","moons"]

# Random seeds (passed as comma to --seeds; run_methods_simple loops internally)
SEEDS: List[int] = [0,1,2,3,4,5,6,7,8,9]                          # e.g., [0,1,2,3]

# --- Global sample counts & noise (used by both datasets) ---
N_NORM_LIST:   List[int]   = [1200]              # number of normal points
N_ANOM_LIST:   List[int]   = [800]              # number of anomaly points
NOISE_LIST:    List[float] = [0.06]             # gaussian jitter

# --- Banana-only shape controls ---
BEND_LIST:     List[float] = [0.2]              # curvature; ↑ => more banana curve
SCALE_X_LIST:  List[float] = [1.0]              # stretch along x
SCALE_Y_LIST:  List[float] = [1.0]              # stretch along y

# >>> Advanced Banana (Friedman-style) — optional; appended only for dataset="banana"
BANANA_B_LIST:            List[float]        = [0.25]         # Friedman curvature 'b'
BANANA_S1_LIST:           List[float]        = [2.0]         # std(u1)
BANANA_S2_LIST:           List[float]        = [1.5]         # std(u2)
BANANA_ROTATE_DEG_LIST:   List[float]        = [0]        # rotate after generation
BANANA_ANOM_SPLIT_LIST:   List[float]        = [0.5]         # frac anomalies in lobe-1
BANANA_MU_A1_LIST:        List[Tuple[float,float]] = [(0.0, 3.5)]
BANANA_MU_A2_LIST:        List[Tuple[float,float]] = [(0.0, -4)]
BANANA_COV_A1_LIST:       List[Tuple[float,float]] = [(0.5, 0.75)]  # diag variances (vxx, vyy)
BANANA_COV_A2_LIST:       List[Tuple[float,float]] = [(0.5, 0.75)]

# --- Moons-only anomaly cloud controls ---
GAP_LIST:      List[float] = [0.5]              # vertical shift of anomaly Gaussian
SPREAD_LIST:   List[float] = [0.6]              # std of anomaly Gaussian

# --- Splits & semi-supervised labeling ---
TEST_SIZE_LIST:            List[float] = [0.30] # test proportion
VAL_SIZE_LIST:             List[float] = [0.20] # validation proportion
LABEL_FRAC_ANOM_LIST:      List[float] = [0.1]  # fraction of train anomalies labeled as -1
LABEL_FRAC_NORM_LIST:      List[float] = [0.1]  # fraction of train normals labeled as +1 (new)
LABEL_NORMALS_SUBSET_ONLY: List[bool]  = [True]# True => DO NOT label all normals (+1); they remain unlabeled

# =============================================================================
# MODEL TOGGLES — select which baselines to run
# =============================================================================
RUN_AAG_LIST:      List[bool] = [True]          # proposed AAG-DSVDD
RUN_DEEPSVDD_LIST: List[bool] = [True]
RUN_DEEPSAD_LIST:  List[bool] = [True]
RUN_OCSVM_LIST:    List[bool] = [True]
RUN_KSVDD_LIST:    List[bool] = [True]

# =============================================================================
# AAG-DSVDD (proposed) HYPERS (match TrainConfig in src/models/aag_dsvdd.py)
# =============================================================================
# Hidden layer sizes (comma string); e.g., "128,64"
AAG_HIDDEN_LIST:        List[str]   = ["256,128"]
AAG_OUT_DIM_LIST:       List[int]   = [16]      # embedding dim
AAG_P_LIST:             List[int]   = [2]       # hinge power: 1 (linear) or 2 (squared hinge)
AAG_NU_LIST:            List[float] = [0.05]     # SVDD nu (quantile for R^2 update)
AAG_OMEGA_LIST:         List[float] = [2.0]     # weight on labeled-anomaly hinge
AAG_MARGIN_LIST:        List[float] = [1.0]     # margin m for anomalies
AAG_LAMBDA_U_LIST:      List[float] = [0.1]     # graph Laplacian weight
AAG_K_LIST:             List[int]   = [15]      # k in kNN graph
AAG_GAMMA_EDGES_LIST:   List[float] = [1.0]     # upweight edges touching anomalies (>=1)
AAG_GRAPH_REFRESH_LIST: List[int]   = [2]       # epochs between graph rebuilds
AAG_ETA_UNL_LIST:       List[float] = [1.0]     # DeepSAD-style unlabeled pull-in weight
AAG_CAP_UNLABELED_LIST: List[bool]  = [False]   # cap unlabeled d^2 by (R^2 + cap_offset)
AAG_CAP_OFFSET_LIST:    List[float] = [0.5]     # cap offset if enabled
AAG_LR_LIST:            List[float] = [1e-3]    # learning rate
AAG_WD_LIST:            List[float] = [1e-4]    # weight decay
AAG_EPOCHS_LIST:        List[int]   = [50]      # epochs
AAG_BATCH_LIST:         List[int]   = [128]     # batch size
AAG_WARMUP_LIST:        List[int]   = [2]       # warmup epochs (center init)

# =============================================================================
# DeepSVDD HYPERS
# =============================================================================
DSVDD_HIDDEN_LIST:   List[str]   = ["256,128"]   # "64,32" means two hidden layers
DSVDD_REP_DIM_LIST:  List[int]   = [16]         # embedding dim
DSVDD_NU_LIST:       List[float] = [0.05]        # soft-boundary nu
DSVDD_LR_LIST:       List[float] = [1e-3]
DSVDD_WD_LIST:       List[float] = [1e-6]
DSVDD_EPOCHS_LIST:   List[int]   = [50]
DSVDD_BATCH_LIST:    List[int]   = [128]

# =============================================================================
# DeepSAD HYPERS
# =============================================================================
DSAD_HIDDEN_LIST:   List[str]   = ["256,128"]
DSAD_REP_DIM_LIST:  List[int]   = [16]
DSAD_NU_LIST:       List[float] = [0.05]
DSAD_LR_LIST:       List[float] = [1e-3]
DSAD_WD_LIST:       List[float] = [1e-6]
DSAD_EPOCHS_LIST:   List[int]   = [50]
DSAD_BATCH_LIST:    List[int]   = [128]

# =============================================================================
# OC-SVM HYPERS
# =============================================================================
OCSVM_NU_LIST:     List[float]    = [0.05]      # expected outlier fraction
OCSVM_GAMMA_LIST:  List[Union[str,float]] = ["scale"]  # 'scale' or numeric bandwidth

# =============================================================================
# Kernel SVDD (RBF) HYPERS
# =============================================================================
KSVDD_NU_LIST:     List[float] = [0.05]
KSVDD_GAMMA_LIST:  List[float] = [1]          # RBF gamma

# =============================================================================
# UTILITIES
# =============================================================================

def _csv(xs: Iterable) -> str:
    return ",".join(str(x) for x in xs)

def _first(xs: Iterable):
    return list(xs)[0]

def _bool_flag(flag: str, enabled: bool) -> str:
    """Return flag (like '--plot') if enabled, else empty string."""
    return f" {flag}" if enabled else ""

def _fmt_pair(p: Tuple[float, float]) -> str:
    """Format a pair for CLI as '\"x,y\"'."""
    return f'"{p[0]},{p[1]}"'

def _log(cmd: str):
    print(cmd)
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(cmd + "\n")

def _run(cmd: str):
    _log(cmd)
    if not EXECUTE:
        return
    try:
        res = subprocess.run(shlex.split(cmd), check=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
    except subprocess.CalledProcessError as e:
        print("RUN FAILED:")
        if e.stdout: print(e.stdout)
        if e.stderr: print(e.stderr)

# =============================================================================
# COMMAND BUILDERS
# =============================================================================

def build_base_cmd(
    dataset: str,
    seeds: List[int],
    n_norm: int, n_anom: int, noise: float,
    bend: float, scale_x: float, scale_y: float,     # banana
    gap: float, spread: float,                       # moons
    test_size: float, val_size: float,
    label_frac_anom: float, label_frac_norm: float, label_normals_subset_only: bool,
    # advanced banana (optional)
    banana_b: float, banana_s1: float, banana_s2: float, banana_rotate_deg: float,
    banana_anom_split: float,
    banana_mu_a1: Tuple[float,float], banana_mu_a2: Tuple[float,float],
    banana_cov_a1: Tuple[float,float], banana_cov_a2: Tuple[float,float],
) -> str:
    """Compose the common part of the command for run_methods_simple.py."""
    cmd = (
        f'{PY} "{MAIN}"'
        f' --dataset {dataset}'
        f' --seeds "{_csv(seeds)}"'
        f' --device {DEVICE}'
        f' --out-dir {OUT_DIR}'
        f' --csv-path {CSV_PATH}'
        f' --grid-res {GRID_RES}'
        f' --print-every {PRINT_EVERY}'
        f'{_bool_flag(" --plot", PLOT)}'
        f' --n-norm {n_norm}'
        f' --n-anom {n_anom}'
        f' --noise {noise}'
        f' --test-size {test_size}'
        f' --val-size {val_size}'
        f' --label-frac-anom {label_frac_anom}'
        f' --label-frac-norm {label_frac_norm}'
    )
    if label_normals_subset_only:
        cmd += " --label-normals-subset-only"
    if dataset == "banana":
        cmd += f' --bend {bend} --scale-x {scale_x} --scale-y {scale_y}'
        # append advanced banana controls (only if your run_methods_simple supports them)
        cmd += (
            f' --banana-b {banana_b}'
            f' --banana-s1 {banana_s1}'
            f' --banana-s2 {banana_s2}'
            f' --banana-rotate-deg {banana_rotate_deg}'
            f' --banana-anom-split {banana_anom_split}'
            f' --banana-mu-a1 {_fmt_pair(banana_mu_a1)}'
            f' --banana-mu-a2 {_fmt_pair(banana_mu_a2)}'
            f' --banana-cov-a1 {_fmt_pair(banana_cov_a1)}'
            f' --banana-cov-a2 {_fmt_pair(banana_cov_a2)}'
        )
    elif dataset == "moons":
        cmd += f' --gap {gap} --spread {spread}'
    return cmd

def add_model_toggles(cmd: str,
                      run_aag: bool, run_dsvdd: bool, run_dsad: bool, run_ocsvm: bool, run_ksvdd: bool) -> str:
    if run_aag:    cmd += " --run-aag"
    if run_dsvdd:  cmd += " --run-deepsvdd"
    if run_dsad:   cmd += " --run-deepsad"
    if run_ocsvm:  cmd += " --run-ocsvm"
    if run_ksvdd:  cmd += " --run-ksvdd"
    return cmd

def add_aag_flags(cmd: str,
                  hidden: str, out_dim: int, p: int, nu: float, Omega: float, margin: float,
                  lambda_u: float, k: int, gamma_edges: float, graph_refresh: int,
                  eta_unl: float, cap_unlabeled: bool, cap_offset: float,
                  lr: float, wd: float, epochs: int, batch: int, warmup: int) -> str:
    return (cmd
            + f' --aag-hidden "{hidden}"'
            + f' --aag-out-dim {out_dim}'
            + f' --aag-p {p}'
            + f' --aag-nu {nu}'
            + f' --aag-Omega {Omega}'
            + f' --aag-margin {margin}'
            + f' --aag-lambda-u {lambda_u}'
            + f' --aag-k {k}'
            + f' --aag-gamma-edges {gamma_edges}'
            + f' --aag-graph-refresh {graph_refresh}'
            + f' --aag-eta-unl {eta_unl}'
            + (' --aag-cap-unlabeled' if cap_unlabeled else '')
            + f' --aag-cap-offset {cap_offset}'
            + f' --aag-lr {lr}'
            + f' --aag-wd {wd}'
            + f' --aag-epochs {epochs}'
            + f' --aag-batch {batch}'
            + f' --aag-warmup {warmup}')

def add_dsvdd_flags(cmd: str,
                    hidden: str, rep_dim: int, nu: float, lr: float, wd: float, epochs: int, batch: int) -> str:
    return (cmd
            + f' --dsvdd-hidden "{hidden}"'
            + f' --dsvdd-rep-dim {rep_dim}'
            + f' --dsvdd-nu {nu}'
            + f' --dsvdd-lr {lr}'
            + f' --dsvdd-wd {wd}'
            + f' --dsvdd-epochs {epochs}'
            + f' --dsvdd-batch {batch}')

def add_dsad_flags(cmd: str,
                   hidden: str, rep_dim: int, nu: float, lr: float, wd: float, epochs: int, batch: int) -> str:
    return (cmd
            + f' --dsad-hidden "{hidden}"'
            + f' --dsad-rep-dim {rep_dim}'
            + f' --dsad-nu {nu}'
            + f' --dsad-lr {lr}'
            + f' --dsad-wd {wd}'
            + f' --dsad-epochs {epochs}'
            + f' --dsad-batch {batch}')

def add_ocsvm_flags(cmd: str, nu: float, gamma: Union[str, float]) -> str:
    return cmd + f' --ocsvm-nu {nu} --ocsvm-gamma {gamma}'

def add_ksvdd_flags(cmd: str, nu: float, gamma: float) -> str:
    return cmd + f' --ksvdd-nu {nu} --ksvdd-gamma {gamma}'

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    # separate log section
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(f"\n# ===== {datetime.now().isoformat()} =====\n")

    def one_command(dataset: str,
                    seeds: List[int],
                    n_norm: int, n_anom: int, noise: float,
                    bend: float, scale_x: float, scale_y: float,
                    gap: float, spread: float,
                    test_size: float, val_size: float,
                    label_frac_anom: float, label_frac_norm: float, label_normals_subset_only: bool,
                    run_aag: bool, run_dsvdd: bool, run_dsad: bool, run_ocsvm: bool, run_ksvdd: bool,
                    aag_hidden: str, aag_out: int, aag_p: int, aag_nu: float, aag_Omega: float, aag_margin: float,
                    aag_lambda_u: float, aag_k: int, aag_gamma_edges: float, aag_graph_refresh: int,
                    aag_eta_unl: float, aag_cap_unlabeled: bool, aag_cap_offset: float,
                    aag_lr: float, aag_wd: float, aag_epochs: int, aag_batch: int, aag_warmup: int,
                    dsvdd_hidden: str, dsvdd_rep: int, dsvdd_nu: float, dsvdd_lr: float, dsvdd_wd: float, dsvdd_epochs: int, dsvdd_batch: int,
                    dsad_hidden: str, dsad_rep: int, dsad_nu: float, dsad_lr: float, dsad_wd: float, dsad_epochs: int, dsad_batch: int,
                    ocsvm_nu: float, ocsvm_gamma: Union[str, float],
                    ksvdd_nu: float, ksvdd_gamma: float,
                    # advanced banana
                    banana_b: float, banana_s1: float, banana_s2: float, banana_rotate_deg: float,
                    banana_anom_split: float,
                    banana_mu_a1: Tuple[float,float], banana_mu_a2: Tuple[float,float],
                    banana_cov_a1: Tuple[float,float], banana_cov_a2: Tuple[float,float]
                    ) -> str:

        cmd = build_base_cmd(dataset, seeds, n_norm, n_anom, noise,
                             bend, scale_x, scale_y, gap, spread,
                             test_size, val_size, label_frac_anom, label_frac_norm, label_normals_subset_only,
                             banana_b, banana_s1, banana_s2, banana_rotate_deg,
                             banana_anom_split, banana_mu_a1, banana_mu_a2, banana_cov_a1, banana_cov_a2)
        cmd = add_model_toggles(cmd, run_aag, run_dsvdd, run_dsad, run_ocsvm, run_ksvdd)
        # Hyper-params
        if run_aag:
            cmd = add_aag_flags(cmd, aag_hidden, aag_out, aag_p, aag_nu, aag_Omega, aag_margin,
                                aag_lambda_u, aag_k, aag_gamma_edges, aag_graph_refresh,
                                aag_eta_unl, aag_cap_unlabeled, aag_cap_offset,
                                aag_lr, aag_wd, aag_epochs, aag_batch, aag_warmup)
        if run_dsvdd:
            cmd = add_dsvdd_flags(cmd, dsvdd_hidden, dsvdd_rep, dsvdd_nu, dsvdd_lr, dsvdd_wd, dsvdd_epochs, dsvdd_batch)
        if run_dsad:
            cmd = add_dsad_flags(cmd, dsad_hidden, dsad_rep, dsad_nu, dsad_lr, dsad_wd, dsad_epochs, dsad_batch)
        if run_ocsvm:
            cmd = add_ocsvm_flags(cmd, ocsvm_nu, ocsvm_gamma)
        if run_ksvdd:
            cmd = add_ksvdd_flags(cmd, ksvdd_nu, ksvdd_gamma)
        return cmd

    cmds: List[str] = []

    # -----------------------------
    # ONE SHOT (first value of each)
    # -----------------------------
    if ONE_SHOT:
        for dataset in DATASETS:
            cmd = one_command(
                dataset=dataset,
                seeds=SEEDS,
                n_norm=_first(N_NORM_LIST), n_anom=_first(N_ANOM_LIST), noise=_first(NOISE_LIST),
                bend=_first(BEND_LIST), scale_x=_first(SCALE_X_LIST), scale_y=_first(SCALE_Y_LIST),
                gap=_first(GAP_LIST), spread=_first(SPREAD_LIST),
                test_size=_first(TEST_SIZE_LIST), val_size=_first(VAL_SIZE_LIST),
                label_frac_anom=_first(LABEL_FRAC_ANOM_LIST),
                label_frac_norm=_first(LABEL_FRAC_NORM_LIST),
                label_normals_subset_only=_first(LABEL_NORMALS_SUBSET_ONLY),
                run_aag=_first(RUN_AAG_LIST), run_dsvdd=_first(RUN_DEEPSVDD_LIST),
                run_dsad=_first(RUN_DEEPSAD_LIST), run_ocsvm=_first(RUN_OCSVM_LIST),
                run_ksvdd=_first(RUN_KSVDD_LIST),
                aag_hidden=_first(AAG_HIDDEN_LIST), aag_out=_first(AAG_OUT_DIM_LIST),
                aag_p=_first(AAG_P_LIST), aag_nu=_first(AAG_NU_LIST),
                aag_Omega=_first(AAG_OMEGA_LIST), aag_margin=_first(AAG_MARGIN_LIST),
                aag_lambda_u=_first(AAG_LAMBDA_U_LIST), aag_k=_first(AAG_K_LIST),
                aag_gamma_edges=_first(AAG_GAMMA_EDGES_LIST), aag_graph_refresh=_first(AAG_GRAPH_REFRESH_LIST),
                aag_eta_unl=_first(AAG_ETA_UNL_LIST), aag_cap_unlabeled=_first(AAG_CAP_UNLABELED_LIST),
                aag_cap_offset=_first(AAG_CAP_OFFSET_LIST),
                aag_lr=_first(AAG_LR_LIST), aag_wd=_first(AAG_WD_LIST),
                aag_epochs=_first(AAG_EPOCHS_LIST), aag_batch=_first(AAG_BATCH_LIST),
                aag_warmup=_first(AAG_WARMUP_LIST),
                dsvdd_hidden=_first(DSVDD_HIDDEN_LIST), dsvdd_rep=_first(DSVDD_REP_DIM_LIST),
                dsvdd_nu=_first(DSVDD_NU_LIST), dsvdd_lr=_first(DSVDD_LR_LIST),
                dsvdd_wd=_first(DSVDD_WD_LIST), dsvdd_epochs=_first(DSVDD_EPOCHS_LIST),
                dsvdd_batch=_first(DSVDD_BATCH_LIST),
                dsad_hidden=_first(DSAD_HIDDEN_LIST), dsad_rep=_first(DSAD_REP_DIM_LIST),
                dsad_nu=_first(DSAD_NU_LIST), dsad_lr=_first(DSAD_LR_LIST),
                dsad_wd=_first(DSAD_WD_LIST), dsad_epochs=_first(DSAD_EPOCHS_LIST),
                dsad_batch=_first(DSAD_BATCH_LIST),
                ocsvm_nu=_first(OCSVM_NU_LIST), ocsvm_gamma=_first(OCSVM_GAMMA_LIST),
                ksvdd_nu=_first(KSVDD_NU_LIST), ksvdd_gamma=_first(KSVDD_GAMMA_LIST),
                # advanced banana
                banana_b=_first(BANANA_B_LIST), banana_s1=_first(BANANA_S1_LIST),
                banana_s2=_first(BANANA_S2_LIST), banana_rotate_deg=_first(BANANA_ROTATE_DEG_LIST),
                banana_anom_split=_first(BANANA_ANOM_SPLIT_LIST),
                banana_mu_a1=_first(BANANA_MU_A1_LIST), banana_mu_a2=_first(BANANA_MU_A2_LIST),
                banana_cov_a1=_first(BANANA_COV_A1_LIST), banana_cov_a2=_first(BANANA_COV_A2_LIST),
            )
            cmds.append(cmd)

    # -----------------------------
    # GRID EXPLODE (Cartesian)
    # -----------------------------
    if GRID_EXPLODE:
        for dataset in DATASETS:
            for (seeds,
                 n_norm, n_anom, noise,
                 bend, scale_x, scale_y,
                 gap, spread,
                 test_size, val_size,
                 label_frac_anom, label_frac_norm, label_normals_subset_only,
                 run_aag, run_dsvdd, run_dsad, run_ocsvm, run_ksvdd,
                 aag_hidden, aag_out, aag_p, aag_nu, aag_Omega, aag_margin,
                 aag_lambda_u, aag_k, aag_gamma_edges, aag_graph_refresh,
                 aag_eta_unl, aag_cap_unlabeled, aag_cap_offset,
                 aag_lr, aag_wd, aag_epochs, aag_batch, aag_warmup,
                 dsvdd_hidden, dsvdd_rep, dsvdd_nu, dsvdd_lr, dsvdd_wd, dsvdd_epochs, dsvdd_batch,
                 dsad_hidden, dsad_rep, dsad_nu, dsad_lr, dsad_wd, dsad_epochs, dsad_batch,
                 ocsvm_nu, ocsvm_gamma,
                 ksvdd_nu, ksvdd_gamma,
                 banana_b, banana_s1, banana_s2, banana_rotate_deg,
                 banana_anom_split, banana_mu_a1, banana_mu_a2, banana_cov_a1, banana_cov_a2
                 ) in product(
                    [SEEDS],                                  # seeds list (passed comma-joined)
                    N_NORM_LIST, N_ANOM_LIST, NOISE_LIST,
                    BEND_LIST, SCALE_X_LIST, SCALE_Y_LIST,
                    GAP_LIST, SPREAD_LIST,
                    TEST_SIZE_LIST, VAL_SIZE_LIST,
                    LABEL_FRAC_ANOM_LIST, LABEL_FRAC_NORM_LIST, LABEL_NORMALS_SUBSET_ONLY,
                    RUN_AAG_LIST, RUN_DEEPSVDD_LIST, RUN_DEEPSAD_LIST, RUN_OCSVM_LIST, RUN_KSVDD_LIST,
                    AAG_HIDDEN_LIST, AAG_OUT_DIM_LIST, AAG_P_LIST, AAG_NU_LIST, AAG_OMEGA_LIST, AAG_MARGIN_LIST,
                    AAG_LAMBDA_U_LIST, AAG_K_LIST, AAG_GAMMA_EDGES_LIST, AAG_GRAPH_REFRESH_LIST,
                    AAG_ETA_UNL_LIST, AAG_CAP_UNLABELED_LIST, AAG_CAP_OFFSET_LIST,
                    AAG_LR_LIST, AAG_WD_LIST, AAG_EPOCHS_LIST, AAG_BATCH_LIST, AAG_WARMUP_LIST,
                    DSVDD_HIDDEN_LIST, DSVDD_REP_DIM_LIST, DSVDD_NU_LIST, DSVDD_LR_LIST, DSVDD_WD_LIST, DSVDD_EPOCHS_LIST, DSVDD_BATCH_LIST,
                    DSAD_HIDDEN_LIST, DSAD_REP_DIM_LIST, DSAD_NU_LIST, DSAD_LR_LIST, DSAD_WD_LIST, DSAD_EPOCHS_LIST, DSAD_BATCH_LIST,
                    OCSVM_NU_LIST, OCSVM_GAMMA_LIST,
                    KSVDD_NU_LIST, KSVDD_GAMMA_LIST,
                    BANANA_B_LIST, BANANA_S1_LIST, BANANA_S2_LIST, BANANA_ROTATE_DEG_LIST,
                    BANANA_ANOM_SPLIT_LIST, BANANA_MU_A1_LIST, BANANA_MU_A2_LIST, BANANA_COV_A1_LIST, BANANA_COV_A2_LIST
                 ):
                cmd = one_command(
                    dataset=dataset, seeds=seeds,
                    n_norm=n_norm, n_anom=n_anom, noise=noise,
                    bend=bend, scale_x=scale_x, scale_y=scale_y,
                    gap=gap, spread=spread,
                    test_size=test_size, val_size=val_size,
                    label_frac_anom=label_frac_anom, label_frac_norm=label_frac_norm,
                    label_normals_subset_only=label_normals_subset_only,
                    run_aag=run_aag, run_dsvdd=run_dsvdd, run_dsad=run_dsad, run_ocsvm=run_ocsvm, run_ksvdd=run_ksvdd,
                    aag_hidden=aag_hidden, aag_out=aag_out, aag_p=aag_p, aag_nu=aag_nu, aag_Omega=aag_Omega, aag_margin=aag_margin,
                    aag_lambda_u=aag_lambda_u, aag_k=aag_k, aag_gamma_edges=aag_gamma_edges, aag_graph_refresh=aag_graph_refresh,
                    aag_eta_unl=aag_eta_unl, aag_cap_unlabeled=aag_cap_unlabeled, aag_cap_offset=aag_cap_offset,
                    aag_lr=aag_lr, aag_wd=aag_wd, aag_epochs=aag_epochs, aag_batch=aag_batch, aag_warmup=aag_warmup,
                    dsvdd_hidden=dsvdd_hidden, dsvdd_rep=dsvdd_rep, dsvdd_nu=dsvdd_nu, dsvdd_lr=dsvdd_lr, dsvdd_wd=dsvdd_wd, dsvdd_epochs=dsvdd_epochs, dsvdd_batch=dsvdd_batch,
                    dsad_hidden=dsad_hidden, dsad_rep=dsad_rep, dsad_nu=dsad_nu, dsad_lr=dsad_lr, dsad_wd=dsad_wd, dsad_epochs=dsad_epochs, dsad_batch=dsad_batch,
                    ocsvm_nu=ocsvm_nu, ocsvm_gamma=ocsvm_gamma,
                    ksvdd_nu=ksvdd_nu, ksvdd_gamma=ksvdd_gamma,
                    banana_b=banana_b, banana_s1=banana_s1, banana_s2=banana_s2, banana_rotate_deg=banana_rotate_deg,
                    banana_anom_split=banana_anom_split, banana_mu_a1=banana_mu_a1, banana_mu_a2=banana_mu_a2,
                    banana_cov_a1=banana_cov_a1, banana_cov_a2=banana_cov_a2
                )
                cmds.append(cmd)

    # Execute (optionally in parallel)
    if PARALLEL and GRID_EXPLODE and EXECUTE and len(cmds) > 1:
        with fut.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            list(ex.map(_run, cmds))
    else:
        for c in cmds:
            _run(c)

if __name__ == "__main__":
    main()
