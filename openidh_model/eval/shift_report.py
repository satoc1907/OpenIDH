"""Experiment A over a set of runs: load the dumped features, decompose, tabulate.

Everything here works on runs/<run>/{features_val,features_test,head_weights}.npz
and the existing predictions.csv / calibration.json — no model, no GPU. The
notebook (05_shift_decomposition) only calls these and shows the results.

Units: LOSO-A / LOSO-B are site shifts (case mix AND acquisition, "1a+1b");
field / vendor stay inside UTSW with the same case mix ("1b" probes); rand0 is
the in-distribution negative control.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import ENCODERS, load_features, load_head_weights
from .shift_decomp import age_grade_strata, functional_basis, kitagawa_split, shift_decomposition

UNITS = {           # run prefix -> (label, kind)
    "foldA": ("LOSO-A", "site"),
    "foldB": ("LOSO-B", "site"),
    "field": ("field 3T", "probe"),
    "vendor": ("vendor Philips", "probe"),
    "rand0": ("random f0", "control"),
}
UNIT_ORDER = [v[0] for v in UNITS.values()]
_EPS = 1e-7


def unit_of(run: str) -> tuple[str, str]:
    prefix = run.rsplit("_s", 1)[0]
    if prefix in UNITS:
        return UNITS[prefix]
    # smoke/probe checkpoints (e.g. smoke_foldA_s0): keep them out of the
    # production labels but let the pipeline run on them for a dry run
    return prefix, ("site" if "fold" in prefix else "other")


def load_run(runs_dir: Path, run: str) -> dict | None:
    d = Path(runs_dir) / run
    need = [d / "features_val.npz", d / "features_test.npz", d / "head_weights.npz"]
    if not all(f.is_file() for f in need):
        return None
    meta = json.loads((d / "features_meta.json").read_text()) if (d / "features_meta.json").is_file() else {}
    return {"run": run, "dir": d, "val": load_features(need[0]), "test": load_features(need[1]),
            "W": load_head_weights(need[2]), "meta": meta}


def _active_pair(rd: dict, enc: str):
    """(feat_val, feat_test) restricted to subjects where this encoder was active."""
    fv = rd["val"][f"feat_{enc}"][rd["val"][f"active_{enc}"]]
    ft = rd["test"][f"feat_{enc}"][rd["test"][f"active_{enc}"]]
    return fv, ft


def decompose_run(rd: dict, n_boot: int = 100, seed: int = 0) -> list[dict]:
    """One row per encoder for this run (analysis 1 raw table)."""
    label, kind = unit_of(rd["run"])
    rows = []
    for enc in ENCODERS:
        fv, ft = _active_pair(rd, enc)
        W, b = rd["W"][enc]
        r = shift_decomposition(fv, ft, W, b, n_boot=n_boot, seed=seed)
        rows.append({"run": rd["run"], "unit": label, "kind": kind,
                     "seed": int(rd["run"].rsplit("_s", 1)[1]), "enc": enc,
                     **{k: v for k, v in r.items() if k != "floor"},
                     "floor_dp": r["floor"]["d_p"]["median"], "floor_ds": r["floor"]["d_s"]["median"],
                     "floor_null": r["floor"]["null"]["median"],
                     "floor_null_p95": r["floor"]["null"]["p95"]})
    return rows


def decompose_all(runs_dir: Path, runs: list[str], n_boot: int = 100, seed: int = 0):
    """DataFrame of every (run, encoder) plus the list of runs that had no dump."""
    rows, missing = [], []
    for run in runs:
        rd = load_run(runs_dir, run)
        if rd is None:
            missing.append(run); continue
        rows += decompose_run(rd, n_boot=n_boot, seed=seed)
    df = pd.DataFrame(rows)
    if len(df):
        seen = list(dict.fromkeys(df["unit"]))
        df["unit"] = pd.Categorical(df["unit"], [u for u in UNIT_ORDER if u in seen] + [u for u in seen if u not in UNIT_ORDER])
        df["enc"] = pd.Categorical(df["enc"], ENCODERS)
    return df, missing


SUMMARY_COLS = ["d_p", "d_s", "null_over_floor", "null_over_floor_matched",
                "d_p_over_floor", "d_s_over_floor", "dz_lin", "dS_lin", "dS_exact", "dS_exact_rel",
                "dz_exact", "se_d"]


def summarize_units(df: pd.DataFrame, cols=SUMMARY_COLS) -> pd.DataFrame:
    """mean ± sd over seeds, per unit x encoder."""
    g = df.groupby(["unit", "enc"], observed=True)[cols]
    m, s = g.mean(), g.std(ddof=1).fillna(0.0)
    n = g.size().rename("n_seeds")
    out = pd.concat([m.add_suffix("_mean"), s.add_suffix("_sd"), n], axis=1)
    return out.reset_index()


def fmt_ms(m: float, s: float, nd: int = 2) -> str:
    return f"{m:+.{nd}f} ± {s:.{nd}f}" if s > 0 else f"{m:+.{nd}f}"


def table_markdown(summ: pd.DataFrame) -> str:
    """Analysis 1 as a Discord-ready table: unit x encoder x {d_p, d_s, null/floor}."""
    out = ["| ユニット | enc | d_p (σ) | d_s (σ) | Δ_null / 雑音床 (raw) | Δ_null / 雑音床 (size-matched) | n_seed |",
           "|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        out.append(f"| {r['unit']} | {r['enc']} | {fmt_ms(r['d_p_mean'], r['d_p_sd'])} | "
                   f"{fmt_ms(r['d_s_mean'], r['d_s_sd'])} | "
                   f"{r['null_over_floor_mean']:.2f} ± {r['null_over_floor_sd']:.2f} | "
                   f"{r['null_over_floor_matched_mean']:.2f} ± {r['null_over_floor_matched_sd']:.2f} | "
                   f"{int(r['n_seeds'])} |")
    return "\n".join(out)


# ── analysis 3: stratified delta (LOSO only) ───────────────────────────────────

def stratified_run(rd: dict, min_n: int = 3) -> list[dict]:
    """Kitagawa split of delta into within-stratum (imaging) and composition
    (case mix) parts, projected onto (u_p, u_s) in val-sigma units. Strata are
    grade binary x age tertile; tertile cuts come from the pooled val+test ages
    of this run so both sides share bins. Labels are never used."""
    label, _ = unit_of(rd["run"])
    ages = np.concatenate([rd["val"]["age"], rd["test"]["age"]])
    cuts = tuple(np.nanquantile(ages, [1 / 3, 2 / 3]))
    sv = age_grade_strata(rd["val"]["age"], rd["val"]["grade"], cuts)
    st = age_grade_strata(rd["test"]["age"], rd["test"]["grade"], cuts)
    rows = []
    for enc in ENCODERS:
        fv, ft = _active_pair(rd, enc)
        av, at = rd["val"][f"active_{enc}"], rd["test"][f"active_{enc}"]
        k = kitagawa_split(fv, ft, sv[av], st[at], min_n=min_n)
        basis = functional_basis(rd["W"][enc][0])
        sp = float((fv @ basis["u_p"]).std(ddof=1)); ss = float((fv @ basis["u_s"]).std(ddof=1))
        row = {"run": rd["run"], "unit": label, "enc": enc, "covered": k["covered"],
               "age_cuts": cuts, "n_strata_used": sum(v["delta_k"] is not None for v in k["strata"].values())}
        for part in ("delta", "within", "composition"):
            v = k[part]
            row[f"{part}_dp"] = float(v @ basis["u_p"]) / sp
            row[f"{part}_ds"] = float(v @ basis["u_s"]) / ss
            resid = v - (v @ basis["u_p"]) * basis["u_p"] - (v @ basis["u_s"]) * basis["u_s"]
            row[f"{part}_null"] = float(np.linalg.norm(resid))
        rows.append(row)
    return rows


def strata_table(rd: dict) -> pd.DataFrame:
    """Case counts per stratum on both sides (to see what analysis 3 can cover)."""
    ages = np.concatenate([rd["val"]["age"], rd["test"]["age"]])
    cuts = tuple(np.nanquantile(ages, [1 / 3, 2 / 3]))
    sv = age_grade_strata(rd["val"]["age"], rd["val"]["grade"], cuts)
    st = age_grade_strata(rd["test"]["age"], rd["test"]["grade"], cuts)
    keys = sorted(set(sv) | set(st))
    return pd.DataFrame({"stratum": keys,
                         "n_val": [int((sv == k).sum()) for k in keys],
                         "n_test": [int((st == k).sum()) for k in keys],
                         "mut_val": [float(rd["val"]["y"][sv == k].mean()) if (sv == k).any() else np.nan for k in keys],
                         "mut_test": [float(rd["test"]["y"][st == k].mean()) if (st == k).any() else np.nan for k in keys]})


# ── analysis 5b: does the per-encoder logit shift add up to the observed offset? ──

def _logit(a, b):
    return np.log(np.clip(a, _EPS, None)) - np.log(np.clip(b, _EPS, None))


def logit_shift_check(rd: dict, dec_rows: list[dict]) -> dict:
    """Sum of the first-order per-encoder logit shifts vs what the fused model
    actually did (mean test logit - mean val logit, from predictions.csv) and vs
    the base-rate offset the calibration analysis asked for (calibration.json).

    The first-order sum ignores softplus curvature, volume scaling and the +1, so
    only its sign and rough size are meaningful — that is all the check asks."""
    out = {"run": rd["run"], "unit": unit_of(rd["run"])[0],
           "dz_lin_sum": float(sum(r["dz_lin"] for r in dec_rows)),
           "dz_exact_sum": float(sum(r["dz_exact"] for r in dec_rows)),
           "dz_lin_by_enc": {r["enc"]: r["dz_lin"] for r in dec_rows}}
    pf, cf = rd["dir"] / "predictions.csv", rd["dir"] / "calibration.json"
    if pf.is_file():
        p = pd.read_csv(pf)
        zv = _logit(p[p.role == "val"]["alpha"], p[p.role == "val"]["beta"])
        zt = _logit(p[p.role == "test"]["alpha"], p[p.role == "test"]["beta"])
        out["fused_logit_shift"] = float(zt.mean() - zv.mean())
    if cf.is_file():
        c = json.loads(cf.read_text())
        out["prior_offset"] = float(c.get("prior_offset", np.nan))
        out["val_base_rate"] = c.get("val_base_rate"); out["test_base_rate"] = c.get("test_base_rate")
    return out


# ── analysis 4 helper: measured per-head evidence change from predictions.csv ──

def evidence_change_from_predictions(rd: dict) -> dict:
    """Mean post-scaling evidence (e1+e0) per head, test / val - 1, from the
    calibration job's predictions.csv — the 'evidence halved' observation."""
    pf = rd["dir"] / "predictions.csv"
    if not pf.is_file():
        return {}
    p = pd.read_csv(pf)
    out = {}
    for h in ENCODERS + ["tabular"]:
        S = p[f"e1_{h}"] + p[f"e0_{h}"]
        sv, st = S[p.role == "val"].mean(), S[p.role == "test"].mean()
        out[h] = {"val": float(sv), "test": float(st), "rel": float(st / sv - 1)}
    return out


# ── judgement (spec §判定基準) ──────────────────────────────────────────────────

def judge_unit(summ: pd.DataFrame, unit: str, sig: float = 2.0) -> dict:
    """Which component carries the shift for one unit, by the spec's rules.

    A component 'counts' when it clears `sig` x its own noise floor on average
    over encoders (d_p, d_s against the analytic s.e.; delta_null against the
    size-matched half-split floor). 'dominant' is the counting component with the
    largest floor ratio; if none counts, the features have not moved."""
    s = summ[summ["unit"] == unit]
    if not len(s):
        return {"unit": unit, "verdict": "no data"}
    dp = float((s["d_p_mean"].abs() / s["se_d_mean"]).mean())
    ds = float((s["d_s_mean"].abs() / s["se_d_mean"]).mean())
    dn = float(s["null_over_floor_matched_mean"].mean())
    ratios = {"d_p": dp, "d_s": ds, "delta_null": dn}
    counting = {k: v for k, v in ratios.items() if v >= sig}
    if not counting:
        verdict = "no component clears 2x its floor on the encoder average — see the per-encoder table before calling the features unmoved"
    else:
        dom = max(counting, key=counting.get)
        sign = ""
        if dom in ("d_p", "d_s"):
            sign = " (negative)" if float(s[f"{dom}_mean"].mean()) < 0 else " (positive)"
        verdict = f"{dom} dominant{sign}"
    return {"unit": unit, "ratios": ratios, "verdict": verdict,
            "d_s_sign_by_enc": {str(r["enc"]): float(np.sign(r["d_s_mean"])) for _, r in s.iterrows()}}


# ── analysis 1 figure ──────────────────────────────────────────────────────────

def heatmap(summ: pd.DataFrame, path: str | Path | None = None, title: str = ""):
    """Rows = encoders, columns = units; three panels: d_p, d_s (diverging, zero
    at the neutral midpoint) and delta_null / size-matched floor (sequential).
    Cell text is mean ± sd over seeds. Returns the matplotlib figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    units = [u for u in UNIT_ORDER if u in set(summ["unit"])] + \
            [u for u in summ["unit"].unique() if u not in UNIT_ORDER]
    encs = ENCODERS
    panels = [("d_p", "d_p — criterion shift\n(val-σ units)", "RdBu_r", True),
              ("d_s", "d_s — evidence-mass shift\n(val-σ units)", "RdBu_r", True),
              ("null_over_floor_matched", "Δ_null / noise floor\n(size-matched)", "Blues", False)]
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * len(units) + 3, 1.0 * len(encs) + 1.8), constrained_layout=True)
    for ax, (key, lab, cmap, div) in zip(axes, panels):
        M = np.full((len(encs), len(units)), np.nan); S = np.zeros_like(M)
        for i, e in enumerate(encs):
            for j, u in enumerate(units):
                r = summ[(summ["enc"] == e) & (summ["unit"] == u)]
                if len(r):
                    M[i, j] = r[f"{key}_mean"].iloc[0]; S[i, j] = r[f"{key}_sd"].iloc[0]
        if div:
            lim = float(np.nanmax(np.abs(M))) or 1.0
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        else:
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=float(np.nanmax(M)) or 1.0)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(units)), units, rotation=20, ha="right")
        ax.set_yticks(range(len(encs)), encs)
        ax.set_title(lab, fontsize=10, loc="left")
        for i in range(len(encs)):
            for j in range(len(units)):
                if np.isnan(M[i, j]):
                    continue
                v = M[i, j]
                # text stays in ink tokens; choose light ink only on the darkest cells
                dark = (abs(norm(v) - 0.5) > 0.35) if div else (norm(v) > 0.6)
                ax.text(j, i, fmt_ms(v, S[i, j]) if div else f"{v:.1f} ± {S[i, j]:.1f}",
                        ha="center", va="center", fontsize=8, color="white" if dark else "#222")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=11)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight")
    return fig


def df_markdown(df: pd.DataFrame, nd: int = 2) -> str:
    """Pipe-table without the tabulate dependency; floats rounded to `nd`."""
    cols = [str(c) if not isinstance(c, tuple) else " ".join(str(x) for x in c if x != "") for c in df.columns]
    fmt = lambda v: f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r.tolist()) + " |")
    return "\n".join(out)
