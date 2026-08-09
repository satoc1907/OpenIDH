"""Render the Stage B result report — one self-contained HTML page, no external assets.

Reads the same runs/*/result.json as aggregate_results.py and writes the figures
(AUC vs ECE, uncertainty dumbbells, evidence balance, epoch budget) as inline SVG.

    uv run python scripts/build_report.py
    uv run python scripts/build_report.py --runs-dir runs --out results/stageb-report.html

The prose is written against the 27-run Stage B result; re-run it after a new
sweep and re-read the narrative — the numbers update, the claims do not.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import HEADS, load_runs  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
_ap.add_argument("--out", default=str(_ROOT / "results" / "stageb-report.html"))
_ARGS = _ap.parse_args()

ROWS = load_runs(Path(_ARGS.runs_dir))
if not ROWS:
    raise SystemExit(f"no production runs found under {_ARGS.runs_dir}/")
OOD = [r for r in ROWS if r["kind"] == "OOD"]
IND = [r for r in ROWS if r["kind"] == "in-dist"]
COMMIT = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

# Optional inputs: the calibration pass may not have run yet, and an older pass
# predates the prior-correction oracle. Each section renders only what exists.
_RD = Path(_ARGS.runs_dir)
CAL = {r["run"]: json.loads((_RD / r["run"] / "calibration.json").read_text())
       for r in ROWS if (_RD / r["run"] / "calibration.json").is_file()}
HAS_CAL = len(CAL) == len(ROWS)
HAS_ORACLE = HAS_CAL and all("test_prior_oracle" in c for c in CAL.values())
_pf = [_RD / r["run"] / "predictions.csv" for r in ROWS]
PRED = (pd.concat([pd.read_csv(p) for p in _pf], ignore_index=True)
        if all(p.is_file() for p in _pf) else None)
KIND = {r["run"]: r["kind"] for r in ROWS}
LABEL = {r["run"]: r["label"] for r in ROWS}

JA = {"LOSO-A": "LOSO-A（施設）", "LOSO-B": "LOSO-B（施設）",
      "vendor Philips": "vendor（Philips）", "field 3T": "field（3T）"}
for i in range(5):
    JA[f"random f{i}"] = f"random fold {i}"

UNITS = []          # (label, kind, rows) — OOD first
for kind in ("OOD", "in-dist"):
    seen = []
    for r in ROWS:
        if r["kind"] == kind and r["label"] not in seen:
            seen.append(r["label"])
    for lab in seen:
        UNITS.append((lab, kind, [r for r in ROWS if r["label"] == lab]))


def ms(rs, k):
    v = np.array([r[k] for r in rs], float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def esc(s): return html.escape(str(s))


# ───────────────────────── figure 1: AUC | ECE small multiples ─────────────────
def fig_discrimination_calibration():
    rowh, top, bot = 30, 46, 34
    H = top + rowh * len(UNITS) + bot
    labw, panw, gap = 172, 268, 44
    W = labw + panw * 2 + gap + 66
    p1x, p2x = labw, labw + panw + gap

    def sx(v, lo, hi, x0): return x0 + (v - lo) / (hi - lo) * (panw - 58)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="ユニット別の AUC と ECE（3 seed の平均±標準偏差）">']
    s.append('<title>ユニット別 AUC / ECE</title>')
    for x0, lo, hi, name, ticks in ((p1x, .80, .95, "AUC（判別）", [.80, .85, .90, .95]),
                                    (p2x, .00, .20, "ECE（較正誤差）", [0, .05, .10, .15, .20])):
        s.append(f'<text class="pnl" x="{x0}" y="16">{name}</text>')
        for t in ticks:
            x = sx(t, lo, hi, x0)
            s.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{H - bot + 4}"/>')
            lab = f"{t:.2f}" if hi <= 1 else f"{t}"
            s.append(f'<text class="tick" x="{x:.1f}" y="{H - bot + 20}" text-anchor="middle">{lab}</text>')

    for i, (lab, kind, rs) in enumerate(UNITS):
        y = top + rowh * i + rowh / 2
        cls = "ood" if kind == "OOD" else "ind"
        if i == 4:
            s.append(f'<line class="sep" x1="6" y1="{y - rowh / 2:.1f}" x2="{W - 60}" y2="{y - rowh / 2:.1f}"/>')
        s.append(f'<text class="rowlab" x="{labw - 14}" y="{y + 4:.1f}" text-anchor="end">{esc(JA[lab])}</text>')
        for x0, lo, hi, key in ((p1x, .80, .95, "auc"), (p2x, .00, .20, "ece")):
            m, sd = ms(rs, key)
            cx, x1, x2 = sx(m, lo, hi, x0), sx(m - sd, lo, hi, x0), sx(m + sd, lo, hi, x0)
            tip = f"{JA[lab]} · {key.upper()} {m:.3f} ± {sd:.3f}（3 seed）"
            s.append(f'<g class="mk" data-tip="{esc(tip)}">')
            s.append(f'<line class="whisk {cls}" x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}"/>')
            s.append(f'<circle class="dot {cls}" cx="{cx:.1f}" cy="{y:.1f}" r="5.5"/>')
            s.append(f'<rect class="hit" x="{x1 - 8:.1f}" y="{y - 12:.1f}" width="{x2 - x1 + 16:.1f}" height="24"/>')
            s.append('</g>')
            s.append(f'<text class="val" x="{x0 + panw - 50}" y="{y + 4:.1f}">{m:.3f}</text>')
    s.append('</svg>')
    return "".join(s)


# ───────────────────────── figure 2: uncertainty dumbbells ─────────────────────
def fig_uncertainty():
    rs = sorted(ROWS, key=lambda r: r["S_correct"] - r["S_wrong"])
    rowh, top, bot, labw = 17, 44, 54, 108
    H = top + rowh * len(rs) + bot
    W, plotw = 760, 760 - 108 - 92
    lo, hi = 0, 70

    def sx(v): return labw + (v - lo) / (hi - lo) * plotw

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="27ランすべてで、誤答時の Dirichlet strength が正答時を下回る">']
    s.append('<title>正答時 vs 誤答時の確信度（S）</title>')
    for t in range(0, 71, 10):
        x = sx(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{H - bot + 4}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H - bot + 20}" text-anchor="middle">{t}</text>')
    s.append(f'<text class="tick" x="{sx(35):.1f}" y="{H - bot + 36}" text-anchor="middle">median S（Dirichlet strength：大きいほど確信が強い）</text>')

    for i, r in enumerate(rs):
        y = top + rowh * i + rowh / 2
        cls = "ood" if r["kind"] == "OOD" else "ind"
        xw, xc = sx(r["S_wrong"]), sx(r["S_correct"])
        tip = (f"{r['run']} · 正答 {r['S_correct']:.1f} / 誤答 {r['S_wrong']:.1f}"
               f"（差 {r['S_wrong'] - r['S_correct']:+.1f}）")
        s.append(f'<text class="runlab" x="{labw - 12}" y="{y + 3.5:.1f}" text-anchor="end">{esc(r["run"])}</text>')
        s.append(f'<g class="mk" data-tip="{esc(tip)}">')
        s.append(f'<line class="conn {cls}" x1="{xw:.1f}" y1="{y:.1f}" x2="{xc:.1f}" y2="{y:.1f}"/>')
        s.append(f'<circle class="ring {cls}" cx="{xw:.1f}" cy="{y:.1f}" r="4.2"/>')
        s.append(f'<circle class="dot {cls}" cx="{xc:.1f}" cy="{y:.1f}" r="4.2"/>')
        s.append(f'<rect class="hit" x="{labw - 8}" y="{y - 8:.1f}" width="{plotw + 16}" height="16"/>')
        s.append('</g>')
    return "".join(s) + '</svg>'


# ───────────────────────── figure 3: evidence balance ─────────────────────────
def fig_evidence():
    allm = {h: ms(ROWS, f"ev_{h}")[0] for h in HEADS}
    ven = {h: ms([r for r in ROWS if r["label"] == "vendor Philips"], f"ev_{h}")[0] for h in HEADS}
    rowh, top, bot, labw = 52, 46, 40, 88
    H = top + rowh * len(HEADS) + bot
    W, plotw = 760, 760 - 88 - 84
    hi = 13.0

    def sw(v): return v / hi * plotw

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="モダリティ別の平均 evidence。T1c が支配的で、vendor では T1c 以外が落ちる">']
    s.append('<title>モダリティ別 evidence</title>')
    for t in (0, 4, 8, 12):
        x = labw + sw(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{H - bot + 4}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H - bot + 20}" text-anchor="middle">{t}</text>')
    for i, h in enumerate(HEADS):
        y = top + rowh * i
        s.append(f'<text class="rowlab" x="{labw - 14}" y="{y + 26:.1f}" text-anchor="end">{esc(h)}</text>')
        for j, (src, cls, nm) in enumerate(((allm, "ind", "全27ラン平均"), (ven, "ood", "vendor のみ"))):
            v = src[h]
            by = y + 6 + j * 17
            s.append(f'<g class="mk" data-tip="{esc(f"{h} · {nm} {v:.2f}")}">')
            s.append(f'<rect class="bar {cls}" x="{labw}" y="{by:.1f}" width="{max(sw(v), 2):.1f}" height="13" rx="4"/>')
            s.append(f'<rect class="hit" x="{labw}" y="{by - 2:.1f}" width="{plotw}" height="17"/>')
            s.append('</g>')
            s.append(f'<text class="val sm" x="{labw + sw(v) + 8:.1f}" y="{by + 11:.1f}">{v:.2f}</text>')
    return "".join(s) + '</svg>'


# ───────────────────────── figure 4: epoch budget ─────────────────────────────
def fig_epochs():
    W, H, top, labw = 760, 214, 56, 108
    plotw = W - labw - 40
    hi = 200

    def sx(v): return labw + v / hi * plotw

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="27ランすべてが40 epoch 以内に早期終了し、200 epoch の上限には到達しなかった">']
    s.append('<title>エポック予算の使用状況</title>')
    s.append(f'<rect class="unused" x="{sx(41):.1f}" y="{top - 18}" width="{sx(200) - sx(41):.1f}" height="104" rx="3"/>')
    s.append(f'<text class="note" x="{(sx(41) + sx(200)) / 2:.1f}" y="{top + 46}" text-anchor="middle">未使用（上限 200 epoch まで）</text>')
    for t in (0, 50, 100, 150, 200):
        x = sx(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{top + 88}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{top + 106}" text-anchor="middle">{t}</text>')
    s.append(f'<text class="tick" x="{sx(100):.1f}" y="{top + 124}" text-anchor="middle">epoch</text>')
    for j, (key, lab) in enumerate((("best_epoch", "ベスト epoch"), ("epochs_run", "停止した epoch"))):
        y = top + j * 44
        s.append(f'<text class="rowlab" x="{labw - 14}" y="{y + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        for k, r in enumerate(sorted(ROWS, key=lambda r: r[key])):
            cls = "ood" if r["kind"] == "OOD" else "ind"
            jitter = ((k % 5) - 2) * 3.0
            cx = sx(r[key])
            s.append(f'<g class="mk" data-tip="{esc(f"{r["run"]} · {lab} {r[key]}")}">')
            s.append(f'<circle class="dot soft {cls}" cx="{cx:.1f}" cy="{y + jitter:.1f}" r="4.6"/>')
            s.append(f'<rect class="hit" x="{cx - 7:.1f}" y="{y + jitter - 7:.1f}" width="14" height="14"/>')
            s.append('</g>')
    return "".join(s) + '</svg>'


# ───────────────────────── figure 5: calibration methods ──────────────────────
def fig_calibration():
    """Per unit: raw ECE, after temperature, after the prior-correction oracle.

    Colour keeps meaning OOD/in-dist as everywhere else; the correction method is
    carried by mark shape, so no third hue is needed.
    """
    rowh, top, bot, labw = 34, 46, 40, 172
    H = top + rowh * len(UNITS) + bot
    W, plotw = 760, 760 - 172 - 84
    hi = 0.20

    def sx(v): return labw + min(v, hi) / hi * plotw

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="ユニット別の ECE：生・temperature 較正後'
         f'{"・prior correction オラクル" if HAS_ORACLE else ""}">']
    s.append('<title>較正手法ごとの ECE</title>')
    for t in (0, .05, .10, .15, .20):
        x = sx(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{H - bot + 4}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H - bot + 20}" text-anchor="middle">{t:.2f}</text>')
    s.append(f'<text class="tick" x="{sx(.10):.1f}" y="{H - bot + 36}" text-anchor="middle">ECE（低いほど良い）</text>')

    for i, (lab, kind, rs) in enumerate(UNITS):
        y = top + rowh * i + rowh / 2
        cls = "ood" if kind == "OOD" else "ind"
        cals = [CAL[r["run"]] for r in rs]
        raw = float(np.mean([c["test_raw"]["ece"] for c in cals]))
        tmp = float(np.mean([c["test_calibrated"]["ece"] for c in cals]))
        ora = (float(np.mean([c["test_prior_oracle"]["ece"] for c in cals]))
               if HAS_ORACLE else None)
        if i == 4:
            s.append(f'<line class="sep" x1="6" y1="{y - rowh / 2:.1f}" x2="{W - 70}" y2="{y - rowh / 2:.1f}"/>')
        s.append(f'<text class="rowlab" x="{labw - 14}" y="{y + 4:.1f}" text-anchor="end">{esc(JA[lab])}</text>')
        pts = [(raw, "生"), (tmp, "temperature")] + ([(ora, "prior oracle")] if ora is not None else [])
        s.append(f'<line class="conn {cls}" x1="{sx(min(p for p, _ in pts)):.1f}" y1="{y:.1f}" '
                 f'x2="{sx(max(p for p, _ in pts)):.1f}" y2="{y:.1f}"/>')
        for v, nm in pts:
            tip = f"{JA[lab]} · {nm} ECE {v:.3f}"
            s.append(f'<g class="mk" data-tip="{esc(tip)}">')
            if nm == "生":
                s.append(f'<circle class="dot {cls}" cx="{sx(v):.1f}" cy="{y:.1f}" r="5.5"/>')
            elif nm == "temperature":
                s.append(f'<circle class="ring {cls}" cx="{sx(v):.1f}" cy="{y:.1f}" r="5"/>')
            else:
                s.append(f'<rect class="diamond {cls}" x="{sx(v) - 4.6:.1f}" y="{y - 4.6:.1f}" '
                         f'width="9.2" height="9.2" transform="rotate(45 {sx(v):.1f} {y:.1f})"/>')
            s.append(f'<rect class="hit" x="{sx(v) - 9:.1f}" y="{y - 12:.1f}" width="18" height="24"/>')
            s.append('</g>')
        best = min(p for p, _ in pts)
        s.append(f'<text class="val" x="{labw + plotw + 12}" y="{y + 4:.1f}">{best:.3f}</text>')
    return "".join(s) + '</svg>'


# ───────────────────────── figure 6: reliability diagram ──────────────────────
def _bins(df, n=10):
    edges = np.linspace(0, 1, n + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["p_hat"] > lo) & (df["p_hat"] <= hi) if lo > 0 else (df["p_hat"] <= hi)
        if m.sum() < 5:            # too few subjects to read anything into
            continue
        out.append((float(df.loc[m, "p_hat"].mean()), float(df.loc[m, "y_true"].mean()), int(m.sum())))
    return out


def fig_reliability():
    test = PRED[PRED["role"] == "test"]
    series = [
        ("分布内（random 5-fold）", "ind", test[test["run"].map(KIND) == "in-dist"]),
        ("分布外（全4ユニット）", "ood", test[test["run"].map(KIND) == "OOD"]),
        ("LOSO-B のみ", "acc", test[test["run"].map(LABEL) == "LOSO-B"]),
    ]
    W, H, pad, top = 520, 470, 54, 30
    plot = W - pad - 26

    def X(v): return pad + v * plot

    def Y(v): return top + (1 - v) * plot

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="信頼度図：予測確率と実際の陽性率の対応">']
    s.append('<title>reliability diagram</title>')
    for t in (0, .25, .5, .75, 1.0):
        s.append(f'<line class="grid" x1="{X(t):.1f}" y1="{top}" x2="{X(t):.1f}" y2="{Y(0):.1f}"/>')
        s.append(f'<line class="grid" x1="{pad}" y1="{Y(t):.1f}" x2="{X(1):.1f}" y2="{Y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{X(t):.1f}" y="{Y(0) + 18:.1f}" text-anchor="middle">{t:.2f}</text>')
        s.append(f'<text class="tick" x="{pad - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    s.append(f'<line class="ideal" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>')
    s.append(f'<text class="note" x="{X(.72):.1f}" y="{Y(.78):.1f}">完全較正</text>')
    s.append(f'<text class="tick" x="{X(.5):.1f}" y="{Y(0) + 38:.1f}" text-anchor="middle">予測確率（ビン平均）</text>')
    s.append(f'<text class="tick" x="14" y="{top + plot / 2:.1f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {top + plot / 2:.1f})">実際の陽性率</text>')

    for nm, cls, df in series:
        b = _bins(df)
        if not b:
            continue
        pts = " ".join(f"{X(p):.1f},{Y(o):.1f}" for p, o, _ in b)
        s.append(f'<polyline class="rel {cls}" points="{pts}"/>')
        for p, o, n in b:
            s.append(f'<g class="mk" data-tip="{esc(f"{nm} · 予測 {p:.2f} → 実際 {o:.2f}（n={n}）")}">')
            s.append(f'<circle class="dot {cls}" cx="{X(p):.1f}" cy="{Y(o):.1f}" r="4.6"/>')
            s.append(f'<rect class="hit" x="{X(p) - 9:.1f}" y="{Y(o) - 9:.1f}" width="18" height="18"/>')
            s.append('</g>')
        p, o, _ = b[-1]
        s.append(f'<text class="dirlab {cls}" x="{X(p) + 9:.1f}" y="{Y(o) + 4:.1f}">{esc(nm)}</text>')
    return "".join(s) + '</svg>'


# ───────────────────────── figure 7: in-site vs out-site S ────────────────────
LOSO_SITES = {"LOSO-A": ("UPenn + UTSW", "UCSF"), "LOSO-B": ("UCSF + UPenn", "UTSW")}


def insite_outsite_rows():
    """Same model, in-site held-out (inner-val) vs out-site (test) — spec §14.2."""
    out = []
    for r in ROWS:
        if r["label"] not in LOSO_SITES:
            continue
        d = PRED[PRED["run"] == r["run"]]
        v, t = d[d["role"] == "val"], d[d["role"] == "test"]
        out.append({"run": r["run"], "label": r["label"],
                    "in_site": LOSO_SITES[r["label"]][0], "out_site": LOSO_SITES[r["label"]][1],
                    "S_in": float(v["S"].median()), "S_out": float(t["S"].median()),
                    "n_in": len(v), "n_out": len(t)})
    return sorted(out, key=lambda x: x["run"])


def fig_insite():
    rs = insite_outsite_rows()
    rowh, top, bot, labw = 30, 44, 52, 108
    H = top + rowh * len(rs) + bot
    W, plotw = 700, 700 - 108 - 90
    hi = max(max(r["S_in"], r["S_out"]) for r in rs) * 1.12

    def sx(v): return labw + v / hi * plotw

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="同一モデルでの in-site と out-site の確信度 S の比較">']
    s.append('<title>in-site vs out-site の S</title>')
    for t in np.linspace(0, hi, 6):
        s.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{top - 12}" x2="{sx(t):.1f}" y2="{H - bot + 4}"/>')
        s.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - bot + 20}" text-anchor="middle">{t:.0f}</text>')
    s.append(f'<text class="tick" x="{sx(hi / 2):.1f}" y="{H - bot + 36}" text-anchor="middle">median S</text>')
    for i, r in enumerate(rs):
        y = top + rowh * i + rowh / 2
        s.append(f'<text class="runlab" x="{labw - 12}" y="{y + 3.5:.1f}" text-anchor="end">{esc(r["run"])}</text>')
        tip = (f"{r['run']} · in-site {r['in_site']} S={r['S_in']:.1f} (n={r['n_in']}) / "
               f"out-site {r['out_site']} S={r['S_out']:.1f} (n={r['n_out']})")
        s.append(f'<g class="mk" data-tip="{esc(tip)}">')
        s.append(f'<line class="conn ood" x1="{sx(r["S_out"]):.1f}" y1="{y:.1f}" x2="{sx(r["S_in"]):.1f}" y2="{y:.1f}"/>')
        s.append(f'<circle class="ring ood" cx="{sx(r["S_in"]):.1f}" cy="{y:.1f}" r="4.6"/>')
        s.append(f'<circle class="dot ood" cx="{sx(r["S_out"]):.1f}" cy="{y:.1f}" r="4.6"/>')
        s.append(f'<rect class="hit" x="{labw - 8}" y="{y - 12:.1f}" width="{plotw + 16}" height="24"/>')
        s.append('</g>')
        s.append(f'<text class="val" x="{labw + plotw + 14}" y="{y + 4:.1f}">'
                 f'{(r["S_out"] - r["S_in"]) / r["S_in"] * 100:+.0f}%</text>')
    return "".join(s) + '</svg>'


# ───────────────────────── tables ─────────────────────────────────────────────
def table_units():
    h = ['<table><caption>ユニット別サマリ（3 seed の平均 ± 標準偏差）</caption><thead><tr>'
         '<th scope="col">ユニット</th><th scope="col">条件</th><th scope="col">n(test)</th>'
         '<th scope="col">AUC</th><th scope="col">AUPRC</th><th scope="col">ECE</th>'
         '<th scope="col">Brier</th></tr></thead><tbody>']
    for lab, kind, rs in UNITS:
        cells = "".join(f'<td class="n">{ms(rs, k)[0]:.3f} <span class="sd">± {ms(rs, k)[1]:.3f}</span></td>'
                        for k in ("auc", "auprc", "ece", "brier"))
        chip = ('<span class="k ood">分布外</span>' if kind == "OOD"
                else '<span class="k ind">分布内</span>')
        h.append(f'<tr><th scope="row">{esc(JA[lab])}</th><td>{chip}</td>'
                 f'<td class="n">{rs[0]["n_test"]}</td>{cells}</tr>')
    return "".join(h) + '</tbody></table>'


def table_runs():
    cols = [("run", "ラン"), ("seed", "seed"), ("n_test", "n(test)"), ("auc", "AUC"),
            ("auprc", "AUPRC"), ("ece", "ECE"), ("brier", "Brier"), ("nll", "NLL"),
            ("S_correct", "S 正答"), ("S_wrong", "S 誤答"), ("best_epoch", "ベスト"),
            ("epochs_run", "停止"), ("reg_over_data", "reg/data")]
    h = ['<table><caption>全27ランの生データ</caption><thead><tr>'
         + "".join(f'<th scope="col">{esc(c[1])}</th>' for c in cols) + '</tr></thead><tbody>']
    for r in ROWS:
        tds = []
        for k, _ in cols:
            v = r[k]
            if k == "run":
                tds.append(f'<th scope="row">{esc(v)}</th>')
            elif isinstance(v, float):
                tds.append(f'<td class="n">{v:.3f}</td>')
            else:
                tds.append(f'<td class="n">{v}</td>')
        h.append("<tr>" + "".join(tds) + "</tr>")
    return "".join(h) + '</tbody></table>'


# ───────────────────────── assemble ───────────────────────────────────────────
auc_o, auc_os = ms(OOD, "auc"); auc_i, auc_is = ms(IND, "auc")
ece_o, ece_os = ms(OOD, "ece"); ece_i, ece_is = ms(IND, "ece")
apr_o, _ = ms(OOD, "auprc"); apr_i, _ = ms(IND, "auprc")
nll_o, _ = ms(OOD, "nll"); nll_i, _ = ms(IND, "nll")
bri_o, _ = ms(OOD, "brier"); bri_i, _ = ms(IND, "brier")
best = np.array([r["best_epoch"] for r in ROWS]); ran = np.array([r["epochs_run"] for r in ROWS])
ok_n = sum(1 for r in ROWS if r["unc_ok"])

CSS = """
:root{
  --plane:#eff2f3; --surface:#fbfcfc; --raised:#f5f7f8;
  --ink:#10161a; --ink2:#4d5b63; --muted:#7b8a92;
  --rule:#dbe1e4; --hair:#e6eaec; --grid:#e2e7e9;
  --accent:#1c5cab; --ood:#eb6834; --ind:#2a78d6; --ser3:#1baf7a;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --unused:rgba(16,22,26,.045);
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
  --plane:#0c0f11; --surface:#15191c; --raised:#1b2024;
  --ink:#f2f5f6; --ink2:#b3c0c7; --muted:#8b979e;
  --rule:#283036; --hair:#222a2f; --grid:#232b30;
  --accent:#7fb0ec; --ood:#d95926; --ind:#3987e5; --ser3:#199e70;
  --good:#0ca30c; --warn:#fab219; --crit:#e06a6a;
  --unused:rgba(255,255,255,.05);
  color-scheme:dark;
}}
:root[data-theme=dark]{
  --plane:#0c0f11; --surface:#15191c; --raised:#1b2024;
  --ink:#f2f5f6; --ink2:#b3c0c7; --muted:#8b979e;
  --rule:#283036; --hair:#222a2f; --grid:#232b30;
  --accent:#7fb0ec; --ood:#d95926; --ind:#3987e5; --ser3:#199e70;
  --good:#0ca30c; --warn:#fab219; --crit:#e06a6a;
  --unused:rgba(255,255,255,.05);
  color-scheme:dark;
}
:root[data-theme=light]{
  --plane:#eff2f3; --surface:#fbfcfc; --raised:#f5f7f8;
  --ink:#10161a; --ink2:#4d5b63; --muted:#7b8a92;
  --rule:#dbe1e4; --hair:#e6eaec; --grid:#e2e7e9;
  --accent:#1c5cab; --ood:#eb6834; --ind:#2a78d6; --ser3:#1baf7a;
  --unused:rgba(16,22,26,.045);
  color-scheme:light;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI","Noto Sans JP",sans-serif;
  font-size:16px; line-height:1.75; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:920px; margin:0 auto; padding:clamp(28px,5vw,68px) clamp(18px,4vw,40px) 96px;
      display:flex; flex-direction:column; gap:clamp(40px,5vw,64px)}
.serif{font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua","Hiragino Mincho ProN","Yu Mincho","Noto Serif JP",Georgia,serif}
.mono,.eyebrow,.tick,.val,.runlab,.n,.meta dt,.meta dd,.chip,.k{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}

.eyebrow{font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0}
h1{font-family:inherit; font-size:clamp(30px,4.6vw,46px); line-height:1.22; letter-spacing:-.015em;
   margin:14px 0 0; text-wrap:balance; font-weight:600}
.dek{font-size:17px; color:var(--ink2); max-width:60ch; margin:18px 0 0}
.meta{display:flex; flex-wrap:wrap; gap:6px 26px; margin:26px 0 0; padding-top:20px; border-top:1px solid var(--rule)}
.meta div{display:flex; gap:9px; align-items:baseline}
.meta dt{font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:0}
.meta dd{margin:0; font-size:12.5px; color:var(--ink2); font-variant-numeric:tabular-nums}

.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(216px,1fr)); gap:14px}
.tile{background:var(--surface); border:1px solid var(--hair); border-radius:10px; padding:20px 20px 18px;
      display:flex; flex-direction:column; gap:6px}
.tile .lab{font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
           font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.tile .big{font-size:38px; line-height:1.05; letter-spacing:-.02em; font-weight:600}
.tile .sub{font-size:13px; color:var(--ink2); line-height:1.55}
.chip{display:inline-flex; align-items:center; gap:5px; align-self:flex-start; margin-top:4px;
      font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; padding:3px 9px; border-radius:999px;
      border:1px solid currentColor}
.chip.pass{color:var(--good)} .chip.caution{color:var(--warn)} .chip.watch{color:var(--crit)}

section.finding{display:flex; flex-direction:column; gap:18px}
section.finding h2{font-size:clamp(21px,2.6vw,27px); line-height:1.35; margin:6px 0 0; font-weight:600;
                   letter-spacing:-.01em; text-wrap:balance}
section.finding p{margin:0; max-width:66ch; color:var(--ink2)}
section.finding p strong{color:var(--ink); font-weight:600}
.figure{background:var(--surface); border:1px solid var(--hair); border-radius:10px;
        padding:22px 20px 16px; overflow-x:auto}
.figure figcaption{font-size:12.5px; color:var(--muted); margin-top:14px; padding-top:12px;
                   border-top:1px solid var(--hair); line-height:1.6}
.legend{display:flex; flex-wrap:wrap; gap:8px 20px; margin-bottom:16px; font-size:12.5px; color:var(--ink2)}
.legend span{display:inline-flex; align-items:center; gap:8px}
.legend i{width:11px; height:11px; border-radius:50%; display:inline-block}
.legend i.hollow{background:transparent; border:2px solid currentColor}

svg{display:block; min-width:640px}
.grid{stroke:var(--grid); stroke-width:1}
.sep{stroke:var(--rule); stroke-width:1; stroke-dasharray:3 3}
.tick{font-size:11px; fill:var(--muted); font-variant-numeric:tabular-nums}
.pnl{font-size:12.5px; fill:var(--ink2); letter-spacing:.02em}
.rowlab{font-size:12.5px; fill:var(--ink); font-family:inherit}
.runlab{font-size:10.5px; fill:var(--muted)}
.val{font-size:11.5px; fill:var(--ink2); font-variant-numeric:tabular-nums}
.val.sm{font-size:10.5px}
.note{font-size:11.5px; fill:var(--muted)}
.dot.ood{fill:var(--ood)} .dot.ind{fill:var(--ind)}
.dot{stroke:var(--surface); stroke-width:2}
.dot.soft{opacity:.85}
.ring.ood{stroke:var(--ood)} .ring.ind{stroke:var(--ind)}
.ring{fill:var(--surface); stroke-width:2}
.whisk.ood,.conn.ood{stroke:var(--ood)} .whisk.ind,.conn.ind{stroke:var(--ind)}
.whisk{stroke-width:2; opacity:.5} .conn{stroke-width:2; opacity:.42}
.bar.ood{fill:var(--ood)} .bar.ind{fill:var(--ind)}
.dot.acc{fill:var(--ser3)} .ring.acc{stroke:var(--ser3)}
.diamond{stroke:var(--surface); stroke-width:2}
.diamond.ood{fill:var(--ood)} .diamond.ind{fill:var(--ind)}
.rel{fill:none; stroke-width:2; opacity:.85}
.rel.ood{stroke:var(--ood)} .rel.ind{stroke:var(--ind)} .rel.acc{stroke:var(--ser3)}
.ideal{stroke:var(--muted); stroke-width:1.5; stroke-dasharray:5 4}
.dirlab{font-size:11.5px; font-family:inherit}
.dirlab.ood{fill:var(--ood)} .dirlab.ind{fill:var(--ind)} .dirlab.acc{fill:var(--ser3)}
.unused{fill:var(--unused)}
.hit{fill:transparent}
.mk{cursor:default}
.mk:hover .dot,.mk:hover .bar{filter:brightness(1.12)}

.tablewrap{overflow-x:auto; background:var(--surface); border:1px solid var(--hair); border-radius:10px}
table{border-collapse:collapse; width:100%; font-size:13px; min-width:560px}
caption{text-align:left; padding:18px 20px 12px; font-size:12.5px; color:var(--muted);
        font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; letter-spacing:.04em}
th,td{padding:8px 14px; text-align:left; border-bottom:1px solid var(--hair); white-space:nowrap}
thead th{font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:500;
         font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; position:sticky; top:0;
         background:var(--surface)}
tbody th{font-weight:500}
td.n{text-align:right; font-variant-numeric:tabular-nums;
     font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sd{color:var(--muted); font-size:11.5px}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:none}
.k{font-size:10.5px; padding:2px 8px; border-radius:999px; letter-spacing:.06em}
.k.ood{color:var(--ood); border:1px solid currentColor}
.k.ind{color:var(--ind); border:1px solid currentColor}

details{background:var(--surface); border:1px solid var(--hair); border-radius:10px}
summary{cursor:pointer; padding:16px 20px; font-size:13.5px; color:var(--ink2)}
summary:focus-visible{outline:2px solid var(--accent); outline-offset:-2px; border-radius:10px}
details .tablewrap{border:none; border-top:1px solid var(--hair); border-radius:0; max-height:460px; overflow:auto}

footer{border-top:1px solid var(--rule); padding-top:22px; font-size:12.5px; color:var(--muted);
       display:flex; flex-direction:column; gap:6px}
footer code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--ink2)}

#tip{position:fixed; z-index:9; pointer-events:none; opacity:0; transform:translateY(3px);
     transition:opacity .1s ease, transform .1s ease; background:var(--ink); color:var(--plane);
     font-size:12px; padding:6px 10px; border-radius:6px; max-width:300px; line-height:1.5;
     font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
#tip.on{opacity:1; transform:translateY(0)}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.textContent=t;tip.classList.add('on');move(e);}
  function move(e){
    var x=e.clientX+14,y=e.clientY+16,r=tip.getBoundingClientRect();
    if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;
    if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  document.addEventListener('pointermove',function(e){
    var m=e.target.closest?e.target.closest('.mk'):null;
    if(m&&m.dataset.tip){show(e,m.dataset.tip);}else{tip.classList.remove('on');}
  });
  document.addEventListener('pointerleave',function(){tip.classList.remove('on');});
})();
"""

LEGEND = ('<div class="legend">'
          '<span style="color:var(--ood)"><i style="background:var(--ood)"></i>分布外シフト（LOSO / vendor / field）</span>'
          '<span style="color:var(--ind)"><i style="background:var(--ind)"></i>分布内（random 5-fold）</span>'
          '</div>')

def _cal_mean(rs, sect, k):
    return float(np.mean([CAL[r["run"]][sect][k] for r in rs]))


CAL_SECTION = ""
if HAS_CAL:
    T_all = np.array([CAL[r["run"]]["temperature"] for r in ROWS])
    ece_raw_o, ece_tmp_o = _cal_mean(OOD, "test_raw", "ece"), _cal_mean(OOD, "test_calibrated", "ece")
    ece_raw_i, ece_tmp_i = _cal_mean(IND, "test_raw", "ece"), _cal_mean(IND, "test_calibrated", "ece")
    worse = sum(1 for r in ROWS
                if CAL[r["run"]]["test_calibrated"]["ece"] > CAL[r["run"]]["test_raw"]["ece"])
    lb = [r for r in ROWS if r["label"] == "LOSO-B"]
    lb_raw, lb_tmp = _cal_mean(lb, "test_raw", "ece"), _cal_mean(lb, "test_calibrated", "ece")
    off_lb = float(np.mean([CAL[r["run"]]["prior_offset"] for r in lb])) if HAS_ORACLE else None

    oracle_p = ""
    if HAS_ORACLE:
        eo_o, eo_i = _cal_mean(OOD, "test_prior_oracle", "ece"), _cal_mean(IND, "test_prior_oracle", "ece")
        lb_ora = _cal_mean(lb, "test_prior_oracle", "ece")
        share = (lb_raw - lb_ora) / lb_raw * 100
        oracle_p = f"""
  <p>切片が本当に足りていないのかを確かめるため、<strong>test の陽性率を既知と仮定した
  prior correction をオラクルとして</strong>当てた。LOSO-B なら inner-val 11.8% → test 28.4%、
  ロジットに {off_lb:+.2f} を足す操作にあたる。実運用では test の陽性率は分からないので配備できないが、
  <strong>base rate 補正で取り戻せる上限</strong>を測れる。結果は LOSO-B の ECE が
  <strong>{lb_raw:.3f} → {lb_ora:.3f}（{share:.0f}% 減）</strong>、分布外全体では {ece_raw_o:.3f} → {eo_o:.3f}、
  分布内では {ece_raw_i:.3f} → {eo_i:.3f}。
  <strong>つまり分布外の較正崩れのうち base rate 由来はおよそ {share:.0f}% で、残りは確率の形そのものの歪みである。</strong>
  切片を持つ較正（Platt scaling）に広げても、埋められるのはこの {share:.0f}% ぶんが上限で、
  しかも実運用では test の陽性率を知り得ない以上そこにも届かない。</p>"""

    rel_fig = ""
    if PRED is not None:
        rel_fig = f"""
  <figure class="figure">
    <div class="legend">
      <span style="color:var(--ind)"><i style="background:var(--ind)"></i>分布内</span>
      <span style="color:var(--ood)"><i style="background:var(--ood)"></i>分布外（全4ユニット）</span>
      <span style="color:var(--ser3)"><i style="background:var(--ser3)"></i>LOSO-B のみ</span>
    </div>
    {fig_reliability()}
    <figcaption>予測確率を10ビンに分け、各ビンの平均予測確率（横）と実際の陽性率（縦）を取った。
    対角線が完全較正。被験者5例未満のビンは描いていない。ECE は1つの数字に潰れてしまうが、
    この図はどの確率帯でずれているかを示す。</figcaption>
  </figure>"""

    site_fig = ""
    if PRED is not None:
        rs = insite_outsite_rows()
        d = float(np.mean([(x["S_out"] - x["S_in"]) / x["S_in"] * 100 for x in rs]))
        site_fig = f"""
  <p class="eyebrow" style="margin-top:8px">設計検証 §14.2 — 同一モデル内での in-site / out-site 比較</p>
  <p>ラン間で S を比べると訓練セットの大きさが交絡するので、<strong>同一モデルに
  in-site データと out-site データを通して比較した</strong>。LOSO なら inner-val が
  in-site の held-out（LOSO-B なら UCSF + UPenn）、test が out-site（同 UTSW）にあたる。
  結果は out-site 側で median S が平均 <strong>{d:+.0f}%</strong>。
  {'施設が変わると確信度が下がるという設計意図どおりの挙動である。' if d < 0 else
   'つまり施設が変わっても確信度が下がっていない。設計意図と逆であり、要検討である。'}</p>
  <figure class="figure">
    <div class="legend">
      <span style="color:var(--ood)"><i class="hollow"></i>in-site（inner-val、訓練施設の held-out）</span>
      <span style="color:var(--ood)"><i style="background:var(--ood)"></i>out-site（test、未知施設）</span>
    </div>
    {fig_insite()}
    <figcaption>LOSO の6ラン。右端は out-site の median S が in-site から何%動いたか。</figcaption>
  </figure>"""

    CAL_SECTION = f"""
<section class="finding">
  <p class="eyebrow">所見 — 事後較正</p>
  <h2 class="serif">temperature scaling は効かなかった。実装ではなく、道具が問題に合っていない</h2>
  <p>仕様 §10 は inner-val で temperature をフィットして test に当てることを指定していた。
  27ラン全てに適用した結果、<strong>分布外の ECE は {ece_raw_o:.3f} → {ece_tmp_o:.3f} とほぼ動かず、
  分布内はむしろ {ece_raw_i:.3f} → {ece_tmp_i:.3f} と悪化した</strong>。
  27ラン中 {worse} ランで ECE が悪化している。最も較正の悪い LOSO-B も {lb_raw:.3f} → {lb_tmp:.3f} で、実質ゼロだった。</p>
  <p>原因は <strong>T の平均が {T_all.mean():.3f}（範囲 {T_all.min():.2f}〜{T_all.max():.2f}）</strong>
  だったことに尽きる。inner-val 上ではモデルは既にほぼ較正されており、temperature は
  「補正すべきものが無い」と判断している。これは実装の失敗ではなく、
  <strong>道具が問題に構造的に噛み合っていない</strong>ということである。</p>
  <p>理由は2つある。第一に、§10 が問題視したのは <strong>base rate のシフト</strong>
  （LOSO-B: inner-val 11.8% → test 28.4%）だが、<strong>temperature はロジットを定数倍するだけで切片を持たない</strong>。
  base rate のずれは切片で吸収すべき量なので、原理的に補正できない。第二に、
  <strong>T をフィットする inner-val は train と同分布である</strong>。
  シフトした後に生じる較正ずれは、inner-val からは観測できない。</p>{oracle_p}
  <p>したがってこれは「較正手法の実装に失敗した」ではなく、
  <strong>「事後較正では分布シフト下の較正ずれを吸収できない」という知見</strong>として読むべきである。
  確率値そのものを信頼できない領域が残る以上、<strong>「この予測は信頼できるか」を確率とは別の量で持つ必要がある</strong>。
  27ラン全てで成立していた evidential な不確実性（誤答時に S が下がる）は、
  まさにその役割を担う量であり、この結果はその必要性を補強している。</p>
  <figure class="figure">
    <div class="legend">
      <span style="color:var(--ink2)"><i style="background:currentColor"></i>生の確率（塗り）</span>
      <span style="color:var(--ink2)"><i class="hollow"></i>temperature 較正後（白抜き）</span>
      {'<span style="color:var(--ink2)"><i style="background:currentColor;border-radius:2px;transform:rotate(45deg)"></i>prior correction オラクル（菱形）</span>' if HAS_ORACLE else ''}
    </div>
    {LEGEND}
    {fig_calibration()}
    <figcaption>3 seed の平均 ECE。右端の数値は3手法のうち最も良い値。色は分布外／分布内、
    形が較正手法を示す。</figcaption>
  </figure>{rel_fig}{site_fig}
</section>
"""

doc = f"""<title>OpenIDH Stage B — 27ラン結果解析</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <p class="eyebrow">OpenIDH · Stage B · 全27ラン</p>
  <h1 class="serif">判別性能は分布シフトに耐えた。<br>較正は耐えなかった。</h1>
  <p class="dek">グリオーマ IDH 変異予測モデルを、施設・ベンダー・磁場強度という
  性質の異なる3種類の分布シフトで評価した結果。AUC の低下は {auc_i - auc_o:.3f} にとどまる一方、
  期待較正誤差は {ece_o / ece_i:.1f} 倍に悪化した。</p>
  <dl class="meta">
    <div><dt>構成</dt><dd>9 fold-unit × 3 seed = 27 ラン</dd></div>
    <div><dt>実行環境</dt><dd>A100-SXM4-40GB / 1 GPU per job</dd></div>
    <div><dt>不確実性チェック</dt><dd>{ok_n} / {len(ROWS)} 合格</dd></div>
    <div><dt>commit</dt><dd>{esc(COMMIT)}</dd></div>
  </dl>
</header>

<section class="tiles">
  <div class="tile">
    <span class="lab">AUC の低下幅</span>
    <span class="big serif">−{auc_i - auc_o:.3f}</span>
    <span class="sub">分布外 {auc_o:.3f} ± {auc_os:.3f} ／ 分布内 {auc_i:.3f} ± {auc_is:.3f}。
    AUPRC の差はさらに小さく {apr_i - apr_o:+.3f}。</span>
    <span class="chip pass">✓ 主張は成立</span>
  </div>
  <div class="tile">
    <span class="lab">ECE の悪化倍率</span>
    <span class="big serif">{ece_o / ece_i:.1f}×</span>
    <span class="sub">分布外 {ece_o:.3f} ／ 分布内 {ece_i:.3f}。
    Brier {bri_i:.3f} → {bri_o:.3f}、NLL {nll_i:.3f} → {nll_o:.3f} も同傾向。</span>
    <span class="chip caution">! 要対処</span>
  </div>
  <div class="tile">
    <span class="lab">不確実性の健全性</span>
    <span class="big serif">{ok_n}/{len(ROWS)}</span>
    <span class="sub">全ランで「誤答時のほうが確信度が低い」が成立。
    evidential 設計の中核が意図どおり機能している。</span>
    <span class="chip pass">✓ 例外なし</span>
  </div>
</section>

<section class="finding">
  <p class="eyebrow">所見 — 判別 vs 較正</p>
  <h2 class="serif">性質の違う3種類のシフトで、AUC が揃って 0.85〜0.87 に着地した</h2>
  <p>施設シフト（LOSO-A / LOSO-B）、ベンダーシフト（Philips）、磁場強度シフト（3T）は、
  それぞれ異なる要因でデータ分布を動かす。にもかかわらず AUC は
  <strong>0.854 / 0.856 / 0.861 / 0.867 と 0.013 の幅に収まった</strong>。
  偶然では出にくい一貫性で、判別性能の頑健性を裏づける。</p>
  <p>ところが同じ図の右パネルを見ると、ECE は分布内の 4 ユニットが 0.035〜0.047 に密集するのに対し、
  分布外は 0.082〜0.162 に散る。<strong>順位づけの能力は保たれるが、確信度の目盛りは狂う</strong>——
  これが今回の実験の中心的な知見だと考える。とりわけ LOSO-B の ECE 0.162 は、
  臨床判断に確率値をそのまま使うことを難しくする水準である。</p>
  <figure class="figure">
    {LEGEND}
    {fig_discrimination_calibration()}
    <figcaption>点は 3 seed の平均、横線は標準偏差。左パネルは高いほど良く、右パネルは低いほど良い。
    破線より上が分布外の 4 ユニット。</figcaption>
  </figure>
</section>

{CAL_SECTION}
<section class="finding">
  <p class="eyebrow">所見 — 不確実性</p>
  <h2 class="serif">27ラン全てで、モデルは間違えるときに自信を落としていた</h2>
  <p>evidential モデルの価値は「当たる」ことではなく「外すときに黙る」ことにある。
  その検証として、正答した症例と誤答した症例で Dirichlet strength の中央値を比較した。
  <strong>27ラン全てで誤答時のほうが低く、例外はなかった</strong>（差は −3.1 〜 −19.6）。</p>
  <p>さらに示唆的なのは vendor（Philips）で、S の絶対値そのものが 20〜37 と他ユニット（40〜60）より明確に低い。
  <strong>最も分布の遠いシフトに対して、モデルが自発的に自信を下げている</strong>。
  較正が崩れる領域を、不確実性が部分的に補償できている可能性を示す。</p>
  <figure class="figure">
    <div class="legend">
      <span style="color:var(--ink2)"><i style="background:currentColor"></i>正答した症例の median S（塗り）</span>
      <span style="color:var(--ink2)"><i class="hollow"></i>誤答した症例の median S（白抜き）</span>
    </div>
    {LEGEND}
    {fig_uncertainty()}
    <figcaption>1行が1ラン、差の小さい順に並べた。線が必ず左向き（白抜きが左）であることが、
    「誤答時に確信度が下がる」の成立を意味する。色は分布外／分布内の別を示す。</figcaption>
  </figure>
</section>

<section class="finding">
  <p class="eyebrow">所見 — モダリティ寄与</p>
  <h2 class="serif">T1c への依存が強く、年齢・性別はほとんど使われていない</h2>
  <p>ヘッドごとの平均 evidence を見ると <strong>T1c が 11.56 と突出</strong>している。
  造影 T1 が IDH 判別に効くこと自体は臨床的に妥当なので、これは説明のつく挙動である。
  一方 <strong>tabular（年齢・性別）は 1.26 とほぼ無視されている</strong>。
  年齢は IDH 変異の強い予測因子として知られており、意図した設計でないなら検討の余地がある。</p>
  <p>vendor（Philips）だけを取り出すと、T1c が 8.24 を保つ一方で T1 / T2 / FLAIR が 3.5 前後まで落ちる。
  <strong>ベンダーが変わると T1c 以外のシーケンスから証拠を引き出せなくなっている</strong>。
  前処理やコントラスト正規化がベンダー間で揃っていない可能性を示唆する。</p>
  <figure class="figure">
    <div class="legend">
      <span style="color:var(--ind)"><i style="background:var(--ind)"></i>全27ラン平均</span>
      <span style="color:var(--ood)"><i style="background:var(--ood)"></i>vendor（Philips）のみ</span>
    </div>
    {fig_evidence()}
    <figcaption>スケール済み evidence の平均値。値が大きいヘッドほど最終判断への寄与が大きい。</figcaption>
  </figure>
</section>

<section class="finding">
  <p class="eyebrow">所見 — 学習の実態</p>
  <h2 class="serif">200 epoch を用意したが、実際に使われたのは平均 8 epoch だった</h2>
  <p>全27ランが <strong>{ran.min()}〜{ran.max()} epoch で早期終了</strong>し、上限の 200 epoch に到達したランは一つもない。
  ベスト epoch は最小 {best.min()} / 最大 {best.max()} / 平均 {best.mean():.1f}。
  907〜1046 症例に対して ViT-small×2 trunk（44M パラメータ）という構成なので、
  過学習が速いこと自体は道理である。</p>
  <p>ただし <strong>実質 10 epoch 未満しか学習に使えていない</strong>という事実は、
  学習率・正則化・データ拡張を見直せばまだ伸びしろがあることを示す。
  また計算資源の観点では、24時間のウォールタイム制限に対して実際の所要は 1 ラン約1時間で、
  制限が問題になる場面ではなかった。</p>
  <figure class="figure">
    {LEGEND}
    {fig_epochs()}
    <figcaption>点が1ラン（見やすさのため上下に散らしている）。網掛けは一度も使われなかった epoch 範囲。</figcaption>
  </figure>
</section>

<section class="finding">
  <p class="eyebrow">データ</p>
  <h2 class="serif">数表</h2>
  <div class="tablewrap">{table_units()}</div>
  <details>
    <summary>全27ランの生データを開く</summary>
    <div class="tablewrap">{table_runs()}</div>
  </details>
</section>

<footer>
  <div>集計元：<code>runs/*/result.json</code>（27ラン）／ 再現：<code>uv run python scripts/aggregate_results.py</code></div>
  <div>各 fold の test セットは仕様どおり1回だけ評価している。± は 3 seed の標本標準偏差であり、症例レベルの信頼区間ではない。</div>
  <div>ROC 曲線・キャリブレーションプロットには症例ごとの予測値が必要で、<code>result.json</code> には含まれていない。作成するには推論ジョブの追加実行が要る。</div>
</footer>

</div>
<div id="tip" aria-hidden="true"></div>
<script>{JS}</script>
"""

out = Path(_ARGS.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc, encoding="utf-8")
print(f"wrote {out}  ({len(doc):,} bytes)")
print(f"runs={len(ROWS)} ood={len(OOD)} ind={len(IND)} unc_ok={ok_n}")
print(f"AUC ood={auc_o:.3f}+-{auc_os:.3f} ind={auc_i:.3f}+-{auc_is:.3f} gap={auc_i-auc_o:.3f}")
print(f"ECE ood={ece_o:.3f} ind={ece_i:.3f} ratio={ece_o/ece_i:.2f}")
print(f"epochs {ran.min()}-{ran.max()}, best {best.min()}-{best.max()} mean {best.mean():.1f}")
