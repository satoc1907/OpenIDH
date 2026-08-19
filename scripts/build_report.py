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
        ("分布内", "ind", test[test["run"].map(KIND) == "in-dist"]),
        ("分布外", "ood", test[test["run"].map(KIND) == "OOD"]),
        ("LOSO-B", "acc", test[test["run"].map(LABEL) == "LOSO-B"]),
    ]
    W, H, pad, top = 520, 536, 54, 30
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
        s.append(f'<text class="dirlab {cls}" x="{X(p) - 9:.1f}" y="{Y(o) - 9:.1f}" '
                 f'text-anchor="end">{esc(nm)}</text>')
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


# ───────────────────────── figure 8: sample-size curve ────────────────────────
def fig_sample_size(rows, floor, em=None, target=0.08):
    W, H, pad, top, bot = 720, 350, 62, 34, 62
    plotw, ploth = W - pad - 150, H - top - bot
    hi = max([r["ece_hi"] for r in rows] + [rows[0]["ece_raw"]] + ([em] if em else [])) * 1.06
    xs = [np.log10(r["n"]) for r in rows]
    lo_x, hi_x = min(xs), max(xs)

    def X(n): return pad + (np.log10(n) - lo_x) / (hi_x - lo_x) * plotw

    def Y(v): return top + (1 - min(v, hi) / hi) * ploth

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
         f'aria-label="target ドメインのラベル付き症例数と較正誤差の関係">']
    s.append('<title>必要な症例数と ECE</title>')
    for t in np.linspace(0, hi, 5):
        s.append(f'<line class="grid" x1="{pad}" y1="{Y(t):.1f}" x2="{pad + plotw:.1f}" y2="{Y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{pad - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for r in rows:
        s.append(f'<text class="tick" x="{X(r["n"]):.1f}" y="{Y(0) + 20:.1f}" text-anchor="middle">{r["n"]}</text>')
    s.append(f'<text class="tick" x="{pad + plotw / 2:.1f}" y="{Y(0) + 40:.1f}" text-anchor="middle">'
             f'target ドメインのラベル付き症例数 N（対数目盛）</text>')
    s.append(f'<text class="tick" x="16" y="{top + ploth / 2:.1f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {top + ploth / 2:.1f})">ECE</text>')

    refs = [(rows[0]["ece_raw"], "補正なし", "ood"), (target, f"目標 {target:.2f}", "muted"),
            (floor, "分布内の水準", "ind")]
    if em is not None:
        refs.insert(1, (em, "EM（ラベル0例）", "acc"))
    for v, nm, cls in refs:
        s.append(f'<line class="refline {cls}" x1="{pad}" y1="{Y(v):.1f}" x2="{pad + plotw:.1f}" y2="{Y(v):.1f}"/>')
        s.append(f'<text class="dirlab {cls}" x="{pad + plotw + 10:.1f}" y="{Y(v) + 4:.1f}">{esc(nm)} {v:.3f}</text>')

    band = ([f"{X(r['n']):.1f},{Y(r['ece_hi']):.1f}" for r in rows]
            + [f"{X(r['n']):.1f},{Y(r['ece_lo']):.1f}" for r in reversed(rows)])
    s.append(f'<polygon class="band ood" points="{" ".join(band)}"/>')
    s.append(f'<polyline class="rel ood" points="{" ".join(f"{X(r['n']):.1f},{Y(r['ece_mean']):.1f}" for r in rows)}"/>')
    for r in rows:
        tip = (f"N={r['n']} · ECE {r['ece_mean']:.3f}（95%CI {r['ece_lo']:.3f}–{r['ece_hi']:.3f}、"
               f"{r['reps']}回）· 推定オフセット {r['offset_mean']:+.2f} ± {r['offset_sd']:.2f}")
        s.append(f'<g class="mk" data-tip="{esc(tip)}">')
        s.append(f'<circle class="dot ood" cx="{X(r["n"]):.1f}" cy="{Y(r["ece_mean"]):.1f}" r="5"/>')
        s.append(f'<rect class="hit" x="{X(r["n"]) - 12:.1f}" y="{top}" width="24" height="{ploth:.0f}"/>')
        s.append('</g>')
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
.refline{stroke-width:1.5; stroke-dasharray:5 4; opacity:.75}
.refline.ood{stroke:var(--ood)} .refline.ind{stroke:var(--ind)} .refline.muted{stroke:var(--muted)}
.refline.acc{stroke:var(--ser3); stroke-dasharray:none; stroke-width:2.5}
.dirlab.muted{fill:var(--muted)}
.lead{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px;
      letter-spacing:.1em; color:var(--muted); margin-right:12px; white-space:nowrap}
.callout{border-left:3px solid var(--accent); background:var(--raised); border-radius:0 8px 8px 0;
         padding:16px 20px; margin:0; font-size:14.5px; line-height:1.7; color:var(--ink2)}
.callout strong{color:var(--ink)}
.rules{display:flex; flex-direction:column; gap:10px; margin:0; padding:0; list-style:none}
.rules li{display:flex; gap:14px; align-items:baseline; background:var(--surface);
          border:1px solid var(--hair); border-radius:8px; padding:13px 18px; font-size:14.5px}
.rules b{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px;
         white-space:nowrap; color:var(--ink)}
.band{opacity:.15; stroke:none}
.band.ood{fill:var(--ood)}
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
        cut_lb = (lb_raw - lb_ora) / lb_raw * 100
        share_lb = (lb_raw - lb_ora) / (lb_raw - ece_raw_i) * 100
        share_o = (ece_raw_o - eo_o) / (ece_raw_o - ece_raw_i) * 100
        oracle_p = f"""
  <p>切片が本当に足りていないのかを確かめるため、<strong>test の陽性率を既知と仮定した
  prior correction をオラクルとして</strong>当てた。LOSO-B なら inner-val 11.8% → test 28.4%、
  ロジットに {off_lb:+.2f} を足す操作にあたる。test の陽性率は実運用では分からないので配備はできないが、
  <strong>base rate 補正で取り戻せる上限</strong>が測れる。</p>
  <p>結果は明快だった。<strong>LOSO-B の ECE は {lb_raw:.3f} → {lb_ora:.3f}（{cut_lb:.0f}% 減）</strong>、
  分布外全体でも {ece_raw_o:.3f} → {eo_o:.3f}。分布内は {ece_raw_i:.3f} → {eo_i:.3f} とほぼ動かない
  （シフトが無いのだから当然である）。分布内の ECE {ece_raw_i:.3f} を「較正の下限」とみなして超過分で測ると、
  <strong>LOSO-B の較正崩れの {share_lb:.0f}%、分布外全体では {share_o:.0f}% が base rate 由来</strong>ということになる
  （ロジットで測った場合の割合は次節で併記する）。
  効く場所も一貫している——オフセットが大きいユニットほど改善が大きく（LOSO-B {off_lb:+.2f} → {cut_lb:.0f}% 減）、
  オフセットがほぼゼロの field 3T では何も起きない。</p>
  <p>裏を返せば、<strong>残りは確率の形そのものの歪みであり、base rate 補正では届かない</strong>。
  切片を持つ較正（Platt scaling）に広げても埋められるのはこの上限までで、
  しかも実運用では test の陽性率を知り得ない以上、そこにも届かない。</p>"""

    rel_fig = ""
    if PRED is not None:
        _t = PRED[PRED["role"] == "test"]
        _b = _bins(_t[_t["run"].map(LABEL) == "LOSO-B"])
        _g = [np.log(o / (1 - o)) - np.log(p / (1 - p)) for p, o, n in _b if 0 < o < 1 and n >= 20]
        rel_fig = f"""
  <p>ECE は1つの数字に潰れてしまうので、<strong>どの確率帯でずれているのかを信頼度図で見た</strong>。
  分布内の曲線は対角線にほぼ乗る。分布外は全域で対角線の上、つまり
  <strong>どの確率帯でも実際の陽性率が予測を上回る（過小評価）</strong>。
  LOSO-B は特に顕著で、0.35 と予測した群の実際の陽性率は 0.75 に達する。</p>
  <p>重要なのは<strong>そのずれが高確率側に偏っておらず、全域でほぼ一様だ</strong>という点である。
  LOSO-B のずれをロジットで測ると <strong>+{np.mean(_g):.2f} ± {np.std(_g, ddof=1):.2f}</strong>
  （n≥20 の {len(_g)} ビン、予測確率 0.03〜0.64 の範囲）で、
  <strong>ビンをまたいでほぼ定数</strong>——これは定数のロジットオフセット、すなわち base rate のずれが
  持つ形そのものである。オラクルのオフセット {off_lb:+.2f} とも整合する。
  対処としては「高確率側だけを叩く」補正ではなく、全体を平行移動させる補正が要る、ということになる。</p>
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
        neg = sum(1 for x in rs if x["S_out"] < x["S_in"])
        err = float(np.mean([(r["S_wrong"] - r["S_correct"]) / r["S_correct"] * 100 for r in ROWS]))
        site_fig = f"""
  <p class="eyebrow" style="margin-top:8px">設計検証 §14.2 — 同一モデル内での in-site / out-site 比較</p>
  <p>ラン間で S を比べると訓練セットの大きさが交絡するので、<strong>同一モデルに
  in-site データと out-site データを通して比較した</strong>。LOSO なら inner-val が
  in-site の held-out（LOSO-B なら UCSF + UPenn）、test が out-site（同 UTSW）にあたる。
  vendor と field は train・val・test すべて UTSW なので、この対比自体が存在しない。対象は LOSO の6ラン。</p>
  <p>結果は out-site 側で median S が平均 <strong>{d:+.0f}%</strong>（{neg}/{len(rs)} ランで低下、
  最大 {min((x["S_out"] - x["S_in"]) / x["S_in"] * 100 for x in rs):.0f}%、
  1ランは逆に上昇）。<strong>向きは設計意図どおりだが、効果は弱く seed 間で一貫していない。</strong></p>
  <p>これは同じ S が誤答に対して示す反応と比べると際立つ。<strong>誤答時の S 低下は平均 {err:.0f}%、
  27/27 ランで例外なし</strong>だった。つまりこのモデルの不確実性は
  <strong>「間違えていること」には強く反応するが、「施設が変わったこと」自体にはほとんど反応しない</strong>。</p>
  <p>ただしこれは「検証に失敗した」のではなく、<strong>そもそもその能力を訓練していなかった</strong>と書くのが正確である。
  evidential loss が勾配を与えるのは予測の誤りに対してだけで、正則化項が罰するのも誤方向の evidence だけである。
  <strong>ドメインが変わっても予測さえ当たっていれば損失は下がらない</strong>。
  訓練信号にドメインの情報が一切入っていない以上、ドメイン検知能力が育たないのは道理であり、
  −{abs(d):.0f}% という弱い反応はむしろ設計どおりの帰結である。
  これは設計上の見落としであり、ドメイン検知を求めるなら
  損失かデータ設計の側にその信号を入れる必要がある——不確実性の定式化を変えるだけでは足りない。</p>
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

# ── how many labelled target cases buy the calibration back ───────────────────
SSC_SECTION = ""
if PRED is not None and HAS_CAL:
    from sample_size_curve import build as ssc_build, min_n_for

    _floor = _cal_mean(IND, "test_raw", "ece")
    _ssc = {u: ssc_build(_RD, u, reps=50, how="nll")["curve"] for u in ("LOSO-B", "LOSO-A")}
    from em_prior_shift_eval import per_run as _em_run
    _rng = np.random.default_rng(0)
    EM = {r["run"]: _em_run(PRED, r["run"], 50, _rng) for r in ROWS}

    def _emu(lab, k):
        return float(np.mean([EM[r["run"]][k] for r in ROWS if r["label"] == lab]))

    def _emg(kind, k):
        return float(np.mean([EM[r["run"]][k] for r in ROWS if r["kind"] == kind]))

    _lb = _ssc["LOSO-B"]
    _n80 = min_n_for(_lb, 0.08)
    _asy = float(np.mean([r["ece_asymptote"] for r in _lb]))
    _row = {r["n"]: r for r in _lb}
    _tbl = "".join(
        f'<tr><th scope="row">{r["n"]}</th><td class="n">{r["ece_mean"]:.3f}</td>'
        f'<td class="n">{r["ece_lo"]:.3f} – {r["ece_hi"]:.3f}</td>'
        f'<td class="n">{r["offset_mean"]:+.2f} ± {r["offset_sd"]:.2f}</td></tr>' for r in _lb)
    _la = {r["n"]: r for r in _ssc["LOSO-A"]}
    _POL = {}
    for _lab, _f in (("LOSO-B", "correction_policy_losob.json"),
                     ("LOSO-A", "correction_policy_losoa.json")):
        _p = _ROOT / "results" / _f
        if _p.is_file():
            _POL[_lab] = {r["n"]: r for r in json.loads(_p.read_text())}

    def _pol(lab, n, k):
        return _POL[lab][n][k]

    _poltbl = "".join(
        f'<tr><th scope="row">{esc(lab)}</th><td class="n">{n}</td>'
        f'<td class="n">{r["p_harm"]*100:.0f}%</td><td class="n">{r["p_harm_ruled"]*100:.0f}%</td>'
        f'<td class="n">{r["abstain"]*100:.0f}%</td><td class="n">{r["gain_mean"]:+.3f}</td>'
        f'<td class="n">{r["loss_when_harm"]:.3f}</td></tr>'
        for lab in ("LOSO-B", "LOSO-A") for n, r in _POL.get(lab, {}).items())

    _emtbl = "".join(
        f'<tr><th scope="row">{esc(JA[lab])}</th>'
        f'<td class="n">{_emu(lab,"pi_source")*100:.1f}%</td>'
        f'<td class="n">{_emu(lab,"pi_true")*100:.1f}%</td>'
        f'<td class="n">{_emu(lab,"pi_em")*100:.1f}%</td>'
        f'<td class="n">{(_emu(lab,"pi_em")-_emu(lab,"pi_true"))*100:+.1f}pt</td>'
        f'<td class="n">{_emu(lab,"ece_raw"):.3f} → {_emu(lab,"ece_em"):.3f}</td></tr>'
        for lab, _k, _rs in UNITS)

    SSC_SECTION = f"""
<section class="finding">
  <p class="eyebrow">所見 — 処方箋</p>
  <h2 class="serif">target 施設の {_n80} 例で、較正はほぼ取り戻せる</h2>
  <p>ここまでは「事後較正では届かない」という問題提起で終わっている。だが前節の分解が正しいなら、
  必要なのは<strong>定数を1つ、target 側で決めること</strong>だけのはずである。
  そこで <strong>target 施設のラベル付き症例を N 例だけ使って切片を推定し、残りの症例で ECE を測った</strong>。
  N ごとに 50 回のブートストラップを行い、同じ評価集合の上で補正なしの値とも比べている。</p>
  <p>LOSO-B の結果は明快で、<strong>N = {_n80} 例で ECE 平均 {_row[_n80]["ece_mean"]:.3f}</strong>、
  N = 100 で {_row[100]["ece_mean"]:.3f}、補正なしの {_row[_n80]["ece_raw"]:.3f} から大きく下がる。
  全例を使った場合の到達点は {_asy:.3f} で、<strong>分布内の水準 {_floor:.3f} とほぼ同じ</strong>である。
  LOSO-A でも N = 50 で {_la[50]["ece_mean"]:.3f}（補正なし {_la[50]["ece_raw"]:.3f}）と同じ傾向が出る。</p>
  <p>ここで前節の数字が繋がる。<strong>切片を直接あてはめると LOSO-B の較正崩れはほぼ全部消える</strong>——
  つまり崩れの正体は<strong>ほぼ純粋な定数ロジットオフセット</strong>である。
  一方その定数の大きさは、base rate のずれだけでは説明しきれない。
  実測のずれ +1.42 に対し base rate から予測される値は +1.09 で、
  <strong>ロジットベースでは 77%、ECE ベースでは 86%</strong> が base rate 由来にあたる。
  ECE が非線形な指標なので両者は一致しないが、どちらで測っても主要因が base rate であることは変わらない。
  残りは施設が変わったこと自体が動作点をずらした分である。</p>
  <p>実務的な読み方はこうなる。<strong>未知の施設に持ち込むとき、その施設のラベル付き症例が数十例あれば
  確率値は使える水準に戻る。</strong>ただし {_lb[-1]["n"]} 例まで増やしても
  95% 区間の上端は {_lb[-1]["ece_hi"]:.3f} までしか下がらない。
  <strong>平均としては届くが、個々の施設で必ず届くとは言えない</strong>——
  N が小さいほど推定オフセットのばらつきが大きく（N=10 で ± {_row[10]["offset_sd"]:.2f}、
  N=100 で ± {_row[100]["offset_sd"]:.2f}）、外す側に外すと補正が害になり得る。</p>
  <figure class="figure">
    {fig_sample_size(_lb, _floor, em=_emu("LOSO-B", "ece_em"))}
    <figcaption>LOSO-B。線は50回×3 seed のブートストラップ平均、帯は95%区間。
    横軸は対数目盛。各 N で切片は N 例のみから推定し、ECE は残りの症例で測っている。
    vendor と field は base rate のずれがほぼ無いため、この補正では改善しない（前節のとおり）。</figcaption>
  </figure>
  <div class="tablewrap"><table><caption>LOSO-B — N 例で切片を推定したときの ECE</caption>
    <thead><tr><th scope="col">N</th><th scope="col">ECE 平均</th><th scope="col">95% 区間</th>
    <th scope="col">推定オフセット</th></tr></thead><tbody>{_tbl}</tbody></table></div>
</section>

<section class="finding">
  <p class="eyebrow">所見 — ゼロショットの限界</p>
  <h2 class="serif">ラベル無しで同じ定数を当てられるか。原理的に当てられない</h2>
  <p><span class="lead">仮説</span>較正崩れがほぼ純粋な定数ロジットオフセットなら、
  それは<strong>ラベルシフト</strong>——クラス条件付き分布 p(x|y) は動かず、有病率 p(y) だけが動いた——
  の兆候かもしれない。もしそうなら推定すべきはスカラー1個で、しかもラベルは要らない。
  <strong>EM prior shift（Saerens-Latinne-Decaestecker）は、予測確率の分布の形だけから
  target の有病率を推定する手法</strong>である。27ラン全てに適用した。</p>
  <p><span class="lead">実測</span>外れた。LOSO-B は真の有病率 {_emu("LOSO-B","pi_true")*100:.1f}% に対し
  EM の推定 <strong>{_emu("LOSO-B","pi_em")*100:.1f}%</strong>、LOSO-A は真値 {_emu("LOSO-A","pi_true")*100:.1f}% に対し
  <strong>{_emu("LOSO-A","pi_em")*100:.1f}%</strong>——どちらも 14 ポイント近い過小推定である。
  当たるオフセットも小さすぎ（LOSO-B で {_emu("LOSO-B","offset_em"):+.2f}、必要な値は {_emu("LOSO-B","offset_labelled"):+.2f}）、
  LOSO-A では {_emu("LOSO-A","offset_em"):+.2f} と<strong>符号が逆</strong>になる。
  分布外の ECE は {_emg("OOD","ece_raw"):.3f} → <strong>{_emg("OOD","ece_em"):.3f} と悪化した</strong>。</p>
  <p><span class="lead">反証の排除</span>実装の誤りでも、サンプル不足でも、数値の不安定でもない。
  <strong>27/27 ランで収束した</strong>（発散も振動もなし）。
  <strong>分布内では推定が正確</strong>で、真値 18% 前後に対し誤差 0〜3 ポイントに収まる。
  <strong>合成データでは回収できる</strong>——p(x|y) を固定して有病率を 12% → 30% に動かした設定では
  EM は正しく 30% を返す（単体テスト）。
  <strong>ラベルなし症例数を増やしても改善しない</strong>：LOSO-B の推定は n=20 で 18.8% ± 15.3、
  n=200 で 14.6% ± 4.1 と、ばらつきが縮むだけで<strong>間違った値に精度よく収束する</strong>。
  系統誤差であって分散の問題ではない。残るのは仮定の破れだけである。</p>
  <p><span class="lead">原因</span>施設が変われば撮像装置もプロトコルも患者層も変わる。
  <strong>p(x|y) 自体が動いており、ラベルシフトの仮定が成立していない。</strong>
  決定的な数字はこれである——<strong>LOSO-B の test では予測確率の平均が 12.3%、真の有病率は 28.4%</strong>。
  真の有病率が<em>上がっている</em>のにモデルの出力は<em>押し下げられている</em>。
  ラベルシフトなら両者は同じ方向に動くはずで、逆方向に動いた時点で仮定は否定されている。
  EM はこの押し下げられた分布を「陽性が少ない証拠」と読み、有病率が下がったと結論する。</p>
  <p><span class="lead">一般化</span>失敗は構造的である。
  <strong>EM が使える唯一の信号は、まさにシフトによって歪められた予測確率の分布そのものだ。
  歪みを、歪んだものから推定することはできない。</strong>
  ラベルはこの循環の外にある。前節の N 例曲線が効いた理由と、EM が効かない理由は、同じ構造から出てくる。
  同じことは予測分布のみに依拠する他のゼロショット較正——出力エントロピーや確信度ヒストグラムに基づく手法——
  にも当てはまるはずである。</p>
  <p><span class="lead">結論</span><strong>ゼロショットでの較正回復はこの設定では成立しない。
  target 施設の少数ラベルが要る。</strong>上の図で EM の水平線は {_emu("LOSO-B","ece_em"):.3f} にあり、
  ラベル {_lb[0]["n"]} 例の点（{_lb[0]["ece_mean"]:.3f}）にすら届かない。
  <strong>ラベル無しの推定は、ラベル 10 例分の価値にも満たない。</strong></p>
  <div class="tablewrap"><table><caption>EM の有病率推定（ラベル不使用、3 seed 平均）</caption>
    <thead><tr><th scope="col">ユニット</th><th scope="col">訓練時</th><th scope="col">真の値</th>
    <th scope="col">EM 推定</th><th scope="col">誤差</th><th scope="col">ECE 生 → EM</th></tr></thead>
    <tbody>{_emtbl}</tbody></table></div>
</section>

<section class="finding">
  <p class="eyebrow">運用 — いつ補正してよいか</p>
  <h2 class="serif">平均の利得だけでは足りない。害を出す確率で決める</h2>
  <p>N 例曲線が示すのは平均の利得である。だが公開ツールに必要なのはもう一方の数字——
  <strong>補正がかえって悪化させる確率</strong>だ。N が小さいとオフセットの推定は荒く、
  符号を外せば補正は害になる。EM の LOSO-A がまさにその実例で、
  符号を逆に推定して ECE を {_emu("LOSO-A","ece_raw"):.3f} → {_emu("LOSO-A","ece_em"):.3f} に悪化させた。</p>
  <p>そこで各 N について、<strong>補正後の ECE が補正なしを上回った割合</strong>を数えた。
  あわせて配備可能な判断基準も評価した——<strong>N 例を復元抽出し直してオフセットを推定し直し、
  そのばらつき（SE）の 2 倍を |オフセット| が超えたときだけ補正する</strong>というルールである。
  施設が手元に持っている情報だけで判定できる。</p>
  <div class="tablewrap"><table><caption>補正が害になる確率と、見送りルールの効果（50回 × 3 seed）</caption>
    <thead><tr><th scope="col">ユニット</th><th scope="col">N</th><th scope="col">悪化する確率</th>
    <th scope="col">ルール適用後</th><th scope="col">見送り率</th><th scope="col">ECE 改善</th>
    <th scope="col">悪化時の幅</th></tr></thead><tbody>{_poltbl}</tbody></table></div>
  <p><strong>シフトが大きいユニットでは補正はほぼ安全である。</strong>LOSO-B は N=10 でも悪化率 {_pol("LOSO-B",10,"p_harm")*100:.0f}%、
  N=50 以上では 0% で、改善幅も {_pol("LOSO-B",50,"gain_mean"):.3f} と大きい。
  <strong>危ないのはシフトが小さいときだ。</strong>LOSO-A は N=10 で悪化率 {_pol("LOSO-A",10,"p_harm")*100:.0f}%——
  ほぼコイン投げで、期待利得は {_pol("LOSO-A",10,"gain_mean"):+.3f} とマイナスである。
  補正すべき量が小さいほど、推定誤差が相対的に大きくなるためで、道理ではある。</p>
  <p>見送りルールはこの危ない側を守る。LOSO-A の悪化率は N=10 で {_pol("LOSO-A",10,"p_harm")*100:.0f}% → {_pol("LOSO-A",10,"p_harm_ruled")*100:.0f}%、
  N=20 で {_pol("LOSO-A",20,"p_harm")*100:.0f}% → {_pol("LOSO-A",20,"p_harm_ruled")*100:.0f}% に下がる。
  代償として、シフトが大きいときには過剰に見送る（LOSO-B の N=10 で {_pol("LOSO-B",10,"abstain")*100:.0f}% 見送り、
  利得は {_pol("LOSO-B",10,"gain_mean"):.3f} → {_pol("LOSO-B",10,"gain_ruled"):.3f} に減る）。
  <strong>だが配備時にはどちらの状況にいるか分からない。</strong>取りこぼしより害の回避を優先するのが妥当だと考える。</p>
  <p class="eyebrow" style="margin-top:8px">ポータルの運用ルール（案）</p>
  <ul class="rules">
    <li><b>N &lt; 20</b><span>補正しない。確定診断例が 20 例に満たない施設では、
      推定オフセットの SE が {_pol("LOSO-A",10,"se_mean"):.2f} 前後あり、期待利得が負になり得る。
      確率値は補正なしで提供し、但し書きを添える。</span></li>
    <li><b>20 ≤ N &lt; 50</b><span>ルールが発火したときだけ補正する。
      |オフセット| がブートストラップ SE の 2 倍を超えない限り適用しない。
      この帯では見送りが多数派になる（LOSO-A で {_pol("LOSO-A",20,"abstain")*100:.0f}%）が、それが正しい挙動である。</span></li>
    <li><b>N ≥ 50</b><span>補正を既定にし、ルールはガードとして残す。
      悪化率は {_pol("LOSO-A",50,"p_harm_ruled")*100:.0f}% 以下、改善幅は LOSO-B で {_pol("LOSO-B",50,"gain_mean"):.3f}。
      それでも悪化したときの幅は {_pol("LOSO-A",50,"loss_when_harm"):.3f} 程度で、破滅的ではない。</span></li>
    <li><b>常時</b><span>推定オフセットとその SE、および補正の適用可否を画面に出す。
      補正が効いているかどうかを利用者が知らないまま確率値を読む状態を作らない。</span></li>
  </ul>
  <p class="eyebrow" style="margin-top:8px">補正なしで提供する場合の但し書き（案）</p>
  <p class="callout">この確率値は<strong>他施設のデータで学習したモデル</strong>によるものです。
  当施設の IDH 変異頻度が学習データと異なる場合、確率値は<strong>系統的にずれます</strong>。
  検証では、真の陽性率が 28.4% の施設で予測確率の平均が 12.3% と低く出ました。
  <strong>どの症例がより疑わしいかという順位付けは保たれます</strong>が
  （AUC は施設が変わっても 0.85〜0.87 を維持）、
  <strong>確率値そのものを閾値判断に用いないでください</strong>。
  当施設の確定例を 20 例以上登録いただくと、施設ごとの補正が有効になります。</p>
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
{SSC_SECTION}
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
