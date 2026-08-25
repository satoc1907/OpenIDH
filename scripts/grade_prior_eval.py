"""Does a grade covariate close the gap age left open? (zero-shot calibration, third attempt)

Age explained 81-90% of the LOSO-A prevalence shift but only 46% of LOSO-B, because
UPenn is a GBM-only cohort at 3.1% mutant — grade composition, which age cannot see.
Grade is not known before surgery, but a clinician always has a provisional read
("looks high-grade"), which a portal can ask for in one question. This measures what
that answer would buy.

WHO 2021 defines grade 4 partly by IDH status, so a confirmed grade leaks the label.
Three levels bracket the effect instead of pretending it away:

  A  confirmed grade {2,3,4,NA}          upper bound — partly tautological
  B  binary high(4) / low(2,3)           what a clinician can actually answer
  C  binary with 15% misclassification   pre-operative reads are 80-90% accurate

Level C flips the TARGET side only: training cohorts have pathology, an incoming
case has a guess. --flip-both flips the training side too.

    uv run python scripts/grade_prior_eval.py
    uv run python scripts/grade_prior_eval.py --reps 50 --flip-rate 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from age_prior_eval import FOLDS, _lo, subject_table  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.calibrate import apply_affine, prior_offset  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402


def grade_table(paths) -> pd.DataFrame:
    """subject_id -> WHO grade. UPenn carries no grade column: the cohort is
    UPENN-GBM, i.e. glioblastoma, so grade 4 by construction — an assumption of the
    dataset's definition rather than a value read from metadata."""
    md = Path(paths.metadata_dir)
    u = pd.read_csv(md / "UCSF-PDGM-metadata_v5.csv")
    ucsf = pd.DataFrame({"subject_id": u.iloc[:, 0].astype(str), "grade": u["WHO CNS Grade"]})
    t = pd.read_csv(md / "UTSW_Glioma_Metadata.tsv", sep="\t")
    utsw = pd.DataFrame({"subject_id": t.iloc[:, 0].astype(str), "grade": t["Tumor Grade"]})
    return pd.concat([ucsf, utsw], ignore_index=True)


def attach_grade(subj: pd.DataFrame, paths) -> pd.DataFrame:
    g = grade_table(paths)
    out = subj.merge(g, on="subject_id", how="left")
    out.loc[out["site"] == "UPenn", "grade"] = 4.0          # cohort definition
    return out


def _encode(grade: pd.Series, level: str, rng=None, flip: float = 0.0,
            mode: str = "symmetric") -> np.ndarray:
    """One-hot the grade covariate at the requested level. NA stays its own column.

    Two ways to corrupt the binary read:
      symmetric  every case misread with probability `flip`. High grade is the
                 majority (68% of UTSW), so symmetric error moves more cases
                 high->low than low->high and drags the observed proportion toward
                 50/50 — which inflates the estimated prevalence, since low grade
                 carries the mutations. That is a real deployment risk, not an
                 artefact, but it confounds "is the estimate robust to misreads".
      balanced   per-class rates chosen so the marginal is preserved
                 (p*a == (1-p)*b), isolating misclassification from that drift.
    """
    g = grade.to_numpy(dtype=float)
    na = np.isnan(g)
    if level == "A":
        return np.column_stack([(g == 2) & ~na, (g == 3) & ~na, (g == 4) & ~na, na]).astype(float)
    high = (g == 4) & ~na
    if flip and rng is not None:
        obs = ~na
        if mode == "balanced":
            p = high[obs].mean() if obs.any() else 0.5
            b = min(1.0, flip * p / max(1 - p, 1e-9))
            rate = np.where(high, flip, b)
        else:
            rate = np.full(len(g), flip)
        roll = rng.random(len(g)) < rate
        high = np.where(obs & roll, ~high, high)
    return np.column_stack([high & ~na, (~high) & ~na, na]).astype(float)


def _fit(age, X, y):
    m = LogisticRegression(max_iter=5000, C=1e6)
    m.fit(np.column_stack([age, X]), y)
    return lambda a, Z: m.predict_proba(np.column_stack([a, Z]))[:, 1]


def evaluate(paths, runs_dir: Path, reps: int, flip: float, flip_both: bool, seed: int = 0):
    subj = attach_grade(subject_table(paths), paths)
    rng = np.random.default_rng(seed)
    out = []
    for label, sf in FOLDS.items():
        sp = pd.read_csv(paths.splits_dir / sf).merge(
            subj[["subject_id", "age", "grade"]], on="subject_id", how="left")
        sp["mut"] = (sp["idh"].astype(str).str.strip() == "Mut").astype(int)
        tr = sp[sp.split_role == "train"].dropna(subset=["age"])
        va = sp[sp.split_role == "val"].dropna(subset=["age"])
        te = sp[sp.split_role == "test"].dropna(subset=["age"])
        pi_train, pi_val, pi_true = (float(x["mut"].mean()) for x in (tr, va, te))

        row = {"label": label, "pi_train": pi_train, "pi_val": pi_val, "pi_true": pi_true,
               "n_test": len(te), "grade_na_test": int(te["grade"].isna().sum()),
               "levels": {}}
        for level in ("A", "B", "C", "Cb"):
            def est(rng_=None, fl=0.0, md="symmetric"):
                Xtr = _encode(tr["grade"], level, rng_ if flip_both else None,
                              fl if flip_both else 0.0, md)
                Xte = _encode(te["grade"], level, rng_, fl, md)
                f = _fit(tr["age"].to_numpy(float), Xtr, tr["mut"].to_numpy(int))
                return float(f(te["age"].to_numpy(float), Xte).mean())

            if level in ("C", "Cb"):
                md = "balanced" if level == "Cb" else "symmetric"
                vals = [est(rng, flip, md) for _ in range(reps)]
                pi_hat, lo, hi = float(np.mean(vals)), *np.percentile(vals, [2.5, 97.5])
            else:
                pi_hat = est()
                lo = hi = pi_hat
            den = pi_true - pi_train
            row["levels"][level] = {
                "pi_hat": pi_hat, "pi_lo": float(lo), "pi_hi": float(hi),
                "explained": (pi_hat - pi_train) / den if abs(den) > 1e-9 else float("nan"),
                "offset": _lo(pi_hat) - _lo(pi_val)}
        row["offset_oracle"] = prior_offset(pi_val, pi_true)
        out.append(row)
    return out, subj


def add_ece(rows, runs_dir: Path, paths):
    """ECE on the real predictions when each level's offset is applied."""
    import re
    tag = {"LOSO-A": "foldA", "LOSO-B": "foldB"}
    for r in rows:
        acc = {k: [] for k in ("raw", "A", "B", "C", "Cb", "oracle")}
        for seed in range(3):
            f = runs_dir / f"{tag[r['label']]}_s{seed}" / "predictions.csv"
            if not f.is_file():
                continue
            t = pd.read_csv(f)
            t = t[t.role == "test"]
            a, b, y = t["alpha"].to_numpy(), t["beta"].to_numpy(), t["y_true"].to_numpy(float)
            def one(off):
                aa, bb = apply_affine(a, b, offset=off)
                m = compute_metrics(aa, bb, y)
                pr = aa / (aa + bb)
                sens = float(((pr > .5) & (y == 1)).sum() / max((y == 1).sum(), 1))
                spec = float(((pr <= .5) & (y == 0)).sum() / max((y == 0).sum(), 1))
                return {"ece": m["ece"], "auc": m["auc"], "sens": sens, "spec": spec,
                        "bacc": (sens + spec) / 2}

            acc["raw"].append(one(0.0))
            for lv in ("A", "B", "C", "Cb"):
                acc[lv].append(one(r["levels"][lv]["offset"]))
            acc["oracle"].append(one(r["offset_oracle"]))
        r["metrics"] = {k: {m: float(np.mean([x[m] for x in v]))
                            for m in ("ece", "auc", "sens", "spec", "bacc")}
                        for k, v in acc.items() if v}
        r["ece"] = {k: v["ece"] for k, v in r["metrics"].items()}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--paths-file", default=None)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--flip-rate", type=float, default=0.15)
    ap.add_argument("--flip-both", action="store_true",
                    help="also corrupt the training grades (default: target side only)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    paths = load_paths(a.paths_file or "configs/paths.local.yaml")
    runs_dir = Path(a.runs_dir)
    rows, subj = evaluate(paths, runs_dir, a.reps, a.flip_rate, a.flip_both)
    rows = add_ece(rows, runs_dir, paths)

    d = subj.dropna(subset=["age"])
    print("## grade の分布とラベルリークの確認\n")
    print("| サイト | grade 2 | grade 3 | grade 4 | NA | grade 4 の WT 率 |")
    print("|---|---|---|---|---|---|")
    for s, g in d.groupby("site"):
        c = g["grade"].value_counts(dropna=False)
        g4 = g[g["grade"] == 4]
        wt = 1 - g4["mut"].mean() if len(g4) else float("nan")
        print(f"| {s} | {int(c.get(2.0,0))} | {int(c.get(3.0,0))} | {int(c.get(4.0,0))} | "
              f"{int(g['grade'].isna().sum())} | {wt*100:.1f}% |")
    g4 = d[d["grade"] == 4]
    print(f"\n全体で grade 4 の {(1 - g4['mut'].mean())*100:.1f}% が WT"
          f"（{len(g4)}例）。WHO 2021 は grade 4 の定義に IDH を含むので、水準 A はこの分だけ有利になる。")

    print("\n## 有病率推定\n")
    print("| fold | π_train | π_true | 年齢のみ | 水準A 確定 | 水準B 二値 | 水準C ノイズ |")
    print("|---|---|---|---|---|---|---|")
    age_only = {"LOSO-A": (0.199, 0.81), "LOSO-B": (0.195, 0.46)}   # 既報（年齢のみ・線形）
    for r in rows:
        L = r["levels"]
        c = f"{L['C']['pi_hat']*100:.1f}% [{L['C']['pi_lo']*100:.1f}–{L['C']['pi_hi']*100:.1f}]"
        print(f"| {r['label']} | {r['pi_train']*100:.1f}% | {r['pi_true']*100:.1f}% | "
              f"{age_only[r['label']][0]*100:.1f}% | {L['A']['pi_hat']*100:.1f}% | "
              f"{L['B']['pi_hat']*100:.1f}% | {c} |")
    print("\n| fold | 説明率 年齢のみ | 水準A | 水準B | 水準C 対称 | 水準C 周辺保存 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        L = r["levels"]
        print(f"| {r['label']} | {age_only[r['label']][1]*100:.0f}% | {L['A']['explained']*100:.0f}% | "
              f"{L['B']['explained']*100:.0f}% | {L['C']['explained']*100:.0f}% | "
              f"{L['Cb']['explained']*100:.0f}% |")

    print("\n## 切片補正としての ECE（3 seed 平均）\n")
    print("| 手法 | ラベル | " + " | ".join(r["label"] for r in rows) + " |")
    print("|---|---|---|---|")
    for key, nm, lb in (("raw", "補正なし", "—"), ("A", "年齢+grade 水準A（上限）", "0 例"),
                        ("B", "年齢+grade 水準B（二値）", "0 例"),
                        ("C", "年齢+grade 水準C（ノイズ・対称）", "0 例"),
                        ("Cb", "年齢+grade 水準C（ノイズ・周辺保存）", "0 例"),
                        ("oracle", "oracle（真の有病率）", "—")):
        print(f"| {nm} | {lb} | " + " | ".join(f"{r['ece'][key]:.3f}" for r in rows) + " |")
    print("\n| fold | 推定オフセット A | B | C | C-balanced | oracle |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        L = r["levels"]
        print(f"| {r['label']} | {L['A']['offset']:+.2f} | {L['B']['offset']:+.2f} | "
              f"{L['C']['offset']:+.2f} | {L['Cb']['offset']:+.2f} | {r['offset_oracle']:+.2f} |")

    print("\n## 閾値 0.5 での分類性能（3 seed 平均）\n")
    print("| fold | 手法 | 感度 | 特異度 | balanced acc |")
    print("|---|---|---|---|---|")
    for r in rows:
        for key, nm in (("raw", "補正なし"), ("A", "水準A 確定"), ("B", "水準B 二値"),
                        ("C", "水準C ノイズ"), ("Cb", "水準C 周辺保存"), ("oracle", "oracle")):
            m = r["metrics"][key]
            print(f"| {r['label']} | {nm} | {m['sens']*100:.1f}% | {m['spec']*100:.1f}% | "
                  f"{m['bacc']*100:.1f}% |")
    d = max(abs(r["metrics"][k]["auc"] - r["metrics"]["raw"]["auc"])
            for r in rows for k in ("A", "B", "C", "Cb", "oracle"))
    print(f"\n検算: AUC の最大変化量 {d:.2e}（施設一律の切片なので 0 のはず）")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
