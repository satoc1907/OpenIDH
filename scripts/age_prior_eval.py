"""Can age alone estimate the target prevalence? (zero-shot calibration, second attempt)

EM failed because its only input was the distorted predicted distribution. Age is
outside that loop: it never passes through the network, a scanner cannot shift it,
and the age-IDH relation is close to identical across the three sites. So fit
pi(a) on the training sites, push the target site's *age distribution* through it,
and read off a prevalence — no labels, no predictions.

    uv run python scripts/age_prior_eval.py
    uv run python scripts/age_prior_eval.py --json results/age_prior.json

Reports the site age distributions and their KS distances, the estimated vs true
prevalence per LOSO fold, the resulting intercept correction, its ECE, a
per-subject variant, and threshold behaviour at 0.5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import load_runs  # noqa: E402
from openidh_model.data.metadata import lookup_age_sex  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.calibrate import apply_affine, prior_offset, to_logit  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402
from sample_size_curve import _fit_offset  # noqa: E402

FOLDS = {"LOSO-A": "splits_loso_foldA.csv", "LOSO-B": "splits_loso_foldB.csv"}
_EPS = 1e-7


def _lo(p):
    p = float(np.clip(p, _EPS, 1 - _EPS))
    return np.log(p / (1 - p))


def subject_table(paths) -> pd.DataFrame:
    """One row per subject: site, age, sex, label. Same for every split file."""
    sp = pd.read_csv(paths.splits_dir / "splits_loso_foldA.csv")
    rows = []
    for _, r in sp.iterrows():
        age, sex = lookup_age_sex(paths.metadata_dir, r["site"], r["subject_id"])
        rows.append({"subject_id": r["subject_id"], "site": r["site"], "age": age,
                     "sex": sex, "mut": int(str(r["idh"]).strip() == "Mut")})
    return pd.DataFrame(rows)


def site_stats(d: pd.DataFrame) -> dict:
    out = {"missing_age": int(d["age"].isna().sum()), "sites": {}, "ks": {}}
    d = d.dropna(subset=["age"])
    for s, g in d.groupby("site"):
        q1, q3 = g["age"].quantile([.25, .75])
        out["sites"][s] = {"n": len(g), "mean": float(g["age"].mean()),
                           "median": float(g["age"].median()), "q1": float(q1), "q3": float(q3),
                           "iqr": float(q3 - q1), "mut_rate": float(g["mut"].mean()),
                           "age_mut": float(g[g["mut"] == 1]["age"].mean()),
                           "age_wt": float(g[g["mut"] == 0]["age"].mean())}
    for a, b in (("UCSF", "UTSW"), ("UCSF", "UPenn"), ("UPenn", "UTSW")):
        k = stats.ks_2samp(d[d.site == a]["age"], d[d.site == b]["age"])
        out["ks"][f"{a} vs {b}"] = {"D": float(k.statistic), "p": float(k.pvalue),
                                    "mean_diff": float(d[d.site == a]["age"].mean()
                                                       - d[d.site == b]["age"].mean())}
    return out


def _fit_age_model(age, y, quadratic: bool):
    X = np.column_stack([age, age ** 2]) if quadratic else age.reshape(-1, 1)
    m = LogisticRegression(max_iter=2000, C=1e6).fit(X, y)   # essentially unpenalised
    return lambda a: m.predict_proba(
        np.column_stack([a, a ** 2]) if quadratic else np.asarray(a).reshape(-1, 1))[:, 1]


def _cls_metrics(p, y, thr=0.5):
    pred = (p > thr).astype(int)
    y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"sens": sens, "spec": spec, "bacc": (sens + spec) / 2, "tp": tp, "fn": fn,
            "tn": tn, "fp": fp}


def evaluate(paths, runs_dir: Path, quadratic: bool = False) -> list[dict]:
    subj = subject_table(paths)
    meta = [r for r in load_runs(runs_dir) if r["label"] in FOLDS]
    out = []
    for r in meta:
        sp = pd.read_csv(paths.splits_dir / FOLDS[r["label"]])
        sp = sp.merge(subj[["subject_id", "age"]], on="subject_id", how="left")
        sp["mut"] = (sp["idh"].astype(str).str.strip() == "Mut").astype(int)
        tr = sp[sp.split_role == "train"].dropna(subset=["age"])
        va = sp[sp.split_role == "val"].dropna(subset=["age"])
        te = sp[sp.split_role == "test"].dropna(subset=["age"])

        # pi(a) from the TRAINING sites only; test contributes ages, never labels
        pi_of = _fit_age_model(tr["age"].to_numpy(float), tr["mut"].to_numpy(int), quadratic)
        pi_train, pi_val = float(tr["mut"].mean()), float(va["mut"].mean())
        pi_true = float(te["mut"].mean())
        pi_hat = float(pi_of(te["age"].to_numpy(float)).mean())
        pi_hat_train = float(pi_of(tr["age"].to_numpy(float)).mean())
        denom = pi_true - pi_train
        explained = (pi_hat - pi_train) / denom if abs(denom) > 1e-9 else float("nan")

        pred = pd.read_csv(runs_dir / r["run"] / "predictions.csv")
        t = pred[pred.role == "test"].merge(subj[["subject_id", "age"]], on="subject_id", how="left")
        a, b, y = t["alpha"].to_numpy(), t["beta"].to_numpy(), t["y_true"].to_numpy(float)

        off_age = _lo(pi_hat) - _lo(pi_val)                       # uniform intercept
        off_oracle = prior_offset(pi_val, pi_true)
        off_lab = _fit_offset(a, b, y, pi_val, "nll")
        # per-subject: how far this subject's age-implied prevalence sits from the
        # training marginal. (The literal b_i = logit(pi(a)) - logit(pi(a)) is 0.)
        pi_i = pi_of(t["age"].to_numpy(float))
        off_i = np.array([_lo(p) - _lo(pi_val) for p in pi_i])

        def m(off):
            aa, bb = apply_affine(a, b, offset=off)
            mm = compute_metrics(aa, bb, y)
            mm.update(_cls_metrics(aa / (aa + bb), y))
            return mm

        out.append({
            "run": r["run"], "label": r["label"], "seed": r["seed"], "quadratic": quadratic,
            "pi_train": pi_train, "pi_val": pi_val, "pi_true": pi_true,
            "pi_age": pi_hat, "pi_age_on_train": pi_hat_train, "explained": explained,
            "age_mean_train": float(tr["age"].mean()), "age_mean_test": float(te["age"].mean()),
            "offset_age": off_age, "offset_oracle": off_oracle, "offset_labelled": off_lab,
            "offset_age_subject_sd": float(np.std(off_i)),
            "raw": m(0.0), "age": m(off_age), "age_subject": m(off_i),
            "oracle": m(off_oracle), "labelled": m(off_lab),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--paths-file", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    paths = load_paths(a.paths_file or "configs/paths.local.yaml")
    runs_dir = Path(a.runs_dir)

    subj = subject_table(paths)
    st = site_stats(subj)
    print(f"## Step 1 — 年齢分布（欠測 {st['missing_age']}/{len(subj)}）\n")
    print("| サイト | n | 平均 | 中央値 | IQR | 変異率 | Mut 平均年齢 | WT 平均年齢 |")
    print("|---|---|---|---|---|---|---|---|")
    for s, v in st["sites"].items():
        print(f"| {s} | {v['n']} | {v['mean']:.1f} | {v['median']:.1f} | "
              f"{v['q1']:.0f}–{v['q3']:.0f} | {v['mut_rate']*100:.1f}% | "
              f"{v['age_mut']:.1f} | {v['age_wt']:.1f} |")
    print("\n| KS 検定 | D | p | 平均差 |")
    print("|---|---|---|---|")
    for k, v in st["ks"].items():
        print(f"| {k} | {v['D']:.3f} | {v['p']:.3g} | {v['mean_diff']:+.1f} 歳 |")

    res = {q: evaluate(paths, runs_dir, q) for q in (False, True)}
    for q in (False, True):
        tag = "2次項あり" if q else "線形"
        print(f"\n## Step 2 — 年齢からの有病率推定（{tag}）\n")
        print("| fold | π_train | π_true | π̂_age | 説明率 | 平均年齢 train→test |")
        print("|---|---|---|---|---|---|")
        for lab in FOLDS:
            rs = [x for x in res[q] if x["label"] == lab]
            g = lambda k: float(np.mean([x[k] for x in rs]))  # noqa: E731
            print(f"| {lab} | {g('pi_train')*100:.1f}% | {g('pi_true')*100:.1f}% | "
                  f"{g('pi_age')*100:.1f}% | **{g('explained')*100:.0f}%** | "
                  f"{g('age_mean_train'):.1f} → {g('age_mean_test'):.1f} 歳 |")

    rows = res[False]
    print("\n## Step 3 — 切片補正としての ECE\n")
    print("| 手法 | ラベル | " + " | ".join(FOLDS) + " |")
    print("|---|---|---|---|")
    for key, nm, lb in (("raw", "補正なし", "—"), ("age", "年齢ベース切片", "0 例"),
                        ("age_subject", "年齢ベース・症例ごと", "0 例"),
                        ("oracle", "oracle（真の有病率）", "—"),
                        ("labelled", "全例ラベル切片（下限）", "全例")):
        cells = []
        for lab in FOLDS:
            rs = [x for x in rows if x["label"] == lab]
            cells.append(f"{np.mean([x[key]['ece'] for x in rs]):.3f}")
        print(f"| {nm} | {lb} | " + " | ".join(cells) + " |")

    print("\n## Step 4 — 推定オフセット\n")
    print("| fold | 年齢ベース | 症例ごとの sd | oracle | 全例ラベル |")
    print("|---|---|---|---|---|")
    for lab in FOLDS:
        rs = [x for x in rows if x["label"] == lab]
        g = lambda k: float(np.mean([x[k] for x in rs]))  # noqa: E731
        print(f"| {lab} | {g('offset_age'):+.2f} | ±{g('offset_age_subject_sd'):.2f} | "
              f"{g('offset_oracle'):+.2f} | {g('offset_labelled'):+.2f} |")

    print("\n## Step 5 — 閾値 0.5 での分類性能\n")
    print("| fold | 手法 | 感度 | 特異度 | balanced acc | AUC |")
    print("|---|---|---|---|---|---|")
    for lab in FOLDS:
        rs = [x for x in rows if x["label"] == lab]
        for key, nm in (("raw", "補正なし"), ("age", "年齢ベース"),
                        ("age_subject", "年齢・症例ごと"), ("oracle", "oracle"),
                        ("labelled", "全例ラベル")):
            g = lambda k: float(np.mean([x[key][k] for x in rs]))  # noqa: E731
            print(f"| {lab} | {nm} | {g('sens')*100:.1f}% | {g('spec')*100:.1f}% | "
                  f"{g('bacc')*100:.1f}% | {g('auc'):.4f} |")
    d = max(abs(x[k]["auc"] - x["raw"]["auc"]) for x in rows
            for k in ("age", "oracle", "labelled"))
    ds = max(abs(x["age_subject"]["auc"] - x["raw"]["auc"]) for x in rows)
    print(f"\n検算: 一律切片の AUC 最大変化量 {d:.2e}（単調変換なので 0 のはず）")
    print(f"      症例ごと補正の AUC 最大変化量 {ds:.2e}"
          f"（症例ごとに移動量が違うため単調ではなく、変わり得る）")

    if a.json:
        Path(a.json).write_text(json.dumps({"site_stats": st, "linear": res[False],
                                            "quadratic": res[True]}, indent=2, default=float))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
