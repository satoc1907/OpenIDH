# 環境引き継ぎ（DGX Spark → Mac mini）

**作成** 2026-09-28 ／ 対象：次にこのリポジトリを触る人・AI

リポジトリの中身は `git pull` すれば揃う。この文書は **リポジトリを読んでも分からないこと**だけを書く。
プロジェクトの中身は `results/domain_shift_summary.md`、現況は `STATUS.md` を先に読むこと。

---

## 0. 移行前に必ずやること（これを落とすと成果物が消える）

この環境には GitHub の認証情報が無く、**push は人が実行する必要がある**。実験 A/B/C の全成果物が未 push で残っている可能性が高い。

```
git status -sb
```

`ahead N` が出たら、環境を落とす前に：

```
git push origin main
```

`runs/` 配下の npz・CSV は **gitignore されているので push では運ばれない**（下記 §3）。

---

## 1. 3 つのマシンの役割

| マシン | 役割 | 学習・推論 | 解析 |
|---|---|---|---|
| **Mac mini**（移行先） | 解析・執筆・git | **不可**（§2） | 可 |
| **raiden**（理研 AIP, UGE/qsub, x86_64 + A100） | **チェックポイントの保管場所**。学習・推論はここ | 可 | 可 |
| **理究 RIKYU**（R-CCS, Slurm/sbatch, aarch64 + GB200） | 予備の計算機。Step 1/2 検証済み | 可 | 可 |

- **`model_best.pt`（27 ラン分）は raiden にしかない**。ローカルにあるのは `smoke_foldA_s0` と `pm_test` だけ
- したがって「チェックポイントを読む処理」はすべて raiden（または理究）でジョブとして回し、結果ファイルを持ち帰る運用
- 両システムのジョブの書き方・コマンド対応表は `markdown/HPC_GUIDE_raiden_rikyu.md`

## 2. Mac では torch が入らない（重要）

`uv.lock` は torch を CUDA 専用インデックス（`pytorch-cu128`）に固定している。ロックに入っている wheel は
**manylinux (aarch64 / x86_64) と win_amd64 のみで、macOS のものは 0 個**。したがって Mac で `uv sync --frozen`
しても torch は入らない。

**対応済み**：解析パスは torch を使わないようにした（commit `037ed27`）。

```
uv sync --frozen --no-install-package torch --no-install-package torchvision
```

これで numpy / pandas / scipy / scikit-learn / matplotlib / papermill / jupyter が入り、以下がすべて動く：

- `notebooks/05_shift_decomposition.ipynb`（実験 A の解析）
- `scripts/head_ablation.py`、`scripts/evidence_scale_eval.py`、`scripts/ood_score_eval.py`
- `scripts/aggregate_results.py`、`scripts/age_prior_eval.py`、`scripts/grade_prior_eval.py`、`scripts/build_report.py`

動かないもの（torch が要る＝raiden / 理究で回す）：
`scripts/run_fold.py`（学習）、`scripts/calibrate_runs.py`、`scripts/dump_features.py`、`scripts/modality_drop_eval.py`、`tests/test_invariants.py`

テストは解析用のものだけなら Mac で通る：

```
uv run pytest tests/test_shift_decomp.py -q
```

（35 本のうち `test_invariants.py` はモデルを構築するので torch が要る。`tests/test_shift_decomp.py` には
「torch が import できない状況でも解析モジュールが読める」ことを確かめる回帰テストが入っている）

**もし Mac でもモデルを動かしたくなったら**：`pyproject.toml` の `[tool.uv.sources]` に
`marker = "sys_platform == 'linux'"` を足し、macOS では PyPI の torch（MPS 対応）を引く形に `uv lock` し直す。
ただし**ロックが変わるので raiden / 理究でも再検証が必要**。現状は「Mac は解析専用」で困っていない。

## 3. データとファイルの所在

| もの | 場所 | git | 備考 |
|---|---|---|---|
| コード・設定・split CSV・メタデータ | リポジトリ | ○ | split は seed=42 で凍結、再生成禁止 |
| 画像（`*.nii.gz`、約 46GB） | 各マシンにローカル配置 | × | Mac の配置は `configs/paths.mac.yaml` を参照（**未検証**、§4） |
| `model_best.pt`（27 ラン） | **raiden のみ** | × | 再学習は数時間規模 |
| `runs/*/predictions*.csv` | 各所 | ○（force-add 済み） | 実験 B/C はこれだけで再現できる |
| `runs/*/features_*.npz`, `head_weights.npz` | Spark / raiden | × | **実験 A の再解析に必須**。約 47MB。移行前に Mac へコピーすること |
| `runs/*/predictions_drop-*.csv` | Spark / raiden | × | 実験 B の真の欠損版。約 3MB |
| 結果 JSON・報告書・図 | `results/` | ○ | |

**移行時に忘れずコピーするもの**（git では運ばれない）：

```
rsync -av --include='*/' --include='features_*' --include='head_weights.npz' --include='predictions_drop-*' --exclude='*' <現環境>:<repo>/runs/ <Mac>/<repo>/runs/
```

これを持っていけば、Mac で実験 A/B/C の解析を**推論なしで全部再実行できる**。持っていかない場合は raiden で
`qsub qsub/dump_features.sh` と `qsub qsub/modality_drop.sh` を回せば再生成できる（各 15〜20 分）。

## 4. パス設定

環境ごとに yaml 1 枚で切り替える。環境変数 `OPENIDH_PATHS` か `--paths-file` で指定する。

| ファイル | 対象 | 状態 |
|---|---|---|
| `configs/paths.local.yaml` | 汎用（`root: AUTO` でリポジトリ root を自動検出） | 検証済み |
| `configs/paths.hpc.yaml` | raiden | 検証済み |
| `configs/paths.rikyu.yaml` | 理究 | 検証済み |
| `configs/paths.mac.yaml` | Mac mini | **未検証** — 下記で確認すること |

Mac で最初に走らせる確認：

```
OPENIDH_PATHS=configs/paths.mac.yaml uv run python -c "from openidh_model.utils.paths import load_paths; print(load_paths().as_dict())"
```

`data_dir` が実際の画像置き場と違えばその 1 行を直す。画像を Mac に置かない（解析専用にする）なら
`data_dir` は使われないので放置してよい。

## 5. この環境（Spark）特有だったこと — Mac では変わる点

- `/readonly/OpenIDH` に、ユーザのターミナル側リポジトリが読み取り専用でマウントされていた。ファイルの受け渡しはここ経由だった。**Mac では同じ仕組みは無い**（同一マシン上になるはず）
- push できないのはこの環境の制約。Mac では普通に push できる想定
- `python3` が無く `uv run` のみ。Mac では `python3` があるが、**プロジェクトの実行は `uv run` に統一**すること
- aarch64 Linux だったため、理究（同じく aarch64 Linux）向けの wheel 検証にこの環境が使えた。**Mac は macOS なのでその役目は果たせない**（理究向けの offline 手順が必要になったら別途用意が要る）

## 6. 作業の進め方（人の好み）

- **報告は Discord、日本語**。ターミナルの出力は相手に届かないので、伝えたいことは必ず Discord に流す
- **ユーザが実行するコマンドは ``` のコードブロックに、1 コマンド 1 ブロック、説明文をブロック内に混ぜない**
  （混ぜたことで実際に事故が複数回起きている）
- 節目ごとに commit し、Discord で報告して、次に進む前に判断を仰ぐ
- **数値を捏造しない**。ファイルが無いときは「無い」と報告する
- notebook は実験記録。ロジックは `openidh_model/` に置き、notebook からは呼ぶだけにする
- 依頼された作業の対象外のファイルは変更しない

## 7. 最初の 10 分でやるとよいこと（移行直後）

```
git log --oneline -15
```

```
cat STATUS.md
```

```
ls runs/*/features_test.npz | wc -l
```

（13 と出れば実験 A の再解析が可能。0 なら §3 のコピーか raiden での再生成が要る）

```
uv run pytest tests/test_shift_decomp.py -q
```

（35 本中の解析テストが通れば、torch 無しの解析環境として使える）

```
uv run papermill notebooks/05_shift_decomposition.ipynb /tmp/check.ipynb
```

（実験 A の全解析が再実行できれば移行完了。`results/shift_decomposition.md` が更新される）
