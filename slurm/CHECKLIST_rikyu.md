# rikyu 試走チェックリスト（Step 1 → Step 2）

1 手ずつ実行して出力を貼ってください。各手順に「期待される出力」と「違ったとき」を書いてあります。
ログインノード（`c001` など）は無料、`sbatch` で投げたものだけ課金（300 円/GPU 時間）。

前提（2026-09-22 確認済み）：aarch64 / Ubuntu 24.04 / glibc 2.39 / driver 580 (CUDA 13.0) /
pypi・pytorch・HF・astral すべて HTTP 200 / コンテナ Python 3.12.3 / home 26MB 使用。
方針：**torch は uv.lock のまま（2.11.0+cu128, aarch64 wheel, sm_100 入り）**。コンテナは fallback。

---

## 0. repo を最新にする

```
cd ~/OpenIDH && git pull && git log --oneline -1 && ls slurm configs/paths.rikyu.yaml && mkdir -p slurm/logs
```
期待：`slurm/` に `env.sh prepare_env.sh tests.sbatch smoke.sbatch CHECKLIST_rikyu.md inventory_devbox.tsv`。

## 1. データ転送の照合（ログインノード、python3 のみ）

```
python3 scripts/data_inventory.py inventory /data1/rkp00078/satoc/Glioma/images/v1.0.0 -o slurm/inventory_rikyu.tsv
python3 scripts/data_inventory.py compare slurm/inventory_devbox.tsv slurm/inventory_rikyu.tsv --files-from slurm/missing.txt
```
期待（devbox の値）：
```
cohort        subj_dirs   files  nii.gz       GB
ucsf-pdgm           495    4950    4455    20.73
upenn-gbm           611    4277    3666    14.45
utsw-glioma         626    4377    3750    13.88
_global               1      11       0     0.00
TOTAL                     13615            49.06
missing in inventory_rikyu.tsv: 0   extra ...: 0   size mismatch: 0
```
- `missing`/`size mismatch` が 0 でなければ、Mac で
  `rsync -av --files-from=slurm/missing.txt ~/Desktop/OpenIDH/data/openidh/preprocessed/v1.0.0/ rikyu:/data1/rkp00078/satoc/Glioma/images/v1.0.0/`
  （`missing.txt` は rikyu から Mac にコピーしてから）。`extra` は Mac に無いだけなので無視可。
- HDD Lustre なので 1 万ファイルの stat に数分かかることがあります。
- 注：`_global` の差分は split CSV などの repo 由来ファイル。ジョブは repo 側の `_global/splits` を読むので無視可。

## 2. 環境構築（ログインノード、初回 10〜20 分、~7GB を /data1 に）

```
bash slurm/prepare_env.sh 2>&1 | tee slurm/logs/prepare_env.log
```
期待（要点）：
```
[prepare] https://astral.sh -> 200        （4 行とも 200）
uv 0.x.x
Installed Python 3.13.x ...
[prepare] uv sync --frozen (python 3.13)
Installed 80-ish packages ...
[prepare] downloading weights from HF (once)   saved: .../weights/vit_small_patch16_224.augreg_in21k.safetensors
  python 3.13.x  aarch64
  torch 2.11.0+cu128  cuda-build 12.8  timm 1.0.x
  data_dir      /data1/rkp00078/satoc/Glioma/images/v1.0.0  ok   （全行 ok）
  ucsf-pdgm     495 subject dirs / upenn-gbm 611 / utsw-glioma 626
33 passed
[prepare] done — next: sbatch slurm/tests.sbatch
```
違ったとき：
- `no wheels for linux_aarch64` / torch の解決失敗 → **止めて報告**（B 案＝コンテナ torch に切替）
- HF の DL が失敗 → Mac の `weights/` を `rsync -av weights/ rikyu:~/OpenIDH/weights/` で持ち込み、再実行
- `MISSING` のパス → `configs/paths.rikyu.yaml` の `root`/`data_root` を確認
- 疎通 000 → `OFFLINE=1` 方式（devbox から uv キャッシュを rsync）に切替、手順は `prepare_env.sh` 冒頭のコメント

## 3. Step 1：GPU ノードでの環境テスト（~3 分、~15 円）

```
sbatch slurm/tests.sbatch
squeue --me                      # PD→R→消える
tail -40 slurm/logs/openidh_tests.<JOBID>.out
```
期待：
```
[node] host=... gpus=1 cpus=36
GPU 0: NVIDIA GB200 (UUID: ...)
[node] /dev/shm: tmpfs  ...G ...          ← ★計算ノード側の値。raiden は 1G だった
[node] ulimit -n 1048576 ...
  device      : NVIDIA GB200
  capability  : (10, 0)
  arch list   : [..., 'sm_100', ...]
  matmul 4096^2 x10: 0.0x s
  tf32 matmul : False  cudnn tf32: True
  params      : 44224396
  forward     : {'T1': (2, 2), ..., 'tabular': (2, 2)}
33 passed
[tests] done
```
違ったとき：
- `cuda not available` → `USE_CONTAINER=1 sbatch --export=ALL,USE_CONTAINER=1 slurm/tests.sbatch` で同じ venv をコンテナ内から実行して切り分け。それでも駄目なら B 案
- `sm_100` が無い → wheel が違う。`uv run python -c "import torch;print(torch.__version__)"` を貼る
- `/dev/shm` が数 GB 以下 → Step 2 以降 `train.num_workers`/`prefetch_factor` を落とす（報告してください）

## 4. Step 2：smoke 学習（~2 分 GPU、~10 円）

```
sbatch slurm/smoke.sbatch
tail -60 slurm/logs/openidh_smoke.<JOBID>.out
```
期待：
```
[smoke] 50 subject dirs to stage
[stage] 1.5G staged to /tmp/<JOBID>/openidh in <10 s
[stage] OPENIDH_PATHS -> /tmp/<JOBID>/openidh/paths.yaml
=== fold: splits_loso_foldA.csv fold_id=None seed=0 device=cuda ===
train=20 val=10 age_mean=49.5 age_std=16.1
epoch 0: train_loss=... val_nll=...
epoch 1: ...
TEST metrics: {...}
saved -> runs/smoke_rikyu_s0
[smoke] result vs runs/smoke_foldA_s0 (CPU reference):
  n_train      new       20  ref       20  ok
  n_val        new       10  ref       10  ok
  n_test       new       20  ref       20  ok
  best_epoch   new        1  ref        1  ok     ← 0 でも可（2 epoch の smoke なので）
  best_val_nll new 0.6xxx  ref 0.6789        ← 1e-2 程度のずれは TF32/GPU 由来で正常
  test nll / auc / median_S                   ← 同じオーダーなら OK
[smoke] done -> runs/smoke_rikyu_s0
```
違ったとき：
- `stage_data` で rsync が失敗 → `-B /data1` 相当の問題はホスト実行では起きないはず。`ls /data1/rkp00078/satoc/Glioma/images/v1.0.0/utsw-glioma/BT0003` を貼る
- `train=0` や症例数が違う → 照合（手順 1）に戻る
- `unable to allocate shared memory` → `/dev/shm` の値と合わせて報告

## 5. 通ったら（任意）：特徴ダンプの smoke 版

```
sbatch --export=ALL --wrap='source slurm/env.sh; node_report; L=runs/smoke_rikyu_s0/staged_subjects.txt; stage_data $L; export OPENIDH_PATHS=$STAGE_PATHS; run_py python scripts/dump_features.py --runs smoke_rikyu_s0 --include-all --device cuda' --account=rkp00062 --gpus=1 --time=00:10:00 --job-name=openidh_feat_smoke --output=slurm/logs/%x.%j.out
```
期待：`head recompute rel.err=1e-3 前後 (rows swapped: 0.5〜2)` と `(match)`。

## 6. 報告してほしいもの

- 手順 1 の summary と compare の 2 行
- 手順 3 のログ全文（`/dev/shm`・arch list が知りたい）
- 手順 4 の `[smoke] result vs ...` ブロック
- `du -sh ~ /data1/rkp00078/satoc/uv`（容量の実測）
