# RAIDEN と 理究（RIKYU）でジョブを書く — 汎用ガイド

OpenIDH で両システムを実際に使って確認した内容をまとめたもの（2026-09-22 時点）。
別プロジェクトでも、このファイルと `qsub/` `slurm/` のテンプレートを持ち込めばそのまま始められる。
「確認済み」と書いていない項目は一般的な UGE / Slurm の仕様で、施設固有の値は各自の資料で再確認すること。

---

## 0. 一枚で見る違い

| | **RAIDEN**（理研 AIP） | **理究 RIKYU**（理研 R-CCS） |
|---|---|---|
| スケジューラ | UGE（Univa Grid Engine）: `qsub` | Slurm: `sbatch` |
| CPU / GPU | x86_64 / A100 40GB（`gtn-container_g1`） | **aarch64 (Arm Grace)** / GB200（B200, sm_100） |
| OS（計算ノード） | NGC コンテナ（Ubuntu 22.04, Python 3.10） | Ubuntu 24.04 ホスト直（glibc 2.39）、Apptainer は任意 |
| コンテナ | `-jc gtn-container_g1 -ac d=nvcr-pytorch-2401`（必須） | `apptainer exec --nv -B /data1/<group> <sif>`（任意） |
| 外部ネット | proxy 必須 `http://10.1.10.1:8080/` | **proxy なし**、pypi / pytorch / HF / astral すべて到達可（確認済み） |
| ホーム | 大きい（fefs） | **50 GB 固定**（SSD Lustre） |
| グループ領域 | — | `/data1/<group>` 1 TB〜（HDD Lustre、メタデータ遅い） |
| ノードローカル | — | `/tmp` 1.5 TB/GPU NVMe（ジョブ終了で消える） |
| `/dev/shm` | **1 GB**（dataloader が落ちる） | 846 GB（気にしなくてよい） |
| `ulimit -n` | — | 131072 |
| TF32 | **コンテナが強制 ON**（`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`）→ 1e-3 の丸め差 | 既定 OFF |
| 課金 | — | 300 円/GPU 時間、確保時間に課金。`--account` 必須 |
| 使った torch | uv.lock の cu128 (x86_64) | uv.lock の cu128 (**aarch64 wheel に sm_100 入り**) — 同じ lock |

---

## 1. コマンド対応表

| やりたいこと | RAIDEN (UGE) | 理究 (Slurm) |
|---|---|---|
| 投入 | `qsub job.sh` | `sbatch job.sh` |
| 自分のジョブ一覧 | `qstat` | `squeue --me` |
| 詳細 | `qstat -j JOBID` | `scontrol show job JOBID` |
| 終了したジョブの記録 | `qacct -j JOBID` | `sacct -j JOBID` / `sacct -j JOBID --format=JobID,Elapsed,MaxRSS,State` |
| 削除 | `qdel JOBID` | `scancel JOBID` |
| 変数を渡す | `qsub -v A=1,B=2 job.sh` | `sbatch --export=ALL,A=1,B=2 job.sh` |
| ジョブ名 | `#$ -N name` | `#SBATCH --job-name=name` |
| stdout/stderr | `#$ -j y`（結合、`name.oJOBID` がカレントに） | `#SBATCH --output=logs/%x.%j.out`（ディレクトリは事前に作る） |
| 作業ディレクトリ | `#$ -cwd`（指定しないとホーム） | 既定で投入時の cwd（指定不要） |
| 環境変数の継承 | `#$ -V`（必要なら） | 既定で継承 |
| 時間上限 | `#$ -l h_rt=HH:MM:SS`（job class 依存） | `#SBATCH --time=HH:MM:SS`（**必須**、最大 96h） |
| GPU | `#$ -jc gtn-container_g1`（=1 GPU の job class） | `#SBATCH --gpus=N`（N ∈ {1,2,3,4,8,12,16}。`--gres` ではない） |
| コンテナ指定 | `#$ -ac d=nvcr-pytorch-2401` | 不要（スクリプト内で `apptainer exec`） |
| 課題 | — | `#SBATCH --account=<project>`（複数課題所属だと必須） |
| パーティション | job class で決まる | `gpu` のみ、`-p` 不要 |
| インライン投入 | — | `sbatch --wrap='cmd' --gpus=1 --time=00:10:00 -A <project>` |
| 1 確保内で並列 | — | `srun --gpus=1 ... &` を複数（`--gpus=4` で 4 本） |

Slurm の `%x` = ジョブ名、`%j` = JOBID。

---

## 2. スクリプトの骨格

### 2.1 RAIDEN（`qsub/train_one.sh` の要約）

```bash
#!/bin/bash
#$ -cwd
#$ -jc gtn-container_g1          # 1 GPU の job class
#$ -ac d=nvcr-pytorch-2401       # NGC コンテナ（x86_64、Python 3.10）
#$ -N myjob
#$ -j y

# ── コンテナ初期化（施設テンプレートそのまま）──
/usr/local/bin/nvidia_entrypoint.sh || true
. /fefs/opt/dgx/env_set/nvcr-pytorch-2401.sh

# ── 外部は proxy 経由のみ ──
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL HTTPS_PROXY=$MY_PROXY_URL FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL https_proxy=$MY_PROXY_URL ftp_proxy=$MY_PROXY_URL
export LDFLAGS=-L/usr/local/nvidia/lib64

# ── Python は uv（コンテナの 3.10 を使わない）──
unset PYTHONPATH PYTHONUSERBASE PREFIX
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export UV_LINK_MODE=copy                       # fefs はクロス FS の hardlink 不可
export UV_CACHE_DIR="${HOME}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${HOME}/.local/share/uv/python"

set -euo pipefail
cd "${REPO_DIR:-${HOME}/MyProject}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "[job] host=$(hostname) job=${JOB_ID}"; nvidia-smi -L
uv run --frozen python scripts/main.py --device cuda "$@"
```

RAIDEN で踏んだ罠：
- **`/dev/shm` 1 GB**：DataLoader の `num_workers × prefetch_factor × batch_size × 1 サンプルの bytes` が 0.8 GB を超えると "unable to allocate shared memory" で不定期に死ぬ。`prefetch_factor=1` にする。`file_system` 共有戦略でも同じ
- **TF32 が強制 ON**：CPU の再計算と 1e-3 相対でずれる。数値検算は相対許容で書く
- **stdout がブロックバッファ**：`PYTHONUNBUFFERED=1` が無いと 20 時間ジョブの進捗がログに出ない
- `qsub/job.sh` を **`qsub` を付けずに実行するとログインノードで走る**（コンテナも GPU も無い）
- ジョブログ `name.oJOBID` はカレント（リポ root）に落ちるので、うっかり commit しやすい → `.gitignore` に `*.o[0-9]*` を足すか `-o logs/` を使う

### 2.2 理究（`slurm/tests.sbatch` `slurm/smoke.sbatch` の要約）

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=rkp00062          # 複数課題所属なら必須
#SBATCH --gpus=1                    # 1,2,3,4,8,12,16 のみ
#SBATCH --time=00:20:00             # 必須
#SBATCH --output=slurm/logs/%x.%j.out   # mkdir -p slurm/logs を先に
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh                 # 下記
node_report                         # GPU / /dev/shm / ulimit / /tmp をログ冒頭に

LIST=...                            # 使う症例ディレクトリの一覧（1 行 1 dir）
stage_data "${LIST}"                # Lustre(HDD) → /tmp/$SLURM_JOB_ID に rsync、yaml を派生
export OPENIDH_PATHS="${STAGE_PATHS}"
run_py python scripts/main.py --device cuda "$@"
```

共通 `slurm/env.sh` の要点（コピーして使う）：

```bash
export GROUP=/data1/<group>/<you>
unset PYTHONPATH PYTHONUSERBASE PREFIX
export PYTHONNOUSERSITE=1                       # ~/.local を混ぜない
export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy                        # Lustre
export UV_CACHE_DIR="${GROUP}/uv/cache"         # ~6.5 GB → ホーム(50GB)に置かない
export UV_PYTHON_INSTALL_DIR="${GROUP}/uv/python"
export UV_PYTHON_PREFERENCE=only-managed        # /usr/bin/python3 を拾わない
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# venv はリポ内 ~/Proj/.venv（SSD 側、import が速い）。移したいときは UV_PROJECT_ENVIRONMENT
run_py() { if [ "${USE_CONTAINER:-0}" = 1 ]; then apptainer exec --nv -B /data1/<group> "$SIF" uv run --frozen "$@"; else uv run --frozen "$@"; fi; }
```

理究で踏んだ / 分かった点：
- **`--account` は必須**（複数課題所属のとき）。省略すると "Invalid account" で投入失敗
- **`--output` のディレクトリは Slurm が作らない**。`mkdir -p` を先に
- **HDD Lustre の初回 stat は遅い**（13k ファイルで数分。2 回目はキャッシュで 13 秒）。一覧・照合系は `nohup ... &` で
- **Lustre → /tmp の staging は ~80 MB/s**（1.4 GB = 17 秒、46 GB ≈ 10 分）。複数 fold を 1 ジョブで回すなら staging は 1 回
- ログインノードは無料・GPU なし。環境構築（`uv sync`、weights DL、CPU の pytest）はここで済ませる
- aarch64 なので **x86_64 の wheel / バイナリ / SIF は動かない**。ただし PyTorch の cu128 index には aarch64 wheel があり、sm_100 カーネル入り（`strings libtorch_cuda.so | grep sm_100` で確認可）
- Apptainer はグループ領域を自動マウントしない（`-B /data1/<group>`）。ホスト直で uv venv を使うなら不要
- Mac から rsync すると `.DS_Store` が全ディレクトリに付いてくる（無害）。`rsync --exclude .DS_Store`

---

## 3. 両方で共通にしておくと楽な設計

1. **パスは yaml 1 枚に閉じ込める**（`configs/paths.<system>.yaml`、`root` と `data_root` の 2 行だけ差し替え）。環境変数 `OPENIDH_PATHS` 相当で切替。コードは触らない
2. **依存は uv.lock 1 本**。両システム同じ lock（cu128）で動くことを確認済み。Python は `uv python install 3.13` で揃える
3. **計算ノードは完全オフライン前提**：`HF_HUB_OFFLINE=1`、weights は事前にファイルとして置く
4. **ジョブ冒頭に node_report**：`hostname`、`nvidia-smi -L`、`df -h /dev/shm`、`ulimit -n`、`uv --version`。未知のシステムでは最初のジョブでこれを見るだけで大半の問題が分かる
5. **推論だけの「答え合わせジョブ」を持っておく**：既存 checkpoint + 既知の predictions.csv と照合（1e-3 相対）。学習の smoke は float 順序で 1e-2 動くので数値検証には向かない
6. **既知の答えつき smoke 学習**（20 例 × 2 epoch）：パイプライン（データ → 学習 → 保存）の疎通確認用。`result.json` を git 追跡して参照に使う
7. **データ照合スクリプトは標準ライブラリだけで書く**（Mac / 各ログインノードの素の python3 で動く）。`.nii.gz` の本数と GB が一致していれば OK
8. **同期は git 経由**：ローカル → `git push` → 各システムで `git pull`。ジョブログ・checkpoint・npz は gitignore、json/csv の結果だけ `git add -f`

---

## 4. 新プロジェクトを始めるときの手順（両システム共通）

```
# ローカル
uv init / uv add ...; uv lock                  # lock を作る
git push

# ログインノード（両方）
git clone <repo> && cd <repo>
bash <qsub|slurm>/prepare_env.sh               # 疎通 → uv sync --frozen → weights → CPU テスト
# 最初のジョブ = 環境テスト（GPU が見える / モデルが forward できる / pytest）
qsub qsub/tests.sh      or     sbatch slurm/tests.sbatch
# 2 本目 = 既知の答えつき smoke
qsub qsub/smoke.sh      or     sbatch slurm/smoke.sbatch
```

チェックリストの実物：`slurm/CHECKLIST_rikyu.md`（期待出力と失敗時分岐つき）。RAIDEN 版は `qsub/prepare_env.sh` `qsub/train_one.sh` `qsub/calibrate.sh` の冒頭コメントを参照。
