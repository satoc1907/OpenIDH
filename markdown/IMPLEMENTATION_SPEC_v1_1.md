# OpenIDH Phase 1 実装仕様書 v1.1

> 作成日：2026年7月29日
> 改訂：2026年7月29日（v1.1：uv による環境構築、ローカル動作確認→スパコン本番の二段階構成、事前学習重みのローカル配置、Jupyter notebook による実験記録を追加）
> 設計根拠：`MODEL_DESIGN_v1.md` v1.5（数式の導出・設計判断の理由はそちらを参照）
> 本書は「何をどう実装するか」に限定する

---

## 0. 実装前に必ず読むこと

この設計には**間違えやすい箇所が 5 つ**ある。実装前にここだけは目を通してほしい。すべて 12 節のテストで検証する。

| # | 落とし穴 | 正しい実装 |
|---|---|---|
| 1 | 融合で $\alpha$ を足してしまう | **evidence を足してから +1**。$\alpha$ を足すと全欠損時に Beta(6,6) になり、無情報状態が表現できない |
| 2 | 融合 evidence を $|A|$ で割ってしまう | **割らない**。割ると欠損しても $S$ が減らず、設計の根幹が消える。補助損失は逆に**割る** |
| 3 | 腫瘍体積を分類器の入力に入れてしまう | **絶対に入れない**。入れるとネットワークが $s(V)$ を相殺する挙動を学習し、スケーリングが無効化される |
| 4 | 体積スケーリングを補助損失の後にかける | **スケーリング後の evidence で補助損失を計算する** |
| 5 | KL を digamma で実装する | 二値分類では **`log(w) - (w-1)/w`** で厳密に等しい。digamma 不要、float32 で十分 |
| 6 | 事前学習重みを HuggingFace から都度取得する | **ローカルに保存して読む**（1.4 節）。スパコンはオフライン想定 |
| 7 | コードに絶対パスを書く | **`paths.*.yaml` に閉じ込める**（1.2 節）。スパコン移行時に書き換えるのは 1 行のみ |

---

## 1. 実行環境と二段階構成

### 1.1 基本方針：ローカル動作確認 → スパコン本番

**同一のコードを、config のパスだけ差し替えて両環境で動かす。** コードに環境依存の分岐を書かない。

| | Stage A：ローカル | Stage B：スパコン |
|---|---|---|
| 目的 | 動作確認・デバッグ | 本番学習・推論 |
| マシン | Mac mini | GB200 NVL4 ノード |
| データ | 少数症例（20 例程度）| 全 1,635 例 |
| epoch | 2–3 | 200 |
| fold | 1 つだけ | 全 9 fold × 複数 seed |
| デバイス | CPU / MPS | CUDA（bf16）|
| 実行形式 | Jupyter notebook（対話）| papermill でバッチ実行 |
| ネットワーク | あり | **なしと想定**（重み・パッケージは事前配置）|

Stage A で 12 節のテストとモニタリングが全て通ることを確認してから Stage B に移る。

### 1.2 パスの抽象化

環境差は `configs/paths.yaml` の 1 ファイルに閉じ込める。**コード内に絶対パスを書かない。**

```yaml
# configs/paths.local.yaml
root: /Users/satoc/Desktop/OpenIDH
data_dir: ${root}/data/preprocessed/v1.0.0
splits_dir: ${data_dir}/_global/splits
weights_dir: ${root}/weights
output_dir: ${root}/runs
```

```yaml
# configs/paths.hpc.yaml
root: /path/to/scratch/openidh      # ← ここだけ書き換える
data_dir: ${root}/data/preprocessed/v1.0.0
splits_dir: ${data_dir}/_global/splits
weights_dir: ${root}/weights
output_dir: ${root}/runs
```

環境変数 `OPENIDH_PATHS` でどちらを読むか切り替える：

```bash
export OPENIDH_PATHS=configs/paths.hpc.yaml
```

**受け入れ条件**：Stage B への移行時に書き換えるのは `paths.hpc.yaml` の `root` 一行のみ。それ以外のコード・config を触る必要があってはならない。

### 1.3 環境構築（uv）

```bash
uv init openidh
cd openidh
uv add torch torchvision timm numpy nibabel pandas scikit-learn \
       pyyaml matplotlib jupyter papermill safetensors
uv sync
```

実行は全て `uv run` 経由：

```bash
uv run python scripts/train.py --config configs/train.yaml
uv run papermill notebooks/train.ipynb runs/xxx/train_executed.ipynb
```

**スパコン側の注意**：

- GB200（Blackwell）は CUDA 12.8 以降が必要。`pyproject.toml` で torch のインデックスを明示的に指定すること
- `uv.lock` をリポジトリにコミットし、両環境で同一バージョンを保証する
- スパコンがオフラインの場合、`uv export --format requirements-txt` で書き出したものを事前に wheel ごと持ち込むか、ログインノードで `uv sync` してから計算ノードへ持ち込む
- スパコンに uv が無ければ単一バイナリなのでユーザ領域にインストールできる

### 1.4 事前学習重みのローカル配置（必須）

**スパコンでは HuggingFace のキャッシュが機能しない前提で設計する。** 計算ノードがオフライン、`$HOME` の容量制限、キャッシュディレクトリの共有失敗など、失敗要因が多い。

**ローカルで一度だけ実行して重みを保存する**：

```python
# scripts/download_weights.py（ローカルで 1 回だけ実行）
import timm, torch
from safetensors.torch import save_file
from pathlib import Path

MODEL = 'vit_small_patch16_224.augreg_in21k'
out = Path('weights'); out.mkdir(exist_ok=True)

m = timm.create_model(MODEL, pretrained=True)   # num_classes はデフォルトのまま
save_file(m.state_dict(), out / f'{MODEL}.safetensors')
print('saved:', sum(p.numel() for p in m.parameters()))
```

`num_classes=0` を指定せずに保存すること。ヘッドを含む生の state_dict を残しておけば、後から柔軟に読める。

**両環境での読み込み（同一コード）**：

```python
def build_trunk(model_name, in_chans, weights_dir):
    ckpt = Path(weights_dir) / f'{model_name}.safetensors'
    return timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        in_chans=in_chans,
        pretrained_cfg_overlay=dict(file=str(ckpt)),   # ← ローカルファイルを指定
    )
```

`pretrained_cfg_overlay=dict(file=...)` を使うと、timm はネットワークにアクセスせずローカルファイルから読み込みつつ、**`in_chans=12` のチャンネル適応（`adapt_input_conv`）とヘッド破棄も通常どおり適用される**。手動で重みを変換する必要はない。

**スパコン側で追加設定**：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**受け入れ条件**：ネットワークを遮断した状態でモデルが構築できること。Stage A の段階で `HF_HUB_OFFLINE=1` を設定して検証する。

### 1.5 計算資源

| 項目 | 内容 |
|---|---|
| ノード | NVIDIA GB200 NVL4（Blackwell GPU × 4）1 ノード |
| 並列方式 | **1 GPU × 4 ジョブの独立並列**（fold・seed 単位）。単一ジョブの DDP は使わない |
| 精度 | bf16 混合精度。損失計算は fp32 |

モデルが小さく単一ジョブでの多 GPU 分散は通信オーバーヘッドで損をする。fold × seed が大量にあるので embarrassingly parallel が最適。

ジョブ投入スクリプトのテンプレートを `scripts/hpc/` に用意し、スケジューラ（SLURM 等）の記述は 1 ファイルに集約する。

---

## 2. ディレクトリ構成

```
openidh/
├── pyproject.toml                # uv 管理
├── uv.lock                       # ★ 両環境で同一バージョンを保証
├── configs/
│   ├── paths.local.yaml          # ★ ローカルのパス
│   ├── paths.hpc.yaml            # ★ スパコンのパス（root 一行だけ書き換える）
│   ├── base.yaml
│   ├── model.yaml
│   ├── train.yaml                # 本番設定
│   ├── train.smoke.yaml          # ★ Stage A 用（少データ・2 epoch）
│   └── sweep/
├── weights/                      # ★ 事前学習重みをここに配置（git 管理外）
│   └── vit_small_patch16_224.augreg_in21k.safetensors
├── notebooks/
│   ├── 00_setup_check.ipynb      # ★ 環境・重み・オフライン動作の確認
│   ├── 01_data_check.ipynb       # ★ VOI 抽出・体積算出の可視化確認
│   ├── 02_smoke_train.ipynb      # ★ Stage A：少データ学習
│   ├── 03_train.ipynb            # ★ Stage B：本番学習（papermill で実行）
│   └── 04_analyze.ipynb          # ★ 結果分析・設計検証
├── openidh_model/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── dropout.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── heads.py
│   │   ├── fusion.py
│   │   └── openidh.py
│   ├── losses/
│   │   └── evidential.py
│   ├── train/
│   │   ├── loop.py
│   │   ├── monitor.py
│   │   └── calibrate.py
│   ├── eval/
│   │   ├── metrics.py
│   │   └── design_checks.py
│   └── utils/
│       └── paths.py              # ★ OPENIDH_PATHS を読んで解決
├── scripts/
│   ├── download_weights.py       # ★ ローカルで 1 回だけ実行
│   ├── train.py
│   ├── evaluate.py
│   ├── run_fold.py
│   └── hpc/
│       ├── submit.sh             # ★ スケジューラ記述はここに集約
│       └── run_all_folds.sh      # ★ 9 fold × seed を 4 GPU に配分
└── tests/
    └── test_invariants.py
```

**git 管理から除外**：`weights/`、`runs/`、`data/`

---

## 3. データ入出力

### 3.1 入力

前処理済み NIfTI（`data/preprocessed/v1.0.0/{site}/{subject_id}/volumes/`）：

```
T1.nii.gz, T2.nii.gz, FLAIR.nii.gz, T1GD.nii.gz, tumor_seg.nii.gz
```

全て 256×256×N、1mm³ 等方、脳マスク内 z-score、脳外は厳密に 0。

臨床情報は `_global/` の metadata から `subject_id` で結合：`age`, `sex`, `idh`（0/1）。

分割定義は `data/preprocessed/v1.0.0/_global/splits/*.csv`（seed=42 で固定済み、**変更禁止**）。

### 3.2 VOI スライス抽出

```python
def select_slices(seg: np.ndarray) -> list[int]:
    """腫瘍面積が最大の axial スライス ±1 の 3 枚を返す"""
    area = (seg > 0).sum(axis=(0, 1))       # z 軸ごとの腫瘍面積
    z = int(area.argmax())
    z = np.clip(z, 1, seg.shape[2] - 2)     # 端でのはみ出し防止
    return [z - 1, z, z + 1]
```

**同じ 3 枚を全シーケンスで使う**（シーケンスごとに選び直さない）。前処理で共登録済みなので同一の解剖位置を指す。

### 3.3 テンソル化

```
単独シーケンス: (3, 224, 224)      # 3 スライスを 3ch に
統合:          (12, 224, 224)      # [T1×3, T2×3, FLAIR×3, T1GD×3] の順で連結
```

- 256×256 → 224×224 は on-the-fly リサイズ（bilinear）
- **チャンネル順序を固定**：統合入力は上記の順で必ず並べる。順序が変わると学習済み重みが無意味になる

### 3.4 腫瘍体積

```python
volume = int((seg > 0).sum())   # whole tumor（label 1+2+4）
```

**この値はモデルの入力に渡さない**。`fusion.py` のスケーリングにのみ使う。Dataset の返り値では画像・ラベルとは別のキーに入れ、モデルの `forward` では画像経路に混入しないよう注意する。

### 3.5 欠損モダリティ

ファイルが存在しないシーケンスは `mask[modality] = False` とし、テンソルはゼロ埋め。

---

## 4. モデル

### 4.1 構成

```python
def build_trunk(in_chans, cfg):
    """ローカル重みから構築（ネットワーク非依存、1.4 節）"""
    ckpt = Path(cfg.paths.weights_dir) / f'{cfg.model.backbone}.safetensors'
    return timm.create_model(
        cfg.model.backbone,
        pretrained=True,
        num_classes=0,
        in_chans=in_chans,
        pretrained_cfg_overlay=dict(file=str(ckpt)),
    )

class OpenIDH(nn.Module):
    def __init__(self, cfg):
        # 共有トランク（単独 4 シーケンス用）
        self.trunk_single = build_trunk(in_chans=3, cfg=cfg)

        # 独立トランク（12ch 統合用）
        self.trunk_unified = build_trunk(in_chans=12, cfg=cfg)

        # evidence ヘッド（シーケンスごとに独立）
        self.heads_single = nn.ModuleDict({
            m: EvidenceHead(384) for m in ['T1', 'T2', 'FLAIR', 'T1GD']})
        self.head_unified = EvidenceHead(384)

        # テーブル分類器
        self.head_tabular = TabularHead(in_dim=2)   # age, sex のみ
```

`in_chans=12` の指定で timm が `adapt_input_conv` により RGB 重みを 4 回タイルして 1/4 にスケールする。`pretrained_cfg_overlay` でローカルファイルを指定してもこの適応処理は通常どおり働くため、手動での重み変換は不要。

### 4.2 evidence ヘッド

```python
class EvidenceHead(nn.Module):
    def __init__(self, dim):
        self.fc = nn.Linear(dim, 2)
    def forward(self, x):
        e = F.softplus(self.fc(x))
        return e.clamp(min=1e-6)        # e[:, 0] = e1, e[:, 1] = e0
```

**softplus を使う**（exp は発散しやすい）。

### 4.3 テーブル分類器

```python
class TabularHead(nn.Module):
    """入力は age, sex のみ。腫瘍体積は入れない。"""
    def __init__(self, in_dim=2, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))
    def forward(self, x):
        return F.softplus(self.net(x)).clamp(min=1e-6)
```

- `age` は学習データの平均・標準偏差で標準化（統計量は fold ごとに train から算出し保存）
- `sex` は 0/1
- **年齢・性別は modality dropout の対象外**（臨床現場で欠損が稀）

### 4.4 パラメータ数（検証用）

| 要素 | 期待値 |
|---|---:|
| 共有トランク | 21,665,664 |
| 独立トランク（12ch）| 22,550,400 |
| evidence ヘッド × 5 | 3,850 |
| テーブル MLP | 4,482 |
| **合計** | **44,224,396** |

実装後に `sum(p.numel() for p in model.parameters())` で照合すること。

---

## 5. 体積スケーリングと evidence 融合

### 5.1 スケーリング

```python
V_REF = 10_000      # voxel（10 cm³）
GAMMA = 0.5

def volume_scale(V: torch.Tensor) -> torch.Tensor:
    return torch.clamp((V / V_REF) ** GAMMA, max=1.0)
```

**画像分類器 5 つのみに適用。テーブル分類器には適用しない。**

### 5.2 融合

```python
def fuse(evidences: dict, active: dict, volume: torch.Tensor):
    """
    evidences: {name: (B, 2)} 生 evidence
    active:    {name: (B,)} bool、欠損していないか
    """
    s = volume_scale(volume).unsqueeze(-1)      # (B, 1)

    scaled = {}
    for name, e in evidences.items():
        e = e * s if name != 'tabular' else e   # 画像のみスケール
        scaled[name] = e * active[name].unsqueeze(-1)   # 欠損は 0

    # ★ evidence を足してから +1（α を足してはいけない）
    e_sum = sum(scaled.values())                # (B, 2)
    alpha = e_sum[:, 0] + 1.0
    beta  = e_sum[:, 1] + 1.0
    return alpha, beta, scaled
```

**`e_sum` を `|A|` で割らないこと。** 割ると欠損しても $S$ が減らず、設計の根幹（情報が少ないほど裾が広がる）が消える。

返り値の `scaled` は補助損失で使う（スケーリング後の値である点が重要）。

### 5.3 演算順序

```
1. 各分類器が生 evidence を出力
2. 体積スケーリング（画像のみ）
3. modality dropout で active を落とす
4. 補助損失をスケーリング後の evidence で計算
5. evidence を融合して α, β
6. 融合損失を計算
```

---

## 6. 損失

### 6.1 データ項

```python
def loss_data(alpha, beta, y):
    S = alpha + beta
    p = alpha / S
    bias = (y - p) ** 2
    var  = p * (1 - p) / (S + 1)
    return bias + var
```

### 6.2 正則化項（簡潔形）

二値分類では digamma が不要になる。導出は `MODEL_DESIGN_v1.md` 3.3.6 節。

```python
def loss_reg(e1, e0, y):
    """正解方向の evidence は罰さない"""
    e_wrong = torch.where(y > 0.5, e0, e1)
    w = e_wrong + 1.0
    return torch.log(w) - (w - 1.0) / w
```

`lgamma` も `digamma` も使わない。float32 で十分。

> **Phase 2 注意**：多クラス（Dirichlet）に拡張する際はこの簡潔形が使えず digamma が必要になる。その場合は float64 で計算すること。

### 6.3 全体

```python
def total_loss(alpha_f, beta_f, scaled, active, y, lam_reg_t, lam_aux):
    # 融合側
    l_fused_data = loss_data(alpha_f, beta_f, y)
    l_fused_reg  = loss_reg(alpha_f - 1, beta_f - 1, y)
    l_fused = l_fused_data + lam_reg_t * l_fused_reg

    # 補助側（★ こちらは |A| で割る）
    l_aux, n_active = 0.0, 0.0
    for name, e in scaled.items():
        a, b = e[:, 0] + 1, e[:, 1] + 1
        li = loss_data(a, b, y) + lam_reg_t * loss_reg(e[:, 0], e[:, 1], y)
        m = active[name].float()
        l_aux    += li * m
        n_active += m
    l_aux = l_aux / n_active.clamp(min=1)

    return (l_fused + lam_aux * l_aux).mean(), {
        'fused_data': l_fused_data.mean(), 'fused_reg': l_fused_reg.mean(),
        'aux': l_aux.mean()}
```

`fused_data` と `fused_reg` を**別々に返してログする**。融合側では正則化項が支配的になりやすく（設計書 3.1 節の計算例ではデータ項の 60 倍）、比が極端なら $\lambda$ の調整が必要。

### 6.4 アニーリング

```python
lam_reg_t = lam_max * min(1.0, epoch / T_anneal)     # T_anneal = 10〜20
```

初期から正則化が強いと evidence が立ち上がらない。

### 6.5 クラス不均衡

**デフォルトでは重み付けしない**（WT:Mut = 4.5:1）。較正を歪めるため。ablation の 1 項目として残す。

---

## 7. Modality dropout

```python
P_DROP = 0.2

def apply_dropout(mask: dict, training: bool) -> dict:
    """モダリティ単位で落とす。分類器単位ではない。"""
    if not training:
        return mask
    out = {m: mask[m] and (random.random() > P_DROP)
           for m in ['T1', 'T2', 'FLAIR', 'T1GD']}
    if not any(out.values()):                  # 最低 1 つは残す
        out[random.choice([m for m in mask if mask[m]])] = True
    return out
```

**連動関係**：モダリティ $m$ を落としたとき

- 単独分類器 $m$ → `active[m] = False`
- 統合分類器 → 該当 3 チャンネルをゼロ埋め。**統合分類器自体は active のまま**（他のシーケンスが残っているため）
- 全モダリティ欠損時のみ統合分類器も `active = False`

テーブル分類器は常に active。

---

## 8. Augmentation

ドメインシフト源はスキャナーと判明している（UTSW は 5 社・0.3–3T、UPenn は Siemens 99%・3T 88%）。したがって**スキャナー差の模倣**を重点化する。

| 種類 | 目的 | 優先度 |
|---|---|---|
| bias field 付与 | ベンダー間のコイル特性差 | 高 |
| ガウスノイズ（SNR 変動）| 磁場強度差 | 高 |
| 強度ヒストグラム変形（gamma 等）| ベンダー間のコントラスト差 | 高 |
| 解像度・スライス厚の擬似変動（blur → resize）| 撮像プロトコル差 | 中 |
| 左右反転 | 一般的な汎化 | 中 |
| 微小な回転・平行移動（±10°, ±5%）| 位置ずれ耐性 | 低 |

幾何変形は前処理で正規化済みのため優先度は低い。具体的なパラメータは inner-val で調整。

**重要**：augmentation は 4 シーケンスに**同一の幾何変換**を適用する（バラバラだと共登録が壊れる）。強度系はシーケンスごとに独立で構わない。

---

## 9. 学習設定

| 項目 | 値 |
|---|---|
| ファインチューニング | **full fine-tuning**（全層更新、凍結・LoRA なし）|
| optimizer | AdamW |
| 学習率 | 1e-4 前後から探索（事前学習済みなので小さめ）|
| weight decay | 0.05 |
| スケジューラ | cosine + warmup（3〜5 epoch）|
| バッチサイズ | 32 |
| epoch | 200（early stopping あり）|
| early stopping | inner-val の **NLL** で判定（AUC ではない）|
| 精度 | bf16 混合精度、損失は fp32 |

---

## 10. 分割の扱い

分割定義 CSV は**唯一の真実**として固定。動的に生成しない。

```
splits_loso_foldA.csv       主評価（test = UCSF）
splits_loso_foldB.csv       主評価（test = UTSW）
splits_vendor_philips.csv   補助（UTSW 内ベンダー間）
splits_field.csv            補助（UTSW 内 1.5T ↔ 3T）
splits_random_5fold.csv     比較用（楽観値）
```

- `split_role` が `train` / `val` / `test`
- **外側 test は最後に一度だけ評価**。ハイパラ選択・early stopping は inner-val のみ
- IDH-NA 症例は含まれていない（Phase 1 では完全除外）

### 事後較正

fold 間で base rate が異なる（Fold B は train 11.7% → test 28.4%）。**inner-val で temperature scaling をフィット**し、test に適用する。生の確率と較正後の両方で ECE を報告。

---

## 11. Config スキーマ

パスは `paths.*.yaml` に分離し（1.2 節）、以下は環境非依存。

```yaml
# configs/train.yaml（本番）
model:
  backbone: vit_small_patch16_224.augreg_in21k
  pretrained_from_local: true    # ★ weights_dir から読む（1.4 節）
  embed_dim: 384
  share_single_trunk: true       # 単独 4 シーケンスでトランク共有
  separate_heads: true           # シーケンスごとに別ヘッド（案 B）

fusion:
  v_ref: 10000
  gamma: 0.5
  normalize_by_active: false     # ★ 必ず false

loss:
  lambda_aux: 0.4
  lambda_reg_max: 0.1
  t_anneal: 15
  class_weight: null             # デフォルトは重み付けなし

train:
  finetune: full
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 0.05
  batch_size: 32
  epochs: 200
  early_stop_metric: nll
  early_stop_patience: 20
  modality_dropout_p: 0.2
  device: cuda
  amp_dtype: bfloat16

data:
  split_file: splits_loso_foldA.csv
  n_slices: 3
  image_size: 224
  subset_n: null                 # null = 全症例
```

```yaml
# configs/train.smoke.yaml（Stage A：ローカル動作確認）
# train.yaml からの差分のみ
train:
  epochs: 2
  batch_size: 4
  device: cpu                    # または mps
  amp_dtype: null
data:
  subset_n: 20                   # ★ 20 症例だけ使う
```

**Stage A と Stage B で差分はこの 2 ファイルだけ**。モデル構造・損失・融合の設定は共通。

---

## 11.5 Notebook 運用

実験ログは Jupyter notebook として残す。**実行済み notebook（出力付き）が実験記録そのもの**になる。

### ノートブックの役割

| ノートブック | 段階 | 内容 |
|---|---|---|
| `00_setup_check.ipynb` | 準備 | uv 環境、重みのローカル読み込み、**オフライン動作**（`HF_HUB_OFFLINE=1`）、パラメータ数 44,224,396 の確認 |
| `01_data_check.ipynb` | 準備 | VOI スライス選択と腫瘍体積の可視化確認。数例を overlay 表示して目視 |
| `02_smoke_train.ipynb` | Stage A | 20 例 × 2 epoch。12 節のテストと 13 節のモニタリングが動くことを確認 |
| `03_train.ipynb` | Stage B | 本番学習。papermill でパラメータを注入してバッチ実行 |
| `04_analyze.ipynb` | 事後 | 14 節の指標と設計検証。fold 横断の比較 |

### papermill によるバッチ実行

スパコンの計算ノードでは対話的に notebook を開けないため、papermill で実行して出力付き notebook を保存する。

```python
# 03_train.ipynb の先頭セルに "parameters" タグを付ける
config_path = "configs/train.yaml"
split_file  = "splits_loso_foldA.csv"
seed        = 0
output_dir  = "runs/default"
```

```bash
uv run papermill notebooks/03_train.ipynb \
  ${OUTPUT_DIR}/03_train_executed.ipynb \
  -p config_path configs/train.yaml \
  -p split_file splits_loso_foldA.csv \
  -p seed 0 \
  -p output_dir ${OUTPUT_DIR}
```

実行済み notebook は `runs/{run_id}/` に保存され、そのまま実験記録になる。

### 実装との分離

**ロジックは `openidh_model/` のモジュールに書き、notebook からは呼ぶだけにする。** notebook にロジックを直書きすると、Stage A と Stage B で乖離が生じ、再利用もテストもできなくなる。

```python
# notebook 内は原則これだけ
from openidh_model.train.loop import train_fold
result = train_fold(cfg, split_file=split_file, seed=seed)
```

---

## 12. 実装後に必ず通すテスト

`tests/test_invariants.py` に以下を実装する。**これが通らなければ設計が壊れている。**

```python
def test_all_missing_gives_uniform():
    """全分類器を落とすと Beta(1,1) に縮退する"""
    alpha, beta, _ = fuse(ev, active_all_false, vol)
    assert torch.allclose(alpha, torch.ones_like(alpha))
    assert torch.allclose(beta, torch.ones_like(beta))

def test_missing_reduces_S():
    """モダリティを落とすほど S が単調減少する"""
    S_full = fuse(ev, all_active, vol)[0] + fuse(...)[1]
    S_drop1 = ...
    S_drop3 = ...
    assert S_full > S_drop1 > S_drop3

def test_small_volume_reduces_S():
    """小体積で S が減る"""
    S_big = S_at(volume=100_000)
    S_small = S_at(volume=100)
    assert S_small < S_big

def test_volume_not_in_forward_inputs():
    """体積を変えても生 evidence（スケール前）は変わらない
    = 体積が分類器の入力に漏れていない"""
    e1 = model.raw_evidence(x, volume=100)
    e2 = model.raw_evidence(x, volume=100_000)
    assert torch.allclose(e1, e2)

def test_kl_matches_closed_form():
    """簡潔形が digamma 版と一致する"""
    for e0 in [0., 1., 4., 9., 100.]:
        simple = math.log(e0+1) - e0/(e0+1)
        full = kl_beta_uniform_digamma(1.0, e0+1)
        assert abs(simple - full) < 1e-9

def test_kl_reference_values():
    """既知の値と一致する"""
    assert abs(loss_reg_scalar(e0=1) - 0.1931472) < 1e-6
    assert abs(loss_reg_scalar(e0=4) - 0.8094379) < 1e-6
    assert abs(loss_reg_scalar(e0=9) - 1.4025851) < 1e-6

def test_param_count():
    assert sum(p.numel() for p in model.parameters()) == 44_224_396

def test_no_train_test_leak():
    """全分割で subject_id の重複がない"""
    for f in split_files:
        assert set(train_ids) & set(test_ids) == set()
```

---

## 13. 学習中のモニタリング

以下を epoch ごとに記録する。**これがないと設計が機能しているか分からない。**

| 項目 | 異常のサイン |
|---|---|
| 分類器ごとの $e_1, e_0$ 分布 | 特定分類器の evidence が 0 に張り付く（死んでいる）|
| $S_{fused}$ の分布（全揃い / 欠損で層別）| 全揃いと欠損で $S$ が変わらない → 融合の実装ミス |
| **正答例 vs 誤答例の $S$ 分布** | 2 群が重なる → 不確実性が無意味 |
| $S$ の中央値 | > 500 常態化 = 過信、< 5 常態化 = 学習不足。目標は 10〜120 |
| 分類器間の evidence バランス | テーブル分類器だけが支配的 → 画像エンコーダが機能していない |
| `fused_data` と `fused_reg` の比 | 正則化が支配的 → $\lambda$ 調整が必要 |

**特に注意**：年齢は IDH の非常に強い予測因子（3 サイトとも Mut ≈ 37-38 歳 / WT ≈ 62-63 歳）。**テーブル分類器だけが学習し、画像エンコーダが機能しない縮退**が起こりうる。画像分類器の evidence がテーブル分類器に比べて極端に小さければ、この失敗モードを疑う。

---

## 14. 評価

### 14.1 指標

| 指標 | 用途 |
|---|---|
| AUC | 識別性能（文献比較）|
| AUPRC | 不均衡下での性能（陽性率 18%）|
| ECE / reliability diagram | 較正 |
| NLL | 分布としての当てはまり |
| 90% credible interval coverage | 区間の妥当性 |

### 14.2 設計検証（最重要）

`eval/design_checks.py` に実装する。**AUC が出ても、これが成立しなければ設計は失敗。**

| 検証 | 期待 |
|---|---|
| モダリティを 1 つずつ落とす | $S$ が単調減少 |
| 腫瘍体積 vs $S$ | 小体積で低下 |
| LOSO の test site | in-site より $S$ 低下 |
| **予測誤り例 vs 正解例** | **誤り例の $S$ が低い** |

最後の項目が最重要。「間違えるときは自信がない」が成立しなければ、この不確実性は臨床的に無価値。

### 14.3 報告

**random split と LOSO の両方を報告し、その差を議論する。** 多くの IDH 予測論文は random split しか報告していない。「random split の AUC は SOTA より低いが、LOSO での低下が小さい」という対比がロバストネス研究としての貢献になる。

---

## 15. 実装の進め方

### Stage 0：環境準備

1. `uv init` + `uv add`（1.3 節）、`uv.lock` をコミット
2. `scripts/download_weights.py` をローカルで実行し、`weights/` に safetensors を保存
3. `openidh_model/utils/paths.py` と `configs/paths.local.yaml` を作る
4. **`00_setup_check.ipynb`**：`HF_HUB_OFFLINE=1` を設定した状態でモデルが構築でき、パラメータ数が 44,224,396 になることを確認

→ **ここで報告**。オフラインで構築できなければスパコンで動かない。

### Stage A：ローカル動作確認

5. **データローダー**（3 節）→ `01_data_check.ipynb` で VOI 抽出と体積算出を可視化確認
6. **損失関数**（6 節）→ 12 節の KL テストを先に通す。単体で完結するので最初にやると安全
7. **融合**（5 節）→ 12 節の融合テストを通す。**ここが設計の心臓部**

→ **ここで報告**。融合の実装が正しくないと設計全体が無意味になる。

8. **モデル**（4 節）→ パラメータ数を照合
9. **学習ループ**（9 節）+ モニタリング（13 節）
10. **`02_smoke_train.ipynb`**：`configs/train.smoke.yaml` で 20 例 × 2 epoch。12 節のテスト全通過と、13 節のモニタリング項目が想定通り出力されることを確認

→ **ここで報告**。Stage B に進む前の最終ゲート。

### Stage B：スパコン本番

11. `configs/paths.hpc.yaml` の `root` を書き換えるのみ。**他は一切変更しない**
12. データ・`weights/`・`uv.lock` をスパコンへ転送
13. `scripts/hpc/submit.sh` で 1 fold だけ短時間実行し、環境差がないことを確認
14. `scripts/hpc/run_all_folds.sh` で全 9 fold × seed を 4 GPU に配分して実行

**Stage B で「動かないので設定を変えた」が発生したら、それは Stage A の検証が不十分だったということ**。その場合は変更点を記録し、ローカルにも反映して再現性を保つこと。

---

## 16. 不明点があれば

設計判断の理由は `MODEL_DESIGN_v1.md` に記載してある。それでも判断に迷う場合は、**独自判断で進めずに Discord で確認**すること。特に 0 節の 5 つの落とし穴に関わる箇所は、一見冗長に見えても設計上の必然性がある。
