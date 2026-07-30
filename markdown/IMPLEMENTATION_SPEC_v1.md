# OpenIDH Phase 1 実装仕様書 v1.0

> 作成日：2026年7月29日
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

---

## 1. 環境

| 項目 | 内容 |
|---|---|
| 実行環境 | NVIDIA GB200 NVL4（Blackwell GPU × 4）1 ノード |
| 並列方式 | **1 GPU × 4 ジョブの独立並列**（fold・seed 単位）。単一ジョブの DDP は使わない |
| フレームワーク | PyTorch、timm |
| 精度 | bf16 混合精度（Blackwell）。損失計算は fp32 |

モデルが小さく単一ジョブでの多 GPU 分散は通信オーバーヘッドで損をする。fold × seed が大量にあるので embarrassingly parallel が最適。

---

## 2. ディレクトリ構成

```
openidh/
├── configs/
│   ├── base.yaml                 # 共通設定
│   ├── model.yaml                # モデル構造
│   ├── train.yaml                # 学習設定
│   └── sweep/                    # ハイパラ探索用
├── openidh_model/
│   ├── data/
│   │   ├── dataset.py            # Dataset（VOI 抽出・体積算出）
│   │   ├── transforms.py         # augmentation
│   │   └── dropout.py            # modality dropout
│   ├── models/
│   │   ├── encoder.py            # ViT トランク（共有 / 独立）
│   │   ├── heads.py              # evidence ヘッド、テーブル MLP
│   │   ├── fusion.py             # evidence 融合・体積スケーリング
│   │   └── openidh.py            # 全体を束ねる nn.Module
│   ├── losses/
│   │   └── evidential.py         # データ項・正則化項・全体損失
│   ├── train/
│   │   ├── loop.py               # 学習ループ
│   │   ├── monitor.py            # 12 節のモニタリング
│   │   └── calibrate.py          # inner-val での事後較正
│   ├── eval/
│   │   ├── metrics.py            # AUC/AUPRC/ECE/NLL/coverage
│   │   └── design_checks.py      # 13 節の設計検証
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_fold.py
└── tests/
    └── test_invariants.py        # 12 節
```

---

## 3. データ入出力

### 3.1 入力

前処理済み NIfTI（`data/preprocessed/v1.0.0/{site}/{subject_id}/volumes/`）：

```
T1.nii.gz, T2.nii.gz, FLAIR.nii.gz, T1c.nii.gz, tumor_seg.nii.gz
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
統合:          (12, 224, 224)      # [T1×3, T2×3, FLAIR×3, T1c×3] の順で連結
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
class OpenIDH(nn.Module):
    def __init__(self, cfg):
        # 共有トランク（単独 4 シーケンス用）
        self.trunk_single = timm.create_model(
            'vit_small_patch16_224.augreg_in21k',
            pretrained=True, num_classes=0, in_chans=3)

        # 独立トランク（12ch 統合用）
        self.trunk_unified = timm.create_model(
            'vit_small_patch16_224.augreg_in21k',
            pretrained=True, num_classes=0, in_chans=12)

        # evidence ヘッド（シーケンスごとに独立）
        self.heads_single = nn.ModuleDict({
            m: EvidenceHead(384) for m in ['T1', 'T2', 'FLAIR', 'T1c']})
        self.head_unified = EvidenceHead(384)

        # テーブル分類器
        self.head_tabular = TabularHead(in_dim=2)   # age, sex のみ
```

`in_chans=12` の指定で timm が `adapt_input_conv` により RGB 重みを 4 回タイルして 1/4 にスケールする。手動での重み変換は不要。

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
           for m in ['T1', 'T2', 'FLAIR', 'T1c']}
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

```yaml
model:
  backbone: vit_small_patch16_224.augreg_in21k
  pretrained: true
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

data:
  split_file: splits_loso_foldA.csv
  n_slices: 3
  image_size: 224
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

## 15. 実装の進め方（推奨順序）

1. **データローダー**（3.2〜3.5）→ VOI 抽出と体積算出が正しいか、数例を可視化して確認
2. **損失関数**（6 節）→ 12 節の KL テストを先に通す。単体で完結するので最初にやると安全
3. **融合**（5 節）→ 12 節の融合テストを通す。ここが設計の心臓部
4. **モデル**（4 節）→ パラメータ数が 44,224,396 になるか確認
5. **学習ループ**（9 節）+ モニタリング（13 節）
6. **BT0001 相当の 1 fold で少数 epoch を回す** → モニタリング項目が想定通り動くか
7. 本格学習

各段階で止まって報告してほしい。特に 3 と 6 は設計の妥当性が初めて検証される箇所。

---

## 16. 不明点があれば

設計判断の理由は `MODEL_DESIGN_v1.md` に記載してある。それでも判断に迷う場合は、**独自判断で進めずに Discord で確認**すること。特に 0 節の 5 つの落とし穴に関わる箇所は、一見冗長に見えても設計上の必然性がある。
