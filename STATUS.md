---
updated: 2026-09-28T17:05:00+09:00
state: review
waiting_on: me
phase: dev
---

# OpenIDH — 現況

## 現状サマリ

受領した仕様どおり**出力整合（output matching）**による切片推定を実装・検証した。全例で LOSO-A/B の ECE が oracle 並み（0.038 / 0.042）まで下がる。
N 例運用では LOSO-B は N=20 から安全、LOSO-A は N=100（許容 0.01 なら 30）。vendor は切片では直らない。
結果（`results/output_matching.md`）の判定と、レポート本文への反映可否が人の判断待ち。

## 進行中

- 出力整合の結果の判定 `@me` — `results/output_matching.md`、図 `results/figures/output_matching_n.png`

## ブロッカー・要判断

- **出力整合の判定とレポート反映** `@me` — 特に次の 3 点：
  (a) 「害」の定義。仕様どおりの「わずかでも悪化」だと、ずれが小さいユニット（field / vendor / 分布内）はどの N でも 10% を切らない（雑音で 50% 前後に張り付く）。補足として「0.01 超の悪化」も併記した
  (b) vendor は π̂_cov が 30.9%、真値は 35.0% と係留点がずれる上、全例ラベルの切片でも ECE 0.086（補正なし 0.085）。切片の問題ではない
  (c) field の比方式が逆符号（b = −0.33）。π(年齢, 疑い) が val 上で 33.1% と、観測 28.2% からずれているため
- **Mac では torch が入らない** `@me` — `uv.lock` が CUDA 専用 wheel を指しているため。
  解析は torch 不要にしたので影響しないが、学習・推論は raiden / 理究で行う前提。詳細は `markdown/HANDOVER.md`
  この Mac の実体は Docker（Linux aarch64）なので、HANDOVER の `--no-install-package torch torchvision` だけでは CUDA ライブラリ（数 GB）を落としにいく。
  nvidia-* / triton / cuda-* も除外が必要。また `tests/test_shift_decomp.py` の 1 本（softplus 比較）は torch を import するので torch 無しでは 8/9 になる

## 完了

- Stage 0 / A / B（設計・実装・27 ラン学習）。報告書 `results/stageb-report.{html,md}`
- 較正まわり：温度スケーリング、prior オラクル、EM（失敗と判明）、年齢事前確率、grade 事前確率、CBTRUS 外部事前確率
- 学習率分割実験、evidence 重みスイープ
- **実験 A**：施設間シフトの特徴空間分解 → `results/shift_decomposition_report.md`
- **実験 B**：弱いヘッドの除去 / シーケンス欠損 → `results/head_ablation_report.md`
- **実験 C**：症例単位 OOD スコア、evidence 定数縮小（どちらも否定的結果）
- **総まとめ**（外部相談用）→ `results/domain_shift_summary.md`
- 理究（RIKEN R-CCS, Slurm）の環境構築と Step 1/2 検証 → `slurm/`, `markdown/HPC_GUIDE_raiden_rikyu.md`
- 解析パスの torch 依存を除去（Mac 移行対応）
- 未 push の commit を push 済み（main = origin/main を 2026-09-28 に確認）
- Mac での移行確認：解析テスト 8/9 通過、実験 B head-off 540 行・実験 C-2 が committed の JSON と完全一致
- 計算環境の移行（Spark → Mac mini）。解析はスパコンで実行する運用のため、features 類の Mac への転送は不要と確認
- 出力整合の仕様を受領（2026-09-28、Discord 経由）
- **出力整合**（ラベル無し切片推定）の実装と 3 評価 → `openidh_model/eval/output_matching.py`、`notebooks/06_output_matching.ipynb`、`results/output_matching.{md,json}`

## 次にやるべきこと（AI 提案）

次のセッションの人が最初に読む前提で書く。まず `results/output_matching.md` と `results/domain_shift_summary.md`。

1. **出力整合の判定を受けて、処方箋の階段とレポート本文を更新する**（判定後）
   実測の階段（ECE、LOSO-A / LOSO-B）：補正なし 0.082 / 0.162 → 年齢のみ比方式 0.074 / 0.105 → 年齢+疑い比方式 0.070 / 0.064
   → **年齢+疑い出力整合 0.038 / 0.042** → 全例ラベル 0.042 / 0.045。出力整合は「ラベル 20 例」より上、全例ラベルとほぼ同じ段に入る。
2. **vendor の扱い**：切片では直らないことが分かったので、残差は形の歪み（傾き・ビン別のずれ）として別に調べる。
   あるいは先の案 (2)（T2/FLAIR の輝度拡張、再学習あり）の対象にする。
3. 旧案 (1)「装置メタデータから撮像由来の切片を推定」は、出力整合が撮像由来の分も一括で吸収する（LOSO-B で +0.35、field で +0.53）ので、優先度が下がった。
