---
updated: 2026-09-28T09:30:00+09:00
state: review
waiting_on: me
phase: dev
---

# OpenIDH — 現況

## 現状サマリ

ドメインシフト調査（実験 A/B/C）が完了し、「分布外の較正誤差は確信度の水準ではなく**切片**の問題」と確定した。
次の一手の候補は 2 つに絞れており、いまは他の AI への相談結果を待つ判断待ち。
計算環境を DGX Spark から Mac mini へ移行中（解析は torch 無しで動くよう対応済み、学習・推論は raiden / 理究）。

## 進行中

- 他の AI へ `results/domain_shift_summary.md` を渡して相談 `@me`
- 計算環境の移行（Spark → Mac mini）`@me`

## ブロッカー・要判断

- **未 push の commit が 5 本ある** `@me` — 移行前に必ず `git push origin main`。
  この環境には GitHub 認証が無く、push は人が実行する必要がある（失うと実験 A/B/C の成果物ごと消える）
- **次の一手の選択** `@me` — 下記「次にやるべきこと」の (1) か (2)。相談結果が出てから決める
- **Mac では torch が入らない** `@me` — `uv.lock` が CUDA 専用 wheel を指しているため。
  解析は torch 不要にしたので影響しないが、学習・推論は raiden / 理究で行う前提。詳細は `markdown/HANDOVER.md`

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

## 次にやるべきこと（AI 提案）

次のセッションの人が最初に読む前提で書く。まず `results/domain_shift_summary.md` を読めば全体像が掴める。

1. **撮像由来の切片ずれを装置メタデータから推定する**（本命）
   実験 A が見つけた穴。既存の切片推定手段（年齢・grade・CBTRUS・少数ラベル）はすべて「症例構成由来」しか扱えず、
   撮像由来の成分（field で有病率不変のままロジット −0.33、LOSO-B で criterion −0.24〜−0.43σ）は誰も推定できていない。
   ベンダー・磁場強度は推論時に既知なので、原理的には補正可能。まず既存の分解表で回帰が立つか確認する（推論不要）。

2. **T2 / FLAIR 限定のエンコーダ側介入**（輝度のアフィン不変な拡張）
   実験 A の判定では介入対象は T2 と FLAIR に限定してよい（T1c は動いていない）。
   ただし効果の上限は「vendor で失った 2〜3 割の evidence 回復＋ criterion 0.3σ」で、AUC は大きく動かない見込み。
   **再学習が必要**なので raiden か理究でのジョブになる。

どちらに進むにせよ、`results/domain_shift_summary.md` 6 節に**想定される反論を 4 点**書いてある。
相談相手からの指摘があれば、そこに追記してから設計に入るのがよい。
