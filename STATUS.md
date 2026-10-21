---
updated: 2026-10-21T12:30:00+09:00
state: review
waiting_on: me
phase: writing
---

# OpenIDH — 現況

## 現状サマリ

研究全体を論文形式の草稿 v0.3 にまとめた（`results/manuscript_draft.md`）。最有望は「出力整合（年齢+疑い）＋有意性ゲート k=2.5, N_min=20」で、ポータル定義案も草稿 5 章に入れた。
草稿のレビューと、k・N_min の採否が人の判断待ち。

## 進行中

- 草稿 v0.3 のレビュー `@me` — `results/manuscript_draft.md`（データ・目的・実験・較正補正の比較・ポータル定義案）
- 年齢のみモデルの較正（08）を Raiden で実行 `@me` — `notebooks/08_age_baseline_calibration.ipynb`（ロジックは `scripts/age_baseline_calibration.py`）。
  年齢のみ・画像・画像 + 年齢の ECE / 較正切片 / スロープを 9 ユニットで出す。未実行（この環境では Python が動かない）。結果が届いたら草稿 3.2 節に表と図を入れる `@ai`
- 07 の結果の判定 `@me` — `results/gated_correction.md`、図 `results/figures/gated_correction_{fire,ece}.png`

## ブロッカー・要判断

- **k = 2.5 / N_min = 20 の採否** `@me` — 事前予測のうち LOSO-B（N=20 でほぼ常に発火）と field（N とともに発火が増える）は外れた。
  LOSO-B の利得が戻るのは N=30〜50。k=2.0 は in-dist の P(害>0.01) が 11% で、基準を僅差で落ちている
- **レポート反映** `@me` — 4/4 の書き方指示（「oracle 超え」と書かない、vendor を限界事項へ、Rao-Blackwell の段落、b̂ の内訳）は判定後に反映する
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
- 出力整合 06 の判定事項（害の定義・vendor・field 比方式の逆符号）→ 07 の仕様として回答を受領
- **出力整合**（ラベル無し切片推定）の実装と 3 評価 → `openidh_model/eval/output_matching.py`、`notebooks/06_output_matching.ipynb`、`results/output_matching.{md,json}`
- **有意性ゲート**の実装と評価 → `openidh_model/eval/gated_correction.py`、`notebooks/07_gated_correction.ipynb`、`results/gated_correction.{md,json}`
- **論文形式の総まとめ草稿 v0.1** → `results/manuscript_draft.md`（2026-10-20、Discord 依頼。既存レポートからの転記のみで、新しい計算はしていない）。v0.2 で図 12 枚。v0.3 で「画像を使わないモデル（年齢のみ）との比較」を 2.8・3.2 節に独立させた（既存結果 `results/age_prior.json` の転記）。v0.2 の内訳は図 12 枚（notebook 由来 5、Stage B レポート由来 5、新規作図 2：`results/figures/model_architecture.svg`・`beta_width.svg`）。モデル詳説（2.3）と「なぜ Beta 分布か」（2.4）を追加し、数式を $$ ブロックとプレーン表記に統一

## 次にやるべきこと（AI 提案）

次のセッションの人が最初に読む前提で書く。全体像はまず `results/manuscript_draft.md`、詳細は `results/gated_correction.md`、`results/output_matching.md`。

0. **草稿へのコメントを受けて改稿する**。k・N_min が確定したら 3.5 節・6 章の `【要判断】` を外す。疑いの代用（確定 grade の 2 値化）と vendor の書き方を確認してもらう

1. **k・N_min の判定を受けて、レポート本文を更新する**（判定後）。書き方の指示は 2026-09-28 の Discord 4/4 にある：
   「oracle 超え」ではなく「ラベル 0 で下限に到達」と書く／vendor の微悪化を原因（係留点の誤差：π̂_cov 30.9% 対 真値 35.0%）とともに限界事項へ入れ、ゲート後の変化（P(害) 81→3%）を追記／
   ラベルなしがラベルありより少ない N で安定する理由（Rao-Blackwell 化、LOSO-B でラベルなし 20 例 ≈ ラベルあり 44 例）／b̂ の内訳（+1.42 = +1.07 + 0.35）を実験 A と対応づける
2. LOSO-B の利得損失を減らしたければ、候補は 2 つ：ゲートの判定を N とともに逐次更新する（N=30 で発火 82%）、または k を N に依存させる。ただし事前固定した規則の外に出るので、やるなら新しい評価として切り分ける
3. vendor は切片では直らない（全例ラベルの切片でも ECE 0.086、補正なし 0.085）。残差は形の歪みとして別に扱う
