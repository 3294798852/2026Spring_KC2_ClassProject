# CV Class Project 2026 - 方向A 物体放置助手

本项目是一个本地可运行的物体放置评分应用。输入背景图和前景图后，系统输出 Top-K 推荐位置、分数解释、热力图与可导出结果。

当前代码已支持多后端：
- `原始 SimOPA`
- `Student CNN`（MobileNetV3-Small 4ch 蒸馏版）
- `Student Dual+Geom (exp)`（实验原型）
- `Student Mid (5-8M)`（ResNet18-4ch 宽度缩放版，精度优先）

## 核心能力

- 输入统一为 `composite RGB + foreground mask`（4 通道评分范式）。
- 推理支持两种策略：
  - `热力图引导搜索（默认）`
  - `DenseMap加速（实验）`
- UI 支持后端切换、双后端对比、手动拖拽微调、热力图可视化与结果导出。
- 训练支持蒸馏损失组合：`CE + KD + Feature + Rank`。
- 训练脚本支持性能优化：AMP、TF32、channels_last、torch.compile（可选）。
- 训练自动记录日志：`config.json`、`metrics.csv`、`summary.json`。

---

## 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. 权重准备

- SimOPA 权重：`models/SimOPA.pth`
- Student CNN 权重：`models/student_cnn.pth`
- Student Dual 权重：`models/student_dual_geom.pth`
- Student Mid 权重：`models/student_mid.pth`

你可以在 App 侧边栏点击“下载/检查 SimOPA 参考权重”自动准备 SimOPA 权重。

---

## 3. 启动应用

```bash
streamlit run app.py
```

建议流程：
1. 在侧边栏选择后端模型（SimOPA / Student CNN / Student Dual / Student Mid）。
2. 上传背景图与前景图（支持 `jpg/jpeg/png/webp/bmp`）。
3. 选择前景处理（不抠图 / 一键抠图 / 手工抠图）。
4. 选择搜索预算、热力图密度、推理策略（默认或 DenseMap）。
5. 点击“开始推荐”查看 Top-K、热力图、解释提示与导出结果。

---

## 4. 训练（云端推荐）

### 4.1 Student CNN 蒸馏训练

```bash
python scripts/train_student_cnn.py --device cuda --epochs 20 --batch-size 128 --num-workers 8 --channels-last
```

可选提速参数：

```bash
--compile-model --compile-mode reduce-overhead
```

训练日志自动落盘到：

```text
logs/student_cnn_YYYYmmdd_HHMMSS/
  config.json
  metrics.csv
  summary.json
```

### 4.2 Student Dual+Geom 训练

```bash
python scripts/train_student_dual.py --device cuda --epochs 20 --batch-size 128 --num-workers 8 --channels-last
```

日志结构与 Student CNN 一致。

### 4.3 Student Mid (5-8M) 训练

```bash
python scripts/train_student_mid.py --device cuda --epochs 20 --batch-size 128 --num-workers 8 --channels-last
```

推荐首版参数（方案默认）：

```bash
--lr 2e-4 --temperature 2.5 --alpha-kd 0.5 --beta-feat 0.10 --gamma-rank 0.03 --distill-warmup-epochs 3 --freeze-head-epochs 2 --ema-decay 0.999
```

---

## 5. 评测脚本

### 单样本评测

```bash
python scripts/evaluate.py --bg path/to/bg.jpg --fg path/to/fg.png --backend 原始 SimOPA
```

四后端对比：

```bash
python scripts/evaluate.py --bg path/to/bg.jpg --fg path/to/fg.png --compare-all --out-json outputs/eval_report.json
```

### 批量评测

```bash
python scripts/batch_evaluate.py --bg-dir data/bg --fg-dir data/fg --compare-all --out-csv outputs/batch_eval.csv
```

---

## 6. 数据组织（训练）

默认参数下，脚本读取：

- `new_OPA/train_set.csv`
- `new_OPA/test_set.csv`

CSV 关键字段：

- `img_name`：合成图路径
- `mask_name`：对应 mask 路径
- `label`：0/1

图像建议放在 `new_OPA/` 子目录中，CSV 写相对路径。

---

## 7. 代码结构

```text
app.py
src/
  infer.py
  opa.py
  reference_opa.py
  student_opa.py
  student_opa_dual.py
  student_opa_mid.py
  foreground.py
  compositor.py
  user_feedback.py
scripts/
  train_student_cnn.py
  train_student_dual.py
  train_student_mid.py
  evaluate.py
  batch_evaluate.py
docs/
  simopa_upgrade_plan.md
  report_outline.md
  ppt_outline.md
```

---

## 8. 说明

- 首次运行若网络受限，可能导致 SimOPA 权重下载失败，可手动放入 `models/SimOPA.pth`。
- 若使用 `torch.compile` 遇到环境兼容问题，可去掉 `--compile-model`，保留 AMP 仍有明显加速。
