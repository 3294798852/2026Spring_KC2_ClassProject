# CV Class Project 2026 - 方向A 物体放置助手

这个项目是一个可交互的小应用：输入背景图 + 前景图，系统在多个候选位置上做本地推理评分，输出 Top-K 推荐位置。
默认评分模型为 `BCMI/libcom` 的 `SimOPA` 参考预训练模型（方向 A 推荐路线），并保留 legacy 对照模型。

项目同时覆盖课程要求中的关键点：
- 可交互应用：`Streamlit` 本地 App。
- 模型适配：输入从 RGB 改为 `RGB+mask`（4 通道）。
- 输出改造：模型输出 `0~1` 连续合理性分数，并映射到 `推荐/可接受/不推荐` 三档。
- 功能类改动：候选位置自动生成 + 排序推荐。
- 候选优化：两阶段搜索（全局粗搜 + 局部步长收缩优化），替代纯随机/稀疏网格候选。
- 本地推理：CPU 可运行，不依赖云服务。
- 模型压缩：教师-学生蒸馏 + 剪枝 + 动态量化。
- 前景处理增强：支持 `jpg/jpeg/png/webp/bmp`，支持自动抠图（人物/前景）与前景缩放。
- 背景图输入支持：`jpg/jpeg/png/webp/bmp`。
- 大图推理加速：推理前可按长边自动缩放到 `1080P(1920)` 或 `2K(2560)`。
- 参考模型接入：`libcom/OPA` 的 SimOPA 预训练权重本地推理，评分差异显著高于随机小模型。

---

## 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 一键初始化模型（训练+压缩）

```bash
# 推荐：高质量离线预训练（首次）
python scripts/bootstrap_and_compress.py --profile standard

# 调试：快速版本
python scripts/bootstrap_and_compress.py --profile quick
```

输出：
- `models/teacher.pth`（参考大模型）
- `models/student.pth`（小模型）
- `models/student_compressed.pth`（压缩后模型）

## 3. 启动可交互 APP

```bash
streamlit run app.py
```

操作流程：
1. 建议先离线执行一次训练+压缩脚本（使用阶段只推理）。
2. 上传背景图（jpg/png）和前景图（支持 jpg/jpeg/png/webp/bmp）。
3. 可选开启“自动扣除背景”，并选择 `person` 或 `foreground` 抠图目标。
4. 用“前景缩放比例”调节前景大小。
5. 可选切换“原始模型/压缩模型”推理。
6. 点击“开始推荐”，查看 Top-K 推荐结果与分数。

## 4. 结果验证（示例脚本）

```bash
python scripts/evaluate.py
```

会输出：
- 原始学生模型平均推理耗时
- 压缩模型平均推理耗时
- 原始与压缩模型输出分数相关性

---

## 5. 与课程评分点对应关系

### 基础项
- 应用目标与参考代码功能定位：本项目实现“物体放置合理性评分与推荐”的完整输入-处理-输出闭环。
- 基础模型小改动（本体类）：输入改造为 `RGB+mask`，输出改为连续分数+三档标签。
- 可交互应用：`app.py` 完成本地上传、模型推理、排序结果与图像可视化。
- 测试案例与基本结果说明：`scripts/evaluate.py` 提供速度/一致性指标；可补充真实案例截图。

### 进阶项（可计分）
- 模型改造：教师-学生蒸馏（大模型到小模型）。
- 本地推理：CPU 端本地推理（不依赖外部 API）。
- 复杂交互：候选数量、Top-K 可调，多结果并排比较。

---

## 6. 代码结构

```text
app.py                         # 交互应用入口
src/
  compositor.py                # 前景贴图与候选生成
  foreground.py                # 自动抠图、缩放、尺寸适配
  models.py                    # Teacher/Student 模型定义
  data_synth.py                # 合成训练样本生成
  train.py                     # 教师训练 + 学生蒸馏
  compress.py                  # 剪枝 + 动态量化
  infer.py                     # 推理与候选排序
scripts/
  bootstrap_and_compress.py    # 一键训练+压缩
  evaluate.py                  # 速度与一致性评估
```

---

## 7. 说明与限制

- 这是课程项目的最小可行版本，重点是“完整链路可运行 + 模型改造可说明”。
- 当前训练数据为程序生成的合成样本（`src/data_synth.py`），建议后续替换为真实合成数据以提高效果。
- 若需移动端 App，可将 `app.py` 的推理逻辑封装为本地 Python 服务，再由 Android/iOS/Flutter 前端调用。
