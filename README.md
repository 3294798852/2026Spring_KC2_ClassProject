# CV Class Project 2026 - 方向A 物体放置助手

这个项目是一个可交互的小应用：输入背景图 + 前景图，系统在多个候选位置上做本地推理评分，输出 Top-K 推荐位置。
默认评分模型为 `BCMI/libcom` 的 `SimOPA` 参考预训练模型（方向 A 推荐路线）。本项目已移除 legacy 主线，统一使用 SimOPA。

项目同时覆盖课程要求中的关键点：
- 可交互应用：`Streamlit` 本地 App。
- 模型适配：输入从 RGB 改为 `RGB+mask`（4 通道）。
- 输出改造：模型输出 `0~1` 连续合理性分数，并映射到 `推荐/可接受/不推荐` 三档。
- 功能类改动：候选位置自动生成 + 排序推荐。
- 候选优化：两阶段搜索（全局粗搜 + 局部步长收缩优化），替代纯随机/稀疏网格候选。
- 多尺度优化：可在当前缩放附近联合搜索多个尺度，提高推荐位置差异性与稳定性。
- 本地推理：CPU 可运行，不依赖云服务。
- 前景处理增强：支持 `jpg/jpeg/png/webp/bmp`，支持 U2Net 一键抠图与增强手工抠图（保留/擦除画笔、曲线多边形、掩码扩张/收缩、边缘羽化、智能结果精修、最大连通域保留、自动裁边与反选掩码）。
- 背景图输入支持：`jpg/jpeg/png/webp/bmp`。
- 大图推理加速：推理前可按长边自动缩放到 `1080P(1920)` 或 `2K(2560)`。
- 参考模型接入：`libcom/OPA` 的 SimOPA 预训练权重本地推理，评分差异显著高于随机小模型。
- 结果可解释提示：输出边界风险、支撑关系风险、语义区域偏移等提示。
- 手动微调与热力图：支持手动指定位置打分，热力图叠加 Top-K 标记点。
- 结果导出：支持一键导出 `ranking.json` 和 `simopa_results.zip`（含Top-K图、热力图图）。
- GPU 优先推理：默认自动优先 GPU（CUDA），无 GPU 时回退 CPU。

---

## 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 准备 SimOPA 权重

```bash
# 启动后在侧边栏点击“下载/检查 SimOPA 参考权重”
# 或直接在推理脚本中自动触发下载
```

输出：
- `models/SimOPA.pth`

## 3. 启动可交互 APP

```bash
streamlit run app.py
```

操作流程：
1. 建议先在侧边栏完成 SimOPA 权重检查。
2. 上传背景图（jpg/png）和前景图（支持 jpg/jpeg/png/webp/bmp）。
3. 可选开启“自动扣除背景”，并选择 `person` 或 `foreground` 抠图目标。
4. 用“前景缩放比例”调节前景大小。
5. 可选开启多尺度搜索，提升位置推荐质量。
6. 可选“推理时同步生成热力图”，在同一次推理中完成评分与热力图。
7. 点击“开始推荐”，查看 Top-K 推荐结果、分数分布与解释提示。

说明：
- “搜索预算（候选中心数）”是两阶段搜索中的候选中心数量，不是最终 Top-K 数量。
- “热力图网格密度”是每边采样点，实际采样约为 `grid^2`（例如 20 -> 400 点）。

## 4. 结果验证（示例脚本）

```bash
python scripts/evaluate.py --bg path/to/background.jpg --fg path/to/foreground.png
```

会输出：
- Top-K 推理耗时
- 分数 min/max/gap/std（检验是否出现“分数挤在一起”）
- 每个推荐候选的位置与尺度

批量评测（按同名文件配对）：

```bash
python scripts/batch_evaluate.py --bg-dir data/bg --fg-dir data/fg --out-csv outputs/simopa_batch.csv
```

---

## 5. 与课程评分点对应关系

### 基础项
- 应用目标与参考代码功能定位：本项目实现“物体放置合理性评分与推荐”的完整输入-处理-输出闭环。
- 基础模型小改动（本体类）：输入改造为 `RGB+mask`，输出改为连续分数+三档标签。
- 可交互应用：`app.py` 完成本地上传、模型推理、排序结果与图像可视化。
- 测试案例与基本结果说明：`scripts/evaluate.py` 提供速度/一致性指标；可补充真实案例截图。

### 进阶项（可计分）
- 本地推理：CPU 端本地推理（不依赖外部 API）。
- 复杂交互：候选数量、Top-K 可调、多尺度搜索、结果解释提示。

---

## 6. 代码结构

```text
app.py                         # 交互应用入口
src/
  compositor.py                # 前景贴图与候选生成
  foreground.py                # 自动抠图、缩放、尺寸适配
  reference_opa.py             # SimOPA 模型与权重管理
  user_feedback.py             # 候选位置解释提示
  infer.py                     # 推理与候选排序
scripts/
  evaluate.py                  # SimOPA 真实图对评测
```

---

## 7. 说明与限制

- 这是课程项目的 SimOPA 主线版本，重点是“参考模型接入 + 可解释推荐 + 可运行演示”。
- 首次运行若网络受限，可能导致权重下载失败。可手动将 `SimOPA.pth` 放入 `models/` 后重试。
- 若需移动端 App，可将 `app.py` 的推理逻辑封装为本地 Python 服务，再由 Android/iOS/Flutter 前端调用。
