# 项目报告建议结构（可直接填充）

## 1. 应用目标与场景
- 目标：帮助用户完成前景物体在背景图中的合理放置与推荐。
- 输入：背景图 + 前景图（含 alpha）。
- 输出：Top-K 放置建议、每个候选位置评分、三档标签。

## 2. 参考代码功能定位
- 参考方向：BCMI `libcom` / OPA / TopNet 的“物体放置与质量判断”思想。
- 本项目定位：复现“候选位置评分 + 排序推荐”主链路，完成本地可运行 MVP。

## 3. 模型小改动（必须重点写）
- 本体类改动 1：输入由 RGB 改为 `RGB+mask`（4 通道）。
- 本体类改动 2：输出由二分类改为连续分数 `0~1`，并映射到三档标签。
- 本体类改动 3（轻量化）：Teacher（SimOPA）蒸馏到 Student CNN（MobileNetV3-Small 4ch），并增加 `KD + Feature + Rank` 蒸馏损失。
- 本体类改动 4（原型）：Dual Encoder + Geometry MLP（Student Dual+Geom）。
- 本体类改动 5（中型增强）：`Student Mid (ResNet18-4ch-width0.75)`，参数量约 6.49M，在速度和精度间取得更平衡效果。
- 功能类改动：多候选自动生成与排序返回 Top-K；DenseMap 加速路径（实验）。

## 4. 可交互应用与推理链路
- 入口：`streamlit run app.py`。
- 数据流：上传图像 -> 候选/热力图评分 -> 排序 -> 可视化 -> 导出。
- 前端展示：后端切换（SimOPA/Student CNN/Student Dual/Student Mid）、多后端对比、推理策略切换。
- 本地推理证据：展示终端训练/推理日志与代码中的权重加载位置。

## 5. 测试案例与结果
- 至少展示 6 组候选位置对比。
- 给出“效果较好”和“效果较差”的典型案例。
- 记录推理时间、参数量、分数分布（gap/std）、Spearman 排序相关。
- 对比报告来源：`scripts/evaluate.py` 与 `scripts/batch_evaluate.py` 的 JSON/CSV。

## 6. 训练工程与日志
- 训练脚本：`scripts/train_student_cnn.py`、`scripts/train_student_dual.py`、`scripts/train_student_mid.py`。
- 加速配置：AMP、TF32、channels_last、可选 torch.compile、DataLoader prefetch/persistent。
- 日志目录：`logs/<run_id>/config.json + metrics.csv + summary.json`。
- 说明如何用日志判断：收敛速度、最佳 epoch、是否过拟合。

## 7. 失败案例与分析
- 边界越界、遮挡冲突、语义不合理时模型仍会误判。
- 合成数据与真实数据分布差异会导致泛化下降。
- DenseMap 模式下可能出现局部最优，需结合局部精修。

## 8. AI 辅助说明
- AI 参与部分：脚手架与基础代码生成、界面模板、文档草稿。
- 人工接管部分：数据流核对、模型输入输出核实、蒸馏训练与评测脚本调试。
- 关键问题定位：训练速度瓶颈、torch.compile/cudagraph冲突、拖拽状态同步问题。

## 9. 分工与总结
- 成员分工、完成度、后续优化计划（真实数据、移动端封装、模型解释）。
