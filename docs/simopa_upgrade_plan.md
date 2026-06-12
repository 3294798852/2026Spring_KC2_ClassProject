# SimOPA 主线改进设计（v2）

## 1. 目标
- 将项目主线统一到 `SimOPA` 评分范式（4ch：RGB+mask）。
- 在保证质量的前提下推进轻量化：`SimOPA -> Student CNN -> Student Dual+Geom -> Student Mid (5-8M)`。
- 提升端到端速度：搜索策略升级为“热力图引导 + DenseMap加速（实验）”。
- 完善工程可复现性：训练日志、评测报告、UI双后端对比。

## 2. 当前问题与对应方案

### P0. 训练慢且不可追踪
- 问题：本地/云训练耗时高，缺少结构化日志，不利于对比实验。
- 方案：训练脚本加入 AMP、TF32、channels_last、可选 compile、DataLoader 优化，并自动记录 `config.json + metrics.csv + summary.json`。
- 验收：每次训练均生成独立日志目录，能追溯超参与每轮指标。

### P0. 学生模型效果不足
- 问题：早期轻量 CNN 精度下降明显。
- 方案：Student CNN 升级为 MobileNetV3-Small 4ch，并采用 `CE + KD + Feature + Rank` 蒸馏训练。
- 验收：在统一评测脚本下，学生模型速度提升且精度差距可控。

### P0.5 中型学生模型精度补强（5-8M）
- 问题：超轻量模型速度优势明显，但与 SimOPA 仍存在质量差距。
- 方案：新增 `Student Mid (ResNet18-4ch-width0.75)`，参数量约 6.49M，训练采用 `CE + KD + Feature + Rank`，并加入分段解冻和 EMA 评估稳定训练后期波动。
- 验收：在 `evaluate.py/batch_evaluate.py --compare-all` 下，`Student Mid` 的质量指标（val_acc/Spearman）显著优于 `Student CNN`，且时延仍优于原始 SimOPA。

### P1. 评测闭环不完整
- 问题：缺少统一的“质量 + 速度 + 参数量”横向对比。
- 方案：`evaluate.py / batch_evaluate.py` 支持单后端与双后端对比，导出 JSON/CSV 报告。
- 验收：课程报告可直接使用脚本导出的对比数据。

### P1. UI 与后端能力不同步
- 问题：后端切换、双后端对比、推理策略切换在展示层不完整。
- 方案：UI 增加后端选择、双后端对比开关、DenseMap加速策略选择，并同步导出 metadata。
- 验收：App 内可一键对比后端时延与分数统计。

### P2. Dense/Fast 路线验证
- 问题：逐点打分前向次数高，端到端时延仍偏大。
- 方案：在 `infer.py` 增加 `rank_candidates_dense_map`（先密集网格评分，后局部精修）。
- 验收：相同预算下 DenseMap 模式时延可下降，质量基本可用。

## 3. 实施顺序
1. 训练加速与日志化  
2. Student CNN 蒸馏升级  
3. 评测脚本闭环（单后端/四后端对比）  
4. UI 展示与导出同步  
5. Student Dual+Geom 原型  
6. Student Mid (5-8M) 接入与训练脚本  
7. DenseMap 加速路径  
8. README/docs 同步

## 4. 后续增强（v3）
- 增加训练自动 benchmark 模式（自动搜索 batch-size / num-workers）。
- 增加 AUC/Spearman 更严格评测（引入真实标注或半自动标注集）。
- 进一步探索 FOPA 风格单次前向密集预测网络。
