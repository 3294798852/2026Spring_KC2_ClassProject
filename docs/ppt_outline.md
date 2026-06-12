# 课堂汇报 PPT 建议（第十五周）

## 1. 题目与应用目标
- 方向 A：智能物体放置与合成质量评价
- 应用目标、用户流程、输入输出

## 2. 系统架构与主链路
- 上传背景/前景 -> 候选生成 -> 模型评分 -> 排序推荐 -> 结果展示

## 3. 模型改动（重点）
- 输入改造：RGB -> RGB+mask
- 输出改造：二分类 -> 0~1 分数 + 三档标签
- 轻量化蒸馏：Teacher SimOPA -> Student CNN（MobileNetV3-Small 4ch）
- 蒸馏损失：CE + KD + Feature + Rank
- 原型探索：Student Dual+Geom
- 中型增强：Student Mid（ResNet18-4ch-width0.75, 约6.49M）

## 4. 结果展示
- Top-K 推荐截图
- 好案例 / 坏案例对比
- 四后端对比（SimOPA / Student CNN / Student Dual / Student Mid）速度与分数分布
- DenseMap 加速策略效果

## 5. 现场可证明“真实模型推理”
- 展示终端日志
- 指认权重加载与前向推理代码
- 新图片现场测试
- 展示训练日志文件（config/metrics/summary）证明实验可复现

## 6. 已知问题与后续计划
- 真实场景泛化不足
- 增加真实数据、引入更严格评测指标、继续推进 FOPA 风格密集预测
