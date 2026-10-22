# 中训练论断与证据核对

核对日期2026-10-23。下表是人工阅读内容与适用条件的核对；不是把HTTP成功当作论断验证。正文保留具体来源链接。网页动态内容可能更新，论文HTML尽量固定版本。

| 论断/位置 | 主要依据与阅读位置 | 核对与边界 |
| --- | --- | --- |
| 1.1/阶段导读：mid-training命名不统一 | SmolLM3官方报告2025-07-08，Mid-training及Reasoning Mid-training；OLMo3 arXiv2512.13961v2，§2.1、§3.5–3.6 | 前者明确称长上下文/推理为mid-training；后者单列长上下文扩展。不是强制定名 |
| 2–3章：领域继续学习/任务适配与对照 | Don't Stop Pretraining，2004.10964v3，§3–4 | 原论文RoBERTa/掩码LM与分类，不能写成本书decoder结果复现 |
| 3.3/5.3：长上下文数据和评测 | ProLong2410.02660v1 §2、3.2–3.3、6.1；HELMET2410.02694v1 §2–3 | 多任务/短任务保留与简单检索探针不足；未把单项目长短配比推广成最优 |
| 3.3：位置机制与扩展 | RoFormer2104.09864v5 §3；YaRN2309.00071v3 §3 | RoPE旋转/缩放方法有条件；默认模型学习位置64，不包含这两种机制 |
| 3.5：退火支路与数据整合 | WSD2410.05192v1 §1、3、5；OLMo3 §3.5 | 前者的稳定/衰减动机及理论假设，后者的微退火与整合验证；本轮warmup-cosine未检验WSD |
| 4.1：目标加权、DDP与AMP顺序 | PyTorch2.11 CrossEntropyLoss文档reduction定义、DDP设计及AMP examples中gradient accumulation/gradient clipping | 显式假设普通DDP平均归约；完整副本裁剪与分片裁剪分开，未声称本书跑过DDP/AMP |
| 4.3/5.4：计算与性能估算 | Chinchilla2203.15556v1 Appendix F/FLOP comparison；PyTorch checkpoint文档 | 6ND是密集模型近似，attention/重计算/通信另看；无硬件MFU实测 |
| 4.4：DP/TP/PP/CP/EP与设备数 | NVIDIA-NeMo/Megatron-Bridge docs/parallelisms.md，DP/TP/PP/CP/MoE及DP组大小 | 人工核对逻辑布局；16卡是自编假设，不是已跑集群或框架默认 |
| 4.5–4.6：弹性与恢复 | PyTorch2.11 elastic.run、distributed.checkpoint、saving/loading、randomness | 框架能力不等于本书实现；默认world size1，跨布局严格恢复拒绝 |
| 6.5：公开模型身份与许可 | SmolLM2-135M固定revision93efa2f…模型卡/config/tokenizer与下载文件SHA | 134,515,008独立参数由本地实数；性能仅依据本轮报告，不引用模型卡当作实验成绩 |
| 教学顺序 | 机器人架构与写作原则，本地源0cfa5fe… | 实读原文源码；原网页抓取失败，无原网页截图比较 |

来源入口集中于附录D。所有默认与公开基座数值来自本轮实际JSON，假设设备布局/算例没有被表述成训练结果。实验方法技能归属见ATTRIBUTION.md。
