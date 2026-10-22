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

## 2026-10-23第二轮：联网深读与扩写

以`a3d672a`为基线，29正文全部补充推理链、数字/形状或操作步骤，保留原实验身份和负结果。以下为新增/再次核对的主要支持位置，联网阅读不只检查URL可达。

| 新内容 | 实读原文位置 | 采用与限制 |
| --- | --- | --- |
| 中训练的操作定义与分布桥接 | Emmy Liu、Graham Neubig、Chenyan Xiong，Midtraining Bridges Pretraining and Posttraining Distributions，arXiv2510.14865v2，2026-02-02，§2、5、6 | v2明确研究混合与加入时机/比例交互；论文定义比本书广义适配范围窄，未宣称行业唯一术语；不直接搬运最优比例 |
| 配比不能只追原始高loss | DoReMi，2305.10429v3，§2/Algorithm1与§4 | excess loss、参考/代理/大模型三步；未实现算法，不借其结果证明25%回放 |
| 近似去重与组划分 | HF BigCode官方文章，2023-05-16，MinHash Walkthrough、Beyond Duplicate Pairs、Quality Matters | 实读shingle/签名/LSH/连通组件；集合0.6和三节点图为本站手算，没有生产MinHash成绩 |
| 语言PPL与行为可能不同步 | Ahmed Elhady等，2506.00288v1，§4.1–4.4 | 仅作为多维评测动机，语言适配结果不等于NX任务结果 |
| 领域/任务分布 | DAPT，2004.10964v3，§3.3、4 | 原始掩码LM/分类条件保留；新增20目标手算不是论文成绩 |
| 长度/位置/长短数据 | ProLong2410.02660v1 §2–3、6.1；RoFormer2104.09864v5 §3；YaRN2309.00071v3 §3 | 重新阅读原文的构造、数据与验收条件；没有默认实现RoPE/YaRN或长依赖实测 |
| 稠密attention与IO | FlashAttention，2205.14135v2，摘要与原文方法概述 | tiled exact attention与HBM/SRAM读写；自编B4/S1024→B2/S2048的得分矩阵翻倍，未引用外部加速比为本站性能 |
| TP/PP/SP/CP/EP | NVIDIA-NeMo/Megatron-Bridge parallelisms.md，TP、PP、Activation Partitioning、MoE | 32通道切分、bias重复风险、forward-only75%均独立算例；不等于完整1F1B性能公式或真实多卡运行 |
| 有效分母、AMP边界 | PyTorch2.11 DDP、AMP examples、CrossEntropyLoss | 默认mean归约假设明确；四微批次50/50/300/600是手算，未新增DDP/AMP执行 |
| 事件区间、FLOPs估计 | PyTorch2.11 profiler，profile/Kineto参数、record_function | with_flops仅特定算子公式估计；实跑CPU区间标注，无硬件MFU/吞吐宣称 |
| 弹性/分片与版本 | PyTorch2.11 elastic.run Important Notices、distributed.checkpoint兼容性警告 | RANK/WORLD_SIZE可变不等于应用等价续训；本地仍拒绝跨布局严格resume |
| PPL协议 | Hugging Face官方Perplexity of fixed-length models | 先尝试v5.8.0页面不可访问，改读官方当前页面；独立块/滑窗/重复计分说明，不复制动态代码为本地版本 |
| 局部试验与阶段关系 | OLMo3 2512.13961v2 §3.3–3.6；SmolLM3官方Mid-training/Reasoning Mid-training | 保留项目差异；没有把代码/数学数据的形式当RL算法证据 |
| 保存/来源/实验环境 | PyTorch保存加载教程、2.11 randomness/checkpoint、SmolLM2固定revision模型卡 | 保存加载教程目前显示2.14，与本地torch2.11区分；新增代码实际运行绑定本地版本 |

元数据以arXiv规范记录、官方模型卡与官方文档核对；论文HTML固定版本。附录D给出带问题的阅读索引和直接链接。新增段落以本站语言和自编数值解释机制，不复制论文图表与段落。

原参考机器人样章仍依据前轮固定提交源码的实际阅读，不将本轮抓取其他材料写成重新打开了该原网页。新诊断是事后只读分析，CPU profiler是独立短跑，两者成绩只来自新增JSON。
