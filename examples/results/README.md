# 实验记录与重跑范围

2026-10-15本机实际执行。完整权重和日志留在runs，JSON保留配置、源码/数据身份、逐点指标和原始生成，正文明确区分假设与实测。

| 文件 | 内容 | 重跑入口 |
| --- | --- | --- |
| offline-reference.json | 最终77train/23val，SFT300与DPO120，全部验证输出和权重来源 | prepare_data.py → train.py SFT → train.py DPO，命令见6.2 |
| rlvr-reference.json | 48参数表格策略、12题四候选、80轮160次更新、全部轮次指标 | python examples/rlvr.py --out runs/rlvr |
| checks-reference.json | 16项实际通过的数值/协议/恢复/拒绝检查及代码身份 | python examples/verify.py --out runs/checks |

source_identity按有序源码文件名与规范化UTF-8/LF内容计算SHA-256：语言模型绑定postcore.py、train.py，RL绑定postcore.py、rlvr.py。数据对实际JSONL字节计算；manifest还保存答案覆盖。完整快照仅加载可信文件。

SFT与DPO沿同一权重交接，但不是独立算法胜负试验；没有多种子和独立产品test。pair准确率与生成精确率分别计数，负结果保留。表格RL没有语言模型或未见题泛化，不能与小LM比较能力。

两张图直接从这些JSON绘制，原始数值可核对；图片已独立预览检查，不等同网站桌面/移动截图。LoRA/OPD/安全/裁判/分布式等未运行范围详见VALIDATION.md。
