# 实验记录与重跑范围

## 2026-10-22后续：后训练生命周期与系统连接

`lifecycle-system-reference.json`：实际运行`python examples/posttraining_lifecycle.py --ddp --out runs/lifecycle-integration-final-20261022`，10项检查。12参数表格完成SFT/DPO/RL交接和完成轮恢复；独立固定PPO张量由两个真实CPU/Gloo进程对照单进程，梯度与一次更新最大误差0，错误局部归约梯度差0.05694594。恢复不含生产队列，分布式子例也不构成LLM训练。

`lifecycle-regression-reference.json`保留本轮重跑的原16+5+12项及新增脚本默认单进程模式的报告；没有重训历史能力曲线。`lifecycle-browser-reference.json`与`lifecycle-source-reference.json`分别记录本轮页面验收和新增来源抓取范围；实际状态、限制以报告与VALIDATION.md为准。

## 2026-10-22：RL机制补充

`rl-mechanisms-reference.json`来自`python examples/rl_mechanisms.py --out runs/rl-mechanisms-20261022`，12项CPU/FP64枚举数值与梯度检查，含代码SHA与环境。没有LLM/agent/MARL/蒸馏模型训练。原训练核心未改，历史SFT/DPO/RLVR与基座SFT不在本轮重新训练。

`rl-regression-20261022.json`记录原16项恢复/协议和5项教学检查的本轮重跑；`rl-browser-reference.json`与`browser-regression-20261022.json`分别是新增13组与原13组实际浏览器检查。`rl-source-checks-reference.json`保留10份来源元数据、14个外部片段的批次与单项重试证据，包括超时FAIL；不把网络地址检查称为科学论断验证。

2026-10-15本机实际执行。完整权重和日志留在runs，JSON保留配置、源码/数据身份、逐点指标和原始生成，正文明确区分假设与实测。

| 文件 | 内容 | 重跑入口 |
| --- | --- | --- |
| offline-reference.json | 最终77train/23val，SFT300与DPO120，全部验证输出和权重来源 | prepare_data.py → train.py SFT → train.py DPO，命令见6.2 |
| rlvr-reference.json | 48参数表格策略、12题四候选、80轮160次更新、全部轮次指标 | python examples/rlvr.py --out runs/rlvr |
| checks-reference.json | 16项实际通过的数值/协议/恢复/拒绝检查及代码身份 | python examples/verify.py --out runs/checks |

source_identity按有序源码文件名与规范化UTF-8/LF内容计算SHA-256：语言模型绑定postcore.py、train.py，RL绑定postcore.py、rlvr.py。数据对实际JSONL字节计算；manifest还保存答案覆盖。完整快照仅加载可信文件。

SFT与DPO沿同一权重交接，但不是独立算法胜负试验；没有多种子和独立产品test。pair准确率与生成精确率分别计数，负结果保留。表格RL没有语言模型或未见题泛化，不能与小LM比较能力。

两张图直接从这些JSON绘制，原始数值可核对；图片已独立预览检查，不等同网站桌面/移动截图。LoRA/OPD/安全/裁判/分布式等未运行范围详见VALIDATION.md。

## 2026-10-16教学修订的新增证据

- pretrained-reference.json：固定SmolLM2-135M revision，24步真实基座SFT、18训练/6目标/3保留探针、全部前后输出、源码/下载文件哈希与成本口径。脚本入口pretrained_sft.py；首次显式download，后续cache-only。
- teaching-checks-reference.json、tool-trace-reference.json：5项实际CPU/FP64数值与接口检查，包括RM比较损失、基线抵消和两个assistant动作片段的右移mask；不是神经RM或工具LLM训练。
- checks-revision-20261016.json：本轮重跑原16项CPU恢复/数值检查，训练核心源码身份与首版相同。

新的基座报告与原随机模型负结果分别保留；完整权重仍在忽略的runs目录。基座脚本为新运行接口，没有严格resume承诺。

pretrained-artifact-checks.json：实际重载保存后的基座SFT产物，全部9条生成与结束原因相同，两组NLL误差小于1e-6；不代表严格恢复。browser-checks-reference.json记录本地构建站的13组浏览器检查，不等同所有浏览器/全文的视觉验收。
