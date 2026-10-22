# 关键论断与来源支持位置

## 2026-10-22：RL基础与训练系统贯穿修订

本轮在已有10份方法资料之外，对照下列教材、近期综述与原始系统材料，重组解释关系。正文使用独立订单场景、形状与假设预算，不复制来源图形/实验成绩。先看定义和原文条件，再决定如何接入本书；不是见到“权威”就接受每个经验阈值或所有分类措辞。

| 来源与所读版本 | 已阅读支持位置 | 采用与边界 |
| --- | --- | --- |
| [OpenAI Spinning Up Part1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) | Key Concepts：state/observation、policy、trajectory、return；Value Functions、Bellman Equations、Advantage Functions | RL基础页的任务与G/V/Q/A；订单成本、Q0.4/0.8、V0.6为本站枚举算例；不声称读者短上下文天然Markov |
| [Part2](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)与既有Part3 | Model-Free vs Model-Based、What to Learn、价值/策略路线；Part3基线和policy gradient | 不将语言模型体量大解释成model-based RL；保留动作无关基线条件，不宣称准确V总能取得最小梯度方差 |
| Nathan Lambert，[Training Overview](https://rlhfbook.com/c/03-training-overview)、[RL章](https://rlhfbook.com/c/06-policy-gradients) | Problem Formulation、Canonical Recipes；policy gradient、PPO/GAE与Implementation Details | 阶段与信号连接；原始recipe仍需回原论文，本站四入口图独立组织。在线版2026-10-22检索，主页标构建2026-10-14 |
| [RLHF Book Lecture4](https://rlhfbook.com/teach/course/lec4-chap6-p2/) | From math to code；PPO implementation；The four models in memory；loss aggregation；async infrastructure；summary | 张量/mask/分母/资源接口；不复制课堂监控区间为普遍标准，不把常见PPO+GAE路径说成数学唯一选择 |
| [统一后训练综述2604.07941v2](https://arxiv.org/html/2604.07941v2)，2026-04-16 | §2.2明确与通用online/offline术语不同；§2.3–2.4轨迹/接口；§3.4/3.6–3.7角色与限制；§6阶段交接 | 1.3与阶段契约；作为分析框架而非定理，不将SFT/蒸馏统称一般offline RL，不推导固定必经顺序 |
| [GFCR综述2605.02913v1](https://arxiv.org/html/2605.02913v1) | §3.1–3.2管线分布和接口；§4生成、§5信号、§6预算/控制、§7Replay | 1.3与5.4的采样/筛选/队列/预算；其Replay含义更广，不全当训练经验回放。日期以arXiv记录为准，不从编号推定发表月份 |
| [AReaL2505.24298v5](https://arxiv.org/html/2505.24298v5)，2026-03-02 | §3.2同步长尾；§4.1–4.2可中断生成、KV丢弃重算和多片段版本；§5.1–5.2陈旧度/解耦目标 | 5.4说明系统选择改变算法假设；不复制原文图形或速度，本站没有运行AReaL或GPU异步系统 |
| [PyTorch2.11 DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html) | loss求和时的梯度平均说明及no_sync语义 | 本站固定PPO张量CPU两rank验证P/global_count；不代表分片参数、AMP或完整LLM PPO |
| [PyTorch2.11 AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html) | Working with Unscaled Gradients；Gradient Accumulation；Multiple Models/Losses/Optimizers | 累积结束再unscale与clip、各优化器可能独立跳过；只讲接口，未执行CUDA AMP |
| [DPO原文§3](https://arxiv.org/html/2305.18290v3#S3) | 正则化奖励目标与最优分布关系 | 本站二动作闭式检查奖励/β尺度，0.993307与0.999955不作为神经模型成绩 |

新生命周期脚本绑定同一12参数表格在SFT/DPO/RL之间的参数来源、固定参考、AdamW新建和专用采样Generator；真实完成轮恢复与独立两rank固定张量对照分别报告。8轨迹/28位置/4更新是另一个教学计数算例；36GB状态与128MiB KV是不同的预算假设，不能混作运行实测。权重发布只模拟本地快照确认，不是网络RPC/重新分片验收。

---

## 2026-10-22：RL专题扩充

检索覆盖PPO/GAE、在线/离线与采样校正、GRPO变体、agentic RL、多智能体信用分配、OPD/OPSD和固定学生轨迹蒸馏。先用网页检索发现候选，再回到arXiv/官方文档核对标题、首作者、版本、日期及以下正文位置；没有引用聚合站或二手总结作为技术依据。选择兼顾近期性与教学用途，不声称是完整系统综述或全网最新排名。

| 重点资料（对应附录F编号） | 实际阅读的支持位置 | 采用与限制 |
| --- | --- | --- |
| 1. [verl PPO](https://verl.readthedocs.io/en/latest/algo/ppo.html)，页面标2025-06-19 | Key Components、Configuration、KL Divergence Control | mini/micro/global batch与epoch、KL放置；没有复制页面中的配置拼写或声称运行verl；actor-critic是常见实现，不是剪裁公式的逻辑必要条件 |
| 2. [verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)，页面标2026-09-23；[数学说明](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html) | Overview；数学§1.1–1.3轨迹IS、§2三策略、校正与诊断 | 区分μ/π_old/π_current，补0.4/0.5/0.6自编算例；精确轨迹IS假设、局部代理与截权偏差分别说明，不从官方措辞推出任意分布差都能消除 |
| 3. [DAPO 2503.14476v2](https://arxiv.org/html/2503.14476v2)，2025-05-20 | §2.1–2.3；§3.1–3.4 Eq10–13 | 非对称clip、动态采样、token归约、超长奖励和原实验去KL条件；384生成/160保留是本站独立计数，未复制论文成绩 |
| 4. [GSPO 2507.18071v2](https://arxiv.org/html/2507.18071v2)，2025-07-28 | §3；§4.1 Eq5–7、§4.2 Eq8–10；§5.3–5.4 | 长度归一化序列ratio与梯度；不将几何比值称精确联合IS，也不把论文MoE经验写成普遍保证 |
| 5. [Agentic RL survey 2509.02547v5](https://arxiv.org/html/2509.02547v5)，2026-04-17；arXiv标TMLR | §2.1–2.7的状态、动作、转移、奖励、目标；§3工具/记忆框架 | 组织单轮与交互任务差异，不断言单轮token生成没有序贯结构，具体算法回到原论文 |
| 6. [Agent Lightning 2508.03680v1](https://arxiv.org/html/2508.03680v1)，2025-08-05 | §3.1统一接口、§3.2 MDP/数据提取、§3.3 LightningRL、§3.4执行/训练分离 | 事件/角色/状态与训练交接；附录E工单schema自编，不承诺任意agent零改动或已运行框架 |
| 7. [C3 2603.06859v1](https://arxiv.org/html/2603.06859v1)，2026-03-06 | §2四项接口假设、§3.1冻结上下文、§3.2固定后续分布、§3.3 LOO、§4.1条件 | 决策级反事实问题；[1,0,1,1]为本站构造的等次数LOO，不是C3环境重放。无偏性只在原假设内讨论 |
| 8. [OPSD 2601.18734v3](https://arxiv.org/html/2601.18734v3)，2026-03-20 | §3.2师生上下文、全词表散度Eq6–8、梯度仅过学生、点式剪裁、采样替代；附录D | 特权解答与训练/推理隔离、同参数教师更新。本站reverse-KL例子不是原文默认JSD配方，未训练OPSD |
| 9. [Rethinking OPD 2604.13016v2](https://arxiv.org/html/2604.13016v2)，2026-04-15 | §2.2监督粒度、§3师生差异、§5冷启动与模板、§6长前缀退化 | 区分目标数值与梯度估计器，师生兼容与监督深度是条件化经验，不升级为任意任务的必要充分定理 |
| 10. [Semi-OPD 2610.11291v1](https://arxiv.org/html/2610.11291v1)，2026-10-08 | §2.1–2.2 Eq1–3及缓存流程、§2.3对照协议、§3重叠指标、§4解释 | 固定初始学生轨迹、重算当前logp、离线教师缓存；不从近期预印本外推普遍最优，不把蒸馏叫一般offline RL |

基础补充：[PPO §2–3](https://arxiv.org/pdf/1707.06347)、[GAE §3](https://arxiv.org/html/1506.02438v6#S3)、[Offline RL教程§2–4](https://arxiv.org/html/2005.01643v3#S2)、[RLOO §2.2–2.3](https://arxiv.org/html/2402.14740v2)、[MAPPO §3.2–3.3](https://arxiv.org/html/2103.01955v4#S3.SS2)。这些是原始基础，明确不计作2026新论文。MAPPO用于解释CTDE/参数共享，不借其非语言环境成绩证明LLM协作收益。

以下均为本站独立构造并由`rl_mechanisms.py`枚举核对，不宣称来源论文运行结果：两步GAE0.78/0.4、clip四区间、IS 0.6/截权0.48、缺覆盖0.3/0.8、两种长度归约、GSPO几何权重反例、RLOO尺度、固定状态KL梯度0.15693268与停止梯度、双角色梯度0.09375/0.0625。它们不验证真实语言模型、环境重放或生产采样系统。

可选网络检查`scripts/check-rl-sources.py`仅核对10份资料的标题/作者及新增外部片段是否存在，结果与抓取SHA另存报告。其PASS不意味着科学论断通过；论断支持依据是上表人工阅读的位置。动态官方文档记录检索日期与内容SHA，论文链接固定所读版本。

---

核对日期2026-10-15。下表区分原始来源支持的方法与本站独立手算/实测，不以HTTP可达代替论断核验，也不声称逐句验证了所有公开项目。

| 位置 / 论断 | 已阅读的一手来源位置 | 本书使用边界 |
| --- | --- | --- |
| 1.1/1.2：示范、偏好、RL是不同监督对象 | [InstructGPT](https://arxiv.org/html/2203.02155v1) §3方法与数据；[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 多阶段设计 | 不将其中一种顺序称所有项目必经，不引用其模型成绩当本站结果 |
| 1.3系统关系 | [DPO](https://arxiv.org/html/2305.18290v3) §3/4；[DeepSeekMath](https://arxiv.org/html/2402.03300v3) §4；[HybridFlow](https://arxiv.org/html/2409.19256v2) §2.3–4与Figure4 | 三条路径和接口表由本站重新组织，不套机器人纵向控制层 |
| 2.2模板控制符与特殊token | [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating) 的消息格式、generation prompt与重复special tokens说明 | 只说明接口；本代码用原创ASCII/UTF-8字节约定，不假装运行Transformers tokenizer |
| 2.2/3.1回答监督与mask | [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html) 的输入/目标/reduction定义；本站postcore实现 | 实际byte ID、右移与mask由数值测试验证；prompt不计loss不等于不可见 |
| 2.3/3.4原始DPO目标 | [DPO §4 Eq7、梯度解释与附录B](https://arxiv.org/html/2305.18290v3) | 冻结参考、同prompt成对、回答联合logp；不把长度平均当同一目标，β不偷换为代码里附加KL项 |
| 3.2低秩矩阵 | [LoRA §4.1](https://arxiv.org/html/2106.09685v2) 的W0+BA、冻结参数与缩放说明 | 64×64/r4→512参数是本站手算，不推导整机同比节省，不报告LoRA实测 |
| 3.5方法比较 | [IPO](https://arxiv.org/abs/2310.12036)、[KTO](https://arxiv.org/abs/2402.01306)、[SimPO](https://arxiv.org/abs/2405.14734)、[ORPO](https://arxiv.org/abs/2403.07691)的方法定义 | 只比较输入、参考与目标，不把论文不同条件的结果拼成排行榜；本站未运行这四种方法 |
| 4.1旧策略/采样与4.2剪裁符号 | [PPO §3 Eq7及算法](https://arxiv.org/pdf/1707.06347)；[DeepSeekMath §4.1.1 Eq1–3](https://arxiv.org/html/2402.03300v3) | ρ1.5/A±1的值独立手算并实际测试，不保证clip能约束全部参数变化 |
| 4.3组基线与去critic | [DeepSeekMath §4.1.1–4.1.2](https://arxiv.org/html/2402.03300v3) | 教学总体标准差/eps明确约定，不声称所有GRPO实现相同；表格实验使用精确categorical KL，与原语言token采样实现区分 |
| 2.4/4.4奖励与过程监督 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 的结果/过程监督比较；[Reward Hacking](https://arxiv.org/abs/2209.13085) 的代理目标失配 | 不从可验证性推出不会作弊，不从PRM特定实验推出所有任务胜出 |
| 2.5合成管线 | [Self-Instruct](https://arxiv.org/abs/2212.10560) 的生成/过滤/指令学习流程 | 本项目数据是本地原创规则，没有调用教师API，也没有导入其数据 |
| 4.5 OPD | [Thinking Machines On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) 的学生轨迹与Loss function: reverse KL | 具体项目采用的目标与轨迹来源分别说明；本站未运行OPD，不宣传通用成本优势 |
| 4.6推理预算 | [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) 的预算分配与任务难度条件 | 单次/投票/best-of-N/oracle有不同协议，20题算例为人工构造，不复用他人成绩 |
| 5.2裁判偏差 | [LLM-as-a-Judge §3与附录D](https://arxiv.org/html/2306.05685v4) 的位置、冗长、自我偏好和顺序交换实验 | 不从特定裁判结论推所有模型；Elo数值明示为假设映射手算，本站无投票/排名实验 |
| 5.3安全与回退 | [Constitutional AI](https://arxiv.org/abs/2212.08073) 的原则与AI反馈；[InstructGPT附录E](https://arxiv.org/html/2203.02155v1) 的回归/KL讨论 | 不保证任何原则集和KL配置普遍安全、不遗忘；本项目无安全成绩 |
| 5.4训练/生成职责与放置 | [HybridFlow §2.3–5](https://arxiv.org/html/2409.19256v2)，包括角色、resource pool、数据转移与不同并行策略 | 论文描述其当时实现；不将框架旧比较当当前版本排名，未复现verl或生产吞吐 |
| 6.4完整状态与环境限制 | [PyTorch保存/加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[Reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html)，及本站verify.py | 实际运行16项检查，只支持同CPU环境/布局、声明的边界重启；哈希不是签名 |

SFT/DPO曲线、23条生成、表格RL轨迹均来自examples/results内记录，不借论文权威为本站夹具背书。LoRA、完整PPO/LLM GRPO、OPD、裁判、安全、分布式及生产速度没有本站实测。

参考样章通过本地原仓库对应源码完整阅读，来源提交和许可见ATTRIBUTION.md。所有机器人照片/示意图未使用；后训练系统图由本站重新绘制。

## 2026-10-16补充核验

| 新增论断 | 实际阅读的一手支持位置 | 使用边界 |
| --- | --- | --- |
| 概率梯度与动作无关基线 | [Spinning Up Part3](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) 的Deriving the Simplest Policy Gradient、Expected Grad-Log-Prob Lemma、Baselines in Policy Gradients | 本站用二动作枚举核对0.1875与基线抵消，不保证近似critic始终降方差 |
| 奖励模型比较损失 | [InstructGPT §3.5 RM Eq1](https://arxiv.org/html/2203.02155v1#S3.SS5)；[DPO §3 Eq1–2](https://arxiv.org/html/2305.18290v3#S3) | 分数2/1是本站手算并实测梯度，不报告训练过神经RM |
| 参考策略为何出现在DPO | [DPO §3 Eq3、§4 Eq4–7与附录A](https://arxiv.org/html/2305.18290v3#S4) | 条件最优分布关系与有限模型实际优化区分，不从理论最优推出本站实际质量 |
| PPO到组基线 | [DeepSeekMath §4.1.1与Figure4](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1) | 组内均值不能代替准确逐状态critic；组件仅演示一次初始四候选SGD |
| 已有小基座与加载方式 | [SmolLM2固定revision模型卡](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md)、实际下载的配置/tokenizer/权重哈希 | 只采信模型身份与接口；全部本站成绩来自本地report，不借用原卡能力成绩 |
| 多轮工具轨迹的动作边界 | 本站原创消息/字节协议及verify_teaching.py逐位置检查；与4.1的条件动作定义一致 | 不是某真实模型的chat template，不声称执行过工具型LLM RL |

本轮通读参考项目tutorial_writing_principles.md的§4–7、10–13，采用问题先行、分层深度与真实工作用语检验；所有新文字/交互仍用后训练自己的对象。
