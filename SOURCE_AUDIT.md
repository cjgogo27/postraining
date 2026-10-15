# 关键论断与来源支持位置

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
