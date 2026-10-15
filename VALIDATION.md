# 实际验证记录：《听懂大模型后训练》

日期：2026-10-15（Asia/Shanghai）。本书为独立postraining仓库，六章29节87问答、三类阅读路线、阶段交接导读、四页附录与系统总图。写作前已通读指定样章的原仓库对应源码；网页工具访问该地址失败，未声称做过原站截图比较。来源提交与许可见ATTRIBUTION.md。

## 环境与执行身份

- Windows11、Python3.12.5、PyTorch2.11.0+cpu、FP32、每进程CPU线程1、seed42。无CUDA或外部API调用。
- Node24.14.0、npm11.9.0，Astro7.3.1 / Starlight0.42.0 / KaTeX0.18.7。实际npm ci安装374包；锁文件与Rolldown1.2.13 override固定，训练仅依赖torch。
- 仓库开始为README-only，基线HEAD为3d3e484136b8e4079ba328f9b1792dbf135d8917；执行时有本轮未提交内容。
- 语言模型训练身份：ce985a0f19a482a4090b220f96390992f4c9f39c63b91bf6fa31979053b76c7a，绑定规范化postcore.py、train.py及文件名。
- 有限动作RL身份：2531d44bcdb3f7ac9c3a9f017ba5d79cb5006a68a04433d1725f88760d196d37，绑定postcore.py与rlvr.py。
- 配置、输入身份、原始输出和指标见[examples/results](examples/results/README.md)。完整快照/日志在忽略的runs目录，不上传普通Git。

## 实际语言模型实验

    python examples/prepare_data.py --out runs/final-data
    python examples/train.py --data runs/final-data --out runs/final-sft
    python examples/train.py --stage dpo --data runs/final-data --out runs/final-dpo --steps 120 --lr 0.0001 --init runs/final-sft/latest.pt --eval-every 40

TinyPolicy实际68,608独立参数：257字节词表、EOS256、width64、1个4头Pre-LN因果Transformer、FFN256、位置表32、共享输入输出、dropout0。随机初始化，不是生产预训练基座。数据为本地原创0–9加法；交换加数同组，组SHA-256划分为77条train/43组、23条val/12组，仍共用模板/数字，训练侧答案覆盖也有局限。

| 阶段/终点 | 消费记录 | 回答目标位置 | 验证回答NLL | pair排序正确 | greedy严格正确 |
| --- | --- | --- | --- | --- | --- |
| 随机起点 | 0 | 0 | 5.528908509474534 | 15/23 | 0/23 |
| SFT300 | 2400示范 | 6061 | 1.1132530065683217 | 14/23 | 1/23 |
| 新DPO120 | 960对 | 4927（两回答） | 1.9368952971238356 | 17/23 | 0/23 |

SFT默认B8、lr0.001；DPO B8、lr0.0001、β0.5；AdamW统一weight_decay0.01、梯度裁剪1、无调度/AMP。参考冻结为SFT终点，新DPO重建优化器/随机状态/计数。DPO用原始回答联合logp求和、按pair平均；SFT按有效回答目标位置平均。位置包含EOS和重复消费，不等于唯一数据或全部实际计算量。

验证回答共52个目标位置，固定23题、greedy最多4个新字节、自然EOS且规范文本精确相同。DPO80步曾2/23正确，最终120步0/23；没有选择最佳步替换预定终点。全部23条原始输出和逐点记录发布，未人工润色失败答案。

成功条件是产物、计数、有限数值与接口验证正确，不要求能力变好。此单种子、小模板实验不支持DPO普遍优劣、数学泛化或生产助手结论。

## 实际有限动作RL

    python examples/rlvr.py --out runs/reference-rlvr

12×4 logits、48参数表格策略，每题四个固定候选、无语言生成或未见题。每轮每题采8次、96动作；组总体标准差/eps1e-8、clip[0.8,1.2]、精确KL(policy||uniform)系数0.02、Adam lr0.05、每轮2次更新。

- 实际80轮/160次更新、累计7680动作；初始正确候选期望概率0.25，最终0.990900456905365，训练题greedy选择12/12。
- 最后一轮actor版本158、采样reward均值0.9895833134651184、恒定奖励组11/12，两次epoch clip fraction均0；没有把“启用clip”写成每轮都触发。
- 全轮记录和第一组动作/old概率/奖励/优势在rlvr-reference.json中；精确期望概率与某轮抽样均值分开。
- 只说明有限动作组相对更新，不是完整语言模型GRPO或数学推理成功率。

## 16项实际CPU验证

    python examples/verify.py --out runs/checks

退出码0，status=PASS，16项全部通过，完整名称及身份在checks-reference.json：

1. 真实byte ID、右移、回答mask与EOS位置。
2. 改被屏蔽labels不影响SFT loss。
3. 改未来输入不影响历史logits。
4. 变长回答联合logp含EOS，均匀词表手算匹配。
5. DPO初始ln2、m=0.2/loss=0.598138869的手算与梯度方向。
6. 组总体标准差及常数reward优势全0。
7. PPO正/负优势的clip手算。
8. 严格答案验证器的额外数字、格式和空白反例。
9. SFT8步连续与4步暂停恢复：模型、AdamW、随机/采样状态、计数相同。
10. DPO8步连续与4步暂停恢复：模型、冻结参考、AdamW、随机/采样状态、计数相同。
11. 新DPO的参考仍为来源SFT，优化器计数按新阶段重建。
12. 改训练计划拒绝恢复。
13. 改PyTorch版本或world size拒绝恢复。
14. 实际lr代码乘1.01，因源码身份变化拒绝恢复。
15. 改真实JSONL数据拒绝恢复。
16. 有限动作RL8轮连续与4轮保存恢复：logits、优化器、RNG、计数及历史记录一致。

语言模型只覆盖更新边界，RL只覆盖完整rollout轮边界；不是中途写盘故障、异步队列或多卡容错。检查点含Python序列化对象，仅加载可信文件。

## 文档、公式与交互逻辑

| 实际执行 | 结果 |
| --- | --- |
| npm run check:docs | PASS：40源页面、29节、87角色问答，正文约23940汉字（不含独立JSON问答） |
| npm run build | PASS：40 HTML、Pagefind搜索、sitemap |
| npm run check:built | PASS：12个KaTeX实例无解析错误，内部资源/目标/锚点缺失0，29组三答案结构完整 |
| node标量DPO计算器测试 | 8例通过：手算、ln2、负margin、非法logp/β、NaN、溢出拒绝及大值稳定计算 |
| python -m compileall -q examples | PASS |
| 产物核对脚本 | 当前源码身份、数据SHA、68,608参数、终点/消费计数、52验证位置、输出计分、7680动作、160更新、15个CLI默认值全部通过 |
| 独立手算 | 3.2加权loss、DPO0.598139、总体标准差优势、KL0.09151622及版本/计数与正文相符 |
| 图示与曲线 | 总图自编；两图直接由实际JSON绘制，PNG独立打开检查标题/轴/图例与数值，无模型结果美化 |
| 关键引用 | 已阅读SOURCE_AUDIT表列主来源与支持位置，未用HTTP可达代替论断核验 |
| git diff --check | PASS |

构建仍有依赖级use astro:head-inject等警告和既有404路由优先级提示，无构建失败。未新增训练平台或外部数据依赖。

## 实际发现与修复

- 初次数据划分公式意外按和模5留出整类答案，导致全部验证生成错误。检查标签分布后改为规范组哈希，重新生成/训练SFT与DPO；最终JSON只用新划分，正文2.1保留原因。没有把旧结果混入正式曲线。
- 初次源文档检查缺少尚未生成的曲线资源，生成实际图后重查通过；初次构建产物检查缺favicon，补资源并重建后通过。
- 产物核对脚本初版把Markdown分隔行误当CLI字段，修正匹配为字母开头后重新核对15默认值；没有将失败尝试记为通过。

## 未运行和不能推导的范围

- UI工具只有空MCP Apps，无普通网页标签；未进行网站桌面/移动截图、折叠点击、搜索/主题或计算器DOM输入交互验收。图像预览与标量JS检查不替代这些检查。
- 未运行真实预训练基座的后训练、LoRA、AMP、DDP/CUDA、多机/分片、完整自回归LLM PPO/GRPO、OPD、外部教师/裁判、安全数据集、生产异步编排、GPU性能或成本对照。
- 没有多种子显著性、独立产品test、训练长度/数据/奖励修复消融。未执行的方案只写作机制或设计。

GitHub Pages发布状态以本轮工作流和发布后内容核对为准。
