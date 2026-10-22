# 实际验证记录：《听懂大模型后训练》

## 2026-10-22后续：把RL基础、各阶段目标与执行系统接成一条链

本轮基线`1b14cda41dbb609ffbfd21f383efea4fb477cde0`；开始工作树干净，fetch后远端一致。六章29节87角色问答保留；新增一页RL任务/回报/价值基础，共45页。此记录是同日后续修订，下面较早记录中的未运行范围按各自时间理解。

### 解释与系统连接

- 对照Spinning Up、Nathan Lambert的RLHF Book/实现课程、2026统一后训练与GFCR综述、AReaL v5及PyTorch2.11文档。人工阅读支持位置在SOURCE_AUDIT.md，近期综述用作分析框架，不当作普遍定理或固定recipe。
- 新基础页连接任务/观察/策略、G/V/Q/A、Bellman/MC/TD、GAE与探索；1.3和SystemMap把示范、偏好、交互结果、教师反馈接入共同生命周期，区分学习反馈与实验决策。
- 阶段导读、4.2、5.4和6.4沿输入/形状/分母/更新/发布/保存/验收接力。5.4补DP/分片/TP/PP、两个同步过程、AMP与各优化器成功计数、队列背压/数据年龄、KV身份及预算口径。
- 8轨迹/28有效位置/40填充位置/4更新/56次重复消费是一组自编计数；固定PPO两窗口的2/4位置是独立数值子例。36GB静态状态和128MiB KV均为假设预算，没有GPU实测。

### 新生命周期与真实两进程数值对照

```bash
python examples/posttraining_lifecycle.py --ddp --out runs/lifecycle-integration-final-20261022
python examples/posttraining_lifecycle.py --out runs/lifecycle-single-final-20261022
```

两种模式都实际退出0，各10项PASS。Windows11、Python3.12.5、torch2.11.0+cpu、FP64/单进程内线程1；复用既有依赖，没有安装生产训练平台。源码按LF规范化SHA为`b57ba9fd79860d463a87de0c415ca24c7d39f5f4c671b7e4bcd094f05e33db20`，与发布报告绑定。

- 12参数表格、4上下文/3候选：SFT4更新/16标签消费；DPO3更新/12pair消费，参数继承且AdamW重建、参考冻结；新RL2轮/32采样动作/4更新。采样actor0/2，发布2/4，复用期old logp不变。采样是有放回，不称4个不同候选。
- 准入拒绝重复提交、过旧、未来版本、评分器变化、超时和空批次。第一轮完成后实际保存并重载，第二轮的模型、AdamW、参考、专用Generator、动作记录、提交清单与计数精确相同；改stage/code身份/world size拒绝。
- 单独固定logits[2,4,3]的PPO子例：两个真实本机Gloo进程，各处理一条有效位置2/4的窗口，按默认DDP梯度平均语义使用全局分母6。梯度和一次SGD更新与单进程的最大绝对差均0，容差atol=1e-12/rtol=1e-10；错误平均局部均值的梯度差0.05694593956058974。
- 另核对订单Bellman算例、奖励/β尺度、架构计数与显存假设的算术。没有给这些数值加上模型能力含义。

实际报告为`examples/results/lifecycle-system-reference.json`；默认单进程报告与回归报告在`lifecycle-regression-reference.json`。中途补了未来版本反例和参数来源哈希后，用新目录重新执行，正式报告采用最终源码。Gloo有Windows地址族连接重试提示，两个进程最终正常完成；未把提示当作成功条件本身。

### 本轮实际回归、构建和显示检查

| 实际检查 | 结果 |
| --- | --- |
| python examples/verify.py --out runs/coherent-core-checks-20261022 | PASS：原16项数值/协议/恢复/变化拒绝检查 |
| python examples/verify_teaching.py --out runs/coherent-teaching-checks-20261022 | PASS：原5项手算/梯度/消息mask |
| python examples/rl_mechanisms.py --out runs/coherent-rl-mechanisms-20261022 | PASS：原12项RL机制检查 |
| 原训练源码与基线逐文件diff | postcore/train/rlvr/pretrained_sft/rl_mechanisms均未变，历史训练结果未重跑 |
| python -m compileall -q examples scripts/check-rl-sources.py | PASS |
| npm test | PASS：10项JS |
| npm run check:docs | PASS：45页、29节、87问答，正文约5.15万汉字，不含问答JSON |
| npm run build / npm run check:built | PASS：45 HTML、搜索与sitemap；18个KaTeX实例，无缺失内部资源/目标/片段，29组三角色答案完整 |
| 浏览器3套脚本 | PASS：新增13组与原13+13组共39组，页面JS异常0；详见lifecycle-browser-reference.json |
| 新来源元数据/片段抓取 | 8项最终确认，保留AReaL页面超时的原PARTIAL与单项重试HTTP200/S4存在记录；lifecycle-source-reference.json不代替人工论断核验 |
| git diff --check | PASS；新文件暂存后再次检查 |

浏览器为已有Playwright/Edge155.0.4283.62，实际构建产物preview端口4328；1280×900、390×844，深浅主题、四入口/两反馈总图、基础/阶段/系统/综合实践、折叠与无JS回退。实际打开桌面浅色/移动深色总图和移动预算截图；初次元素截图受固定导航遮挡，改为只在截图时隐藏fixed/sticky导航后重拍，界面断言仍使用原布局。截图在忽略的runs目录。既有head-inject/i18n/404提示不导致构建失败。

### 范围限制

生命周期训练是12参数候选表格，权重确认只复制本地快照，没有网络RPC、生成端重新分片、真实工具/异步队列；恢复只覆盖完成轮边界与专用采样Generator，没有dropout、崩溃、写盘中断或多文件原子提交测试。两rank检查是独立固定张量的梯度/一次更新，不是DDP恢复或完整LLM PPO。没有GPU/AMP/FSDP/TP/PP运行、AReaL/verl生产实验、吞吐/显存/费用实测，也没有重训历史小LM、已有基座或OPD/OPSD。未验收所有浏览器、全站页面、搜索或真机触控。

---

## 2026-10-22：RL分类、机制、Agent与蒸馏补充

本轮基线`43cdcdcbfae46c2a262b322ba4f3d4d7ae9af7c8`，工作树开始干净，fetch后远端main一致。先阅读现有RL六节、编排/实践、问答与历史记录，再检索一手资料。六章29节87问答保留；新增RL地图、Agent/MARL扩展与10份资料导读，共44页面。原postcore.py/train.py/rlvr.py/pretrained_sft.py与基线内容逐文件核对未改。

### 内容与引用

- 在线/离线与on/off-policy分轴，给出缺覆盖反例及Bellman自举的直觉；PPO展开GAE、终止/截断、价值目标、四种clip与批量计数。
- GRPO连接RLOO、DAPO与GSPO，区分长度归约、组标准化、保留样本分布与真实采样成本；Agent/MARL补动作/观察、角色、共享参数、CTDE与条件反事实基线。
- OPD区分前向KL估计与梯度，OPSD明确特权信息、同参数教师和停止梯度；加入近期Semi-OPD条件化对照，不外推普遍优劣。
- 10份重点资料的标题/首作者/版本与具体支持位置见附录F和SOURCE_AUDIT.md。选用2025–2026论文/官方文档，PPO/GAE/offline RL/RLOO/MAPPO基础另外列出，不称全网最新排名。

### 实际CPU与JS检查

环境沿用Windows11、Python3.12.5、torch2.11.0+cpu；未新增训练依赖。Node24.14.0/npm11.9.0及既有Astro7.3.1/Starlight0.42.0用于构建；本轮未声称重新安装。

| 实际运行 | 结果与产物 |
| --- | --- |
| python examples/rl_mechanisms.py --out runs/rl-mechanisms-20261022 | PASS：12项CPU FP64/单线程数值与梯度夹具；rl-mechanisms-reference.json，源码SHA7655ef38f6affc645b67bbda36f904a2a4cc0e42e247d88f54ba12e4a1d72590与当前文件一致 |
| python examples/verify.py --out runs/checks-20261022 | PASS：原16项，含SFT/DPO/有限动作RL的边界恢复和代码/数据/配置/环境变化拒绝；rl-regression-20261022.json保留原报告 |
| python examples/verify_teaching.py --out runs/teaching-checks-20261022 | PASS：原5项手算/梯度/消息mask；同一回归报告内保留 |
| python -m compileall -q examples | PASS |
| npm test | PASS：原7项与新增3项PPO四区间、有限差分、输入检查，共10项 |

12项新增检查实际核对：GAE优势0.78/0.4、截断自举与终止δ0.2/−0.6、四区间目标/梯度、IS目标0.6与截权0.48、缺覆盖的0.3/0.8、两种长度归约、几何ratio与token剪裁、RLOO、自编固定状态KL梯度与朴素反传抵消、教师停止梯度、双策略精确梯度0.09375/0.0625、固定构造回报LOO。全部为独立构造的小分布，容差atol=1e-12/rtol=1e-10；没有训练神经策略或执行真实反事实工具重放。

### 构建、引用链接与实际浏览器

| 实际执行 | 结果 |
| --- | --- |
| npm run check:docs | PASS：44页、29节、87角色问答，正文约4.25万汉字，不含独立问答JSON |
| npm run build | PASS：44 HTML、Pagefind与sitemap；既有head-inject/i18n/404提示，无最终构建错误 |
| npm run check:built | PASS：17个KaTeX实例无解析错误；内部资源/链接/片段缺失0，29组三角色答案完整 |
| python scripts/check-rl-sources.py --out runs/rl-source-checks-final-20261022，及单项补查 | 10份资料标题/作者通过；14个外部片段最终全部确认。批次保留GSPO一次抓取超时的FAIL，单项重试HTTP200并确认S4存在；合并报告rl-source-checks-reference.json保留两份实际结果，不把超时改写成批次通过 |
| scripts/check-rl-browser.cjs | PASS：新增13组，PPO四情境/滑块/键盘、深浅主题、角色折叠、GSPO公式折叠、三新页面与10条资料、关闭JS静态回退 |
| scripts/check-browser.cjs | PASS：原13组重新运行，DPO/字节mask/RL分步/23样本对照/角色/主题/无JS回退 |
| git diff --check | PASS；新文件暂存后再次检查 |

浏览器使用已有Playwright与本机Edge155.0.4283.62，preview端口4327，视口1280×900和390×844，headless、reduced-motion。两组报告都无页面JS异常；分别保存为rl-browser-reference.json与browser-regression-20261022.json。实际打开新增PPO桌面浅色和移动深色组件截图核对，无截字或控件错位；截图留在忽略的runs目录。这是桌面浏览器的移动视口，不是真机。

初次构建遇到details内部的跨行数学定界解析失败，改成合法单行公式后重建通过，并去除中文数学下标警告。外部抓取初次有verl/GAE超时，批次重跑后二者通过、GSPO仍超时；随后单项抓取成功，保留失败与补查证据。URL或片段存在只验证地址，科学论断仍按SOURCE_AUDIT.md的原文位置核对。

### 本轮未运行的范围

没有重训历史300步SFT、120步DPO、80轮表格RL或24步已有基座SFT，其原负结果与原始报告保留。没有完整LLM PPO/GRPO、critic/RM、agentic/MARL、CTDE、C3、OPD/OPSD/Semi-OPD神经训练，没有CUDA/DDP/多机/生产异步系统、吞吐或成本成绩。新增脚本没有优化器训练循环。未做全部页面、搜索、Firefox/Safari或真机触控验收。

---

## 2026-10-16：教学主线与真实任务连接修订

本轮基线HEAD：1dff43b2f6c189a652bed88cd127b61e8a70ebf0。先核对全部正文、阅读路线、附录、87问答、原代码与首版验证记录，再阅读参考样章及tutorial_writing_principles.md；没有只根据目录推断缺失。保留六章29节，增加一页必要基础，构建共41页。

### 本轮实际完成

- 必要基础连接token、条件/序列概率、交叉熵、梯度、采样与期望；1.3先解释监督对象与工单应用，字节ID留在2.2。
- 3.1/3.4/4.2/4.3/4.5先讲动机与方法联系，再讲精确目标。2.3补显式奖励模型比较损失与独立验证；4.1补多轮动作/观察/延迟奖励。
- 新增token/mask视图、RL四步演示、固定23题SFT/DPO输出对照；DPO预设含当前等于参考、两候选绝对概率都降。均为浏览器本地计算，不调用模型服务。
- 角色阅读视角可选择并本地记忆，其余问题折叠；29节补具体交接关系与工作用语自检，阅读路线/术语/模型卡/来源同步。

### 实际CPU检查与文档构建

| 命令 / 检查 | 实际结果 |
| --- | --- |
| npm run check:docs | PASS：41源页面，29节，87角色问答；中文字符计数约3.28万，不含独立问答JSON |
| npm test | PASS：7个JS测试，含DPO情境/非法输入、真实字节对齐、组梯度/常奖励和基础SGD |
| npm run build | PASS：41 HTML，Pagefind、sitemap；仍有依赖级head-inject、i18n及404路由警告，无构建错误 |
| npm run check:built | PASS：14个KaTeX实例，内部资源/链接/锚点缺失0，29组三角色答案完整 |
| python examples/verify.py --out runs/checks-20261016 | PASS：原16项CPU检查全部重跑，含SFT/DPO严格恢复、改代码/数据/计划/环境拒绝与有限动作RL边界恢复 |
| python examples/verify_teaching.py --out runs/teaching-checks-20261016 | PASS：5项CPU FP64手算/梯度/消息接口检查；生成tool-trace.json |
| python -m compileall -q examples | PASS |
| git diff --check | PASS（新增文件纳入暂存后再次检查） |

原训练核心postcore.py/train.py/rlvr.py未变；本轮恢复检查的源码身份与首版一致。原300步SFT/120步DPO及80轮表格RL曲线没有重新训练，也没有改写其原始结果。本轮独立数值：目标概率0.2经示意SGD变为0.22031334；奖励模型分数2/1对应偏好概率0.73105858、loss0.31326169；二动作期望梯度0.1875；四候选一步概率为0.225083/0.225083/0.274917/0.274917。全部由代码实际核对，不是新能力成绩。

### 真实预训练基座的可选SFT

```bash
python examples/pretrained_sft.py --download --out runs/pretrained-revision-20261016
python examples/verify_pretrained_artifact.py --run runs/pretrained-revision-20261016 --out runs/pretrained-artifact-checks-20261016
```

- 实际下载SmolLM2-135M，revision93efa2f097d58c2a74874c7e644dbc9b0cee75a2；134,515,008独立参数。权重/tokenizer/config逐文件SHA在pretrained-reference.json。
- Windows11、Python3.12.5、torch2.11.0+cpu、transformers5.8.0、huggingface_hub1.14.0；FP32、CPU4线程、seed42；所有依赖运行前已可用，本轮未声称重新安装它们。
- 原创英文状态提取模板，18条训练；6个新订单ID但同模板的目标探针；3条基础续写探针。无chat template，prompt与回答分别编码后拼接，回答及EOS计loss；无静默截断。
- 全参数AdamW，lr2e-5、weight_decay0.01、裁剪1、每步2条、固定24更新；消费48条记录/96个回答目标位置。
- 目标严格成功0/6→6/6，回答NLL6.08691470→0.00015670（12位置）；基础续写目标NLL2.38872576→1.98072970（12位置）。基础生成仍不等于目标且未自然结束，3条不足以判断全面保留能力。
- 保存后实际cache-only本地加载：参数量和源码身份吻合，全部9条生成token/文本/结束原因相同，两组NLL误差小于1e-6。不是优化器恢复或跨硬件验收。
- 脚本墙钟273.2秒，含首次下载/重试、评测、训练和保存；不含依赖安装/计时前导入。下载发生read timeout后续传成功。未测峰值内存、GPU吞吐或费用，不将墙钟当纯训练速度。
- 本轮只有一个预定计划与seed。6/6只支持同模板提取行为变化；没有客服泛化、工具调用、安全或大模型RL成绩。大权重留在忽略的runs/，只发布报告与原创任务。

### 本地浏览器检查

使用工作区现有Playwright与Chromium145.0.7632.6，无新增生产依赖。启动实际构建产物预览：npm run preview -- --port 4325；运行scripts/check-browser.cjs（本机通过PLAYWRIGHT_MODULE和BROWSER_EXECUTABLE指向现有运行时）。默认报告在runs/browser-checks；本次用BROWSER_CHECK_OUT=runs/browser-final-20261016。

实际1280×900、390×844两个viewport：角色query/记忆与答案折叠；DPO预设、非法输入、复位与键盘激活；token切换/右移/EOS/可见范围；RL四步/全同奖励/复位；23题固定输出选择；深浅主题与页面横向溢出检查；另测关闭JS的内容回退。共13组PASS，页面JS异常0。报告为browser-checks-reference.json。

实际打开组件截图检查，修正了移动端表头挤压、对照卡片错位和表格首列定位；最终RL移动端为四张候选卡。截图留在runs目录。测试脚本最初错误假设只有一个主题控件，随后又选到隐藏控件，已改为可见控件并通过移动菜单操作；截图还曾被打开的菜单遮挡，现已在关闭菜单后重拍。没有把这些失败尝试计作通过。

源链接检查最初把?reader=参数当作路由，已正确忽略query后重跑通过；手算概率初稿的0.2202按实际计算修为0.2203。未据此修改原训练结果。

### 当前仍未运行的范围

完整语言模型PPO/GRPO、神经奖励模型、工具型LLM训练、OPD/外部裁判、安全、LoRA、CUDA/DDP/分片/多机与生产编排仍未运行。多轮只运行原创消息mask夹具；RM只运行标量损失和梯度。未做全部页面/搜索的浏览器验收，也未测Safari、Firefox、真机触控或性能。原有负结果不因新基座的小任务成功而失效。

关键来源的具体支持位置补在SOURCE_AUDIT.md，InstructGPT奖励模型训练定位为§3.5，DPO推导定位§3–4与附录A；不把HTTP成功当作支持核验。

---

## 2026-10-15首版记录（历史，以下“未运行”仅指当时）


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
- 新文件纳入Git暂存检查后发现Matplotlib SVG大量行尾空格，已清理；XML解析后的全部元素、属性（空白规范化）及文本一致，图形路径保持一致。随后重新检查新增/已跟踪差异并重建。

## 未运行和不能推导的范围

- UI工具只有空MCP Apps，无普通网页标签；未进行网站桌面/移动截图、折叠点击、搜索/主题或计算器DOM输入交互验收。图像预览与标量JS检查不替代这些检查。
- 未运行真实预训练基座的后训练、LoRA、AMP、DDP/CUDA、多机/分片、完整自回归LLM PPO/GRPO、OPD、外部教师/裁判、安全数据集、生产异步编排、GPU性能或成本对照。
- 没有多种子显著性、独立产品test、训练长度/数据/奖励修复消融。未执行的方案只写作机制或设计。

GitHub Pages发布状态以本轮工作流和发布后内容核对为准。
