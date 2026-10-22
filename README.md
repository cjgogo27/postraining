# 《听懂大模型后训练》

新增独立教程：[《听懂大模型中训练》](https://cjgogo27.github.io/postraining/midtraining/00-reading-guides/00-outline/)（六章29节、CPU实验与真实小基座短跑）。原后训练书入口与正文保留。仓库完整build现在同时构建两个独立导航/搜索的站点；中训练说明见[midtraining/README.md](midtraining/README.md)。

从监督信号到策略更新，讲清SFT、偏好优化、PPO/GRPO、RLVR、蒸馏、评测与系统接口。面向跨部门工程师、学生、产品/PR；六章29节87个角色问答。

贯穿路线：[RL任务/回报/价值基础](https://cjgogo27.github.io/postraining/00-reading-guides/07-rl-foundations/) → [共同生命周期总图](https://cjgogo27.github.io/postraining/01-system-overview/03-posttraining-system-architecture/) → 各阶段数据/目标 → [执行系统](https://cjgogo27.github.io/postraining/05-evaluation-diagnosis/04-training-rollout-orchestration/) → [阶段交接与综合验证](https://cjgogo27.github.io/postraining/06-tools-practice/04-stage-handoff-and-resume/)。教材/近期综述的对照位置见SOURCE_AUDIT.md，阶段不是固定必经顺序。

- [在线章节与架构](https://cjgogo27.github.io/postraining/00-reading-guides/00-outline/)
- [实际验证记录](VALIDATION.md)与[引用支持位置](SOURCE_AUDIT.md)
- [写作/呈现归属](ATTRIBUTION.md)；文档CC BY-NC-SA 4.0、代码MIT。

## 离线CPU实验

先创建并激活Python虚拟环境，再安装依赖。准备依赖后，运行不下载模型/语料或调用API。

```bash
python -m pip install -r examples/requirements.txt
python examples/prepare_data.py --out runs/data
python examples/train.py --data runs/data --out runs/sft
python examples/train.py --stage dpo --data runs/data --out runs/dpo --steps 120 --lr 0.0001 --init runs/sft/latest.pt --eval-every 40
python examples/rlvr.py --out runs/rlvr
python examples/verify.py --out runs/checks
```

68,608参数语言模型随机初始化，只验证机制，不是成熟基座后训练复现；48参数有限动作RL不是语言模型GRPO。每个新运行使用新out目录。原始指标和输出在[examples/results](examples/results/)，大权重保留runs/。

实际SFT终点1/23严格生成正确，DPO终点0/23，尽管固定pair排序提高；不隐藏回退，不用表格策略12/12宣传数学泛化。LoRA、完整PPO/LLM GRPO、OPD、裁判、安全与多机只讲机制/方案，没有本站成绩。

## 文档站

Node.js24，Astro7.3.1 / Starlight0.42.0 / KaTeX，支持侧栏、目录、搜索、角色问答、token/mask对齐、RL分步演示、样本对照和DPO手算器。

```bash
npm ci
npm run check:docs
npm run build
npm run check:built
npm run dev
```

打开终端地址下/postraining/。GitHub Pages通过.github/workflows/pages.yml发布。该站借鉴指定机器人教程的教学组织与文档体验，后训练正文和图示重新编写；不套用机器人实时频率或纵向指令层级。

## 教学修订：先理解，再迁移

[必要基础](https://cjgogo27.github.io/postraining/00-reading-guides/05-essential-foundations/)接通token、概率、交叉熵、梯度、采样与期望。算法章节补动机推导，新增奖励模型训练链和多轮工具动作/观察接口；保留六章29节87问答。网页增加token/mask对齐、RL四步演示、固定23题前后对照、DPO情境预设与阅读视角。

可选已有基座路线（与默认离线实验分开）：

```bash
python -m pip install -r examples/requirements-pretrained.txt
python examples/pretrained_sft.py --download --out runs/pretrained-demo
python examples/verify_teaching.py --out runs/teaching-checks
npm test
```

固定SmolLM2-135M revision的CPU SFT实际24步；6个新订单ID的同模板状态题从0/6到6/6严格成功。另保留3个基础续写探针，不能据此推断客服泛化或原能力全面保留。奖励模型与工具轨迹本轮是算例/接口夹具，没有训练神经RM或工具型LLM RL。完整证据见VALIDATION.md与examples/results。

浏览器QA入口为`scripts/check-browser.cjs`，需另行准备Playwright/Chromium并启动本地preview；可用PLAYWRIGHT_MODULE、BROWSER_EXECUTABLE指定已有运行时，不属于默认训练依赖。

## RL专题补充（2026-10-22）

[RL地图](https://cjgogo27.github.io/postraining/00-reading-guides/06-rl-map/)区分在线/离线与on/off-policy；4.2补两步GAE、critic目标、PPO四种剪裁和批量计数，4.3补RLOO/DAPO/GSPO，4.5补OPD梯度、OPSD特权教师与Semi-OPD。[附录E](https://cjgogo27.github.io/postraining/07-appendix/e-agentic-multi-agent-rl/)展开agentic/MARL，[附录F](https://cjgogo27.github.io/postraining/07-appendix/f-rl-reading-list/)整理10份重点一手资料，注明日期、版本和具体支持位置。保持六章29节87问答。

```bash
python examples/rl_mechanisms.py --out runs/rl-mechanisms
```

12项CPU FP64机制/梯度检查已经执行，报告在examples/results/rl-mechanisms-reference.json；没有新增完整语言模型PPO/GRPO、agent/MARL或OPD/OPSD训练成绩。原历史训练结果保留，当前验证范围以VALIDATION.md为准。

## 后训练生命周期与系统综合练习

```bash
python examples/posttraining_lifecycle.py --out runs/lifecycle
python examples/posttraining_lifecycle.py --ddp --out runs/lifecycle-ddp
```

沿用PyTorch依赖，每次新建out目录。12参数表格连接SFT→DPO→RL，检查阶段状态、固定参考、准入/提交、采样身份与边界恢复；`--ddp`另启两个真实本机CPU/Gloo进程，在固定PPO张量上核对有效位置2/4时的全局归约。报告为examples/results/lifecycle-system-reference.json，CPU/FP64，不是LLM分布式训练或生产RPC/队列性能。
