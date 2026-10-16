# 《听懂大模型后训练》

从监督信号到策略更新，讲清SFT、偏好优化、PPO/GRPO、RLVR、蒸馏、评测与系统接口。面向跨部门工程师、学生、产品/PR；六章29节87个角色问答。

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
