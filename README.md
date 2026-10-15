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

Node.js24，Astro7.3.1 / Starlight0.42.0 / KaTeX，支持侧栏、目录、搜索、角色问答和DPO手算器。

```bash
npm ci
npm run check:docs
npm run build
npm run check:built
npm run dev
```

打开终端地址下/postraining/。GitHub Pages通过.github/workflows/pages.yml发布。该站借鉴指定机器人教程的教学组织与文档体验，后训练正文和图示重新编写；不套用机器人实时频率或纵向指令层级。
