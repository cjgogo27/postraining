# 实际验证：《听懂大模型中训练》

日期2026-10-23，基线`7e7de34aa89e7a422b1221ac0110c1e514482125`。开始工作树干净，fetch后本地与远端一致。新书独立放在`midtraining/`，原后训练书不改写为中训练。

## 实验：预定计划、真实数据与失败结果

运行前写`EXPERIMENT_PLAN.md`，匹配新增目标预算、配对seed42/43、独立随机支路顺序、固定终点评测与原始生成协议。没有事后选最好seed/步。以下命令实际执行：

```bash
python midtraining/examples/lab.py --out runs/midtraining-20261023
python midtraining/examples/verify.py --out runs/midtraining-checks-20261023
python midtraining/examples/verify.py --out runs/midtraining-release-checks-20261023
python midtraining/examples/adapt_pretrained.py --download --out runs/midtraining-realbase-final-20261023
python midtraining/examples/plot_results.py
python scripts/check-midtraining-records.py
python -m compileall -q midtraining/examples
```

环境Windows11 10.0.26200、Python3.12.5、torch2.11.0+cpu、NumPy2.0.2。默认CPU FP32/1线程；公开基座4线程，transformers5.8.0、huggingface_hub1.14.0。绘图matplotlib3.9.2。没有本轮全新venv/完整重新安装依赖的验证。

- 默认参数23,040，原创160文档，train128/val32实体组隔离，完全不同文本但共享模板与标签。各源训练流2,176编码位置、验证544；训练含EOS。三个适配支路各96更新、12,288目标。回放实际3,072通用/9,216领域，记录固定验证窗口与源账本。
- 每seed基座120更新/15,360目标；长度接口两支各48更新/6,144目标，源配额1,536/4,608。S32与64接口可运行，未验证长依赖能力。
- 默认完整流程墙钟12.873361秒，包含评测与保存，不能当纯训练性能。两个seed领域/通用固定窗口NLL与曲线已发布，生成均0/4。没有以NLL下降代替任务成功。
- 两次verify均12项PASS，包含连续8与4+恢复4更新的模型/AdamW/采样器/torch RNG/历史/账本/计数精确一致，实际临时源码把学习率乘以1.01后拒绝严格恢复，以及数据/配置/torch/world size/编码不匹配拒绝、新阶段重置与因果性检查。
- 可选公开SmolLM2-135M固定revision`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`，134,515,008独立参数。三支各8更新/1,024目标；原tokenizer/全目标因果LM。保存回放权重和tokenizer后，本地cache-only重载两组NLL误差小于1e-6、两条生成完全相同。生成全0/2；全部支路通用NLL也低于原点，不照搬默认tiny的遗忘结论。
- 公开基座完整流程172.215178秒包含首次下载/加载/训练/评测/保存/重载，不是吞吐基准。先前cache-only尝试因缺缓存失败（`runs/midtraining-realbase-20261023`）；随后显式download成功，有HF未认证与Windows缓存symlink提示，无付费API调用。
- 发布4份JSON是实际runs文件副本；训练源码LF规范化SHA `fc56924e3fc5f7c81778ad741979a9ee4e4244dd0644cd825c9f665b49362a53`、数据SHA `5ee8b4bee983bdbe6052ddb0048cc331f9a36780eb63418c820359a2d6a85406`。`check-midtraining-records.py`实际PASS，另核对公开基座脚本身份、相同起点、源预算、窗口和保留失败输出。

## 文档、构建与浏览器

```bash
npm test
npm run check:docs
npm run build
npm run check:built
npm run preview -- --port 4328
node scripts/check-midtraining-browser.cjs
node scripts/check-browser.cjs
```

- Node测试14/14，原10项加新预算4项。
- 源检查：原书45页/29节/87问答；新书41页/29节/87问答（含6阅读页、4附录、首页/404）。
- 完整双站构建成功：共86 HTML；19处KaTeX、内部目标/锚点/资源缺失0、58组每组三个折叠答案。新书独立Pagefind索引41页。
- 早期独立构建曾因Astro7要求outDir/cacheDir为字符串、Starlight0.42侧栏需items包裹autogenerate而失败，已改为fileURLToPath及新接口后完整通过。构建仍有框架`use astro:head-inject`、空i18n和404路由优先级警告；未隐藏警告。
- Headless Edge155.0.4283.62，Playwright，桌面1280×900与手机宽度390×844，reduced-motion。本地中训练15组检查PASS：两宽度遍历29正文页无文档横向溢出、独立标题/三区总图、角色URL/偏好与折叠答案、预算长短预设/DP/非法值/键盘复位、真实seed切换/失败输出、明暗渲染、原站标题与偏好隔离、无JS静态回退。pageerror为0。
- 原后训练浏览器回归13组PASS：DPO、token/mask、RL步骤、23样本切换、角色问答、主题/溢出及无JS。没有重新训练原后训练模型。
- 额外执行`python examples/verify.py`，原后训练16项CPU检查PASS，涵盖SFT/DPO更新边界恢复、实际代码变化拒绝与有限动作RL恢复；这不等于重新运行原书全部训练实验。
- 已实际查看曲线PNG和新书深色手机总图、手机预算工具、桌面结果组件截图。截图在本地runs，未伪造真实手机/Safari或全页全浏览器截图审查。

最终完整双站build/check:built再次通过；`git diff --cached --check`初检发现Matplotlib生成SVG行尾空白，已在绘图脚本中规范化并重新生成，随后通过。Node v24.14.0、npm11.9.0。

## 引用核对与未测范围

人工论断/段落核对见SOURCE_AUDIT.md。40个独立外部URL/锚点的程序抓取中39成功且有片段者存在；AMP文档一度SSL EOF，随后浏览工具成功读取其unscale/裁剪/累积原文。链接检查不等于论文支持核对。参考机器人网页本轮抓取失败，实际通读固定提交源码及写作原则，未声称原网页截图对照。

未测：CUDA、DDP/TP/PP/CP/EP实际执行、AMP、弹性跨布局、写盘崩溃、生产MFU/性能/费用、真实维修、长上下文任务，以及中训练后继续SFT/RL。公开基座仅一个seed8步可行性短跑；两seed默认夹具不足以作总体显著性推断。代码/文档精确声明这些限制。
