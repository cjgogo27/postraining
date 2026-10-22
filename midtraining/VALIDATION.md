# 实际验证：《听懂大模型中训练》

## 2026-10-23第二轮：联网深读、逐节解释与新增诊断

本轮基线`a3d672a026dfca1229db8f6e86266aeafa3c6b93`，起始工作树干净；发布前再次fetch，origin/main仍为该提交。下文第一轮实验记录保留其原版本/时间含义。

### 内容与来源

- 扩写29个正文小节全部，另补必要基础、阶段连接与3个附录；保持六章29节87问答、41页。21条角色答案随新解释增强。源检查中文字符统计23,461→37,629（约增加60%，包含页面标题/引用等，不当成纯正文词数）。原后训练正文没有修改。
- 实际联网查询并阅读arXiv论文及官方资料，主要支持位置在SOURCE_AUDIT.md与附录D。新增Midtraining Bridges…的v2（2026-02-02）、DoReMi、BigCode近似去重、语言适配与PPL/ICL、FlashAttention、PyTorch Profiler等；既查元数据，也核对具体论断。没有用搜索摘要代替所有原文支持。
- 将配比/加入时机、全文/关键区域/候选/自由生成、source token/专家assignment、可加载/数学等价/容差/位级一致分开。矩阵、Jaccard、梯度、流水线和消融为自编手算或设计，不列为集群实测。
- 外链检查初次发现GitHub标题片段未出现在HTML id中，改为原文链接加明确章节名；修正后扫描57个URL/片段，其中ProLong同一页面的4个URL因一次读取超时未完成，随后单页重试HTTP200且S2/S3/S6锚点存在，其余通过。HTTP检查与人工论断核对分别记录。
- HF Transformers v5.8.0 PPL页面本轮无法通过浏览工具打开，实际改读官方当前页；PyTorch保存加载教程当前显示2.14，其余主要API固定2.11，与实际运行2.11环境区分。

### 新增代码与实际执行

```bash
python midtraining/examples/diagnose.py --run runs/midtraining-20261023 --out runs/midtraining-diagnostics-detail-20261023
python midtraining/examples/profile_cpu.py --out runs/midtraining-profiler-detail-20261023
python midtraining/examples/verify_diagnostics.py --out runs/midtraining-diagnostics-oracle-20261023
python scripts/check-midtraining-records.py
python -m compileall -q midtraining/examples scripts/check-midtraining-records.py
```

- diagnose实际PASS：原两seed八个状态，核对旧lab/数据及每个权重身份；每源前四个留出实体，逐篇33目标，前缀104、答案含EOS28。领域与回放支路关键区域NLL也改善；八状态各源四候选皆1/4，原自由生成0/4保持。表和逐候选logp发布于diagnostics-reference.json。这是事后解释性分析，不是新的预定主指标、独立test或追加训练。
- 新诊断独立恒定logits手算检查5项PASS：CE=ln257、区域104/28分母、各7目标候选logp、并列候选的固定选择、函数调用不改变torch RNG。该夹具不代表真实模型能力。
- CPU profiler实际PASS：新随机模型、固定lr0.001，2次预热后记录4更新/512目标；四类人工区间各4调用。trace.json约2.20MB，实际解析确认16条人工完整区间；汇总发布profiler-reference.json。CPU累计时间见6.1，不能换算GPU吞吐或MFU。
- profiler发出“每个cycle末尾清理事件”的通用警告；该次没有多个记录cycle，4次调用与导出16事件均已核对。未声称在时间线UI中人工操作检查。
- check-midtraining-records扩展后实际PASS：含新脚本SHA、旧权重身份、区域分母、候选计数、CPU记录数与oracle。原lab.py、adapt_pretrained.py、verify.py及原训练JSON与基线无diff；本轮未重跑原配比/真实基座训练，也未重跑原后训练训练实验。

### 文档与显示检查

```bash
npm test
npm run check:docs
npm run build
npm run check:built
node scripts/check-midtraining-browser.cjs
```

- Node14项PASS。源检查原站45页/29节/87问答，新站41页/29节/87问答；双站86 HTML、19 KaTeX、缺失内部目标/锚点0、58问答组通过。
- 一次构建在Jaccard示意的未转义集合花括号处报MDX ReferenceError，已改为行内代码；随后完整双站构建通过。原框架指令/i18n/404警告仍存在。
- Headless Edge155.0.4283.62，1280×900及390×844，reduced-motion；初次17组PASS。实际查看诊断表截图发现手机表头逐字竖排，添加仅中训练站的手机表头nowrap，使用表内横向滚动；修复后19组PASS。
- 最终19组涵盖两宽度遍历29正文与10阅读/附录页无页面横向溢出、新基础折叠区键盘展开/关闭、角色/预算/seed/原负输出、明暗、原站标题与偏好隔离及无JS。手机实际操作表内横滚并检查右侧0/4列；已查看桌面、修复前/后手机表截图。没有真实手机/Safari、线上浏览器或所有页面截图覆盖的宣称。

新增脚本使用同一Python3.12.5/torch2.11.0+cpu环境，默认CPU1线程；未新建安装环境。未新增CUDA、多rank、TP/PP/CP/EP、AMP、弹性故障恢复、真实长依赖、生产性能与后续SFT/RL验证。上述解释和设计不填成绩。

---

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
