# 《听懂大模型中训练》

[目录](https://cjgogo27.github.io/postraining/midtraining/00-reading-guides/00-outline/) · [架构](https://cjgogo27.github.io/postraining/midtraining/01-system-overview/03-midtraining-system-architecture/)

六章29节、87个角色问题。面向跨部门工程师、学生和产品/PR，沿虚构设备标签语料说明：任务契约→数据准备→领域/上下文适配→训练与基础设施→验收→交付。中训练没有统一阶段边界；本书明确每个项目与实验的约定。

原后训练书保留于仓库根站，新书是独立Starlight子站，标题、侧栏、搜索与阅读偏好分开。完整`npm run build`先生成原站，再生成`dist/midtraining`；GitHub Pages部署同一个dist目录。

从仓库根目录执行：

```bash
python -m pip install -r midtraining/examples/requirements.txt
python midtraining/examples/verify.py --out runs/my-midtraining-checks
python midtraining/examples/lab.py --out runs/my-midtraining
npm run check:docs
npm test
npm run build
npm run check:built
```

依赖准备后默认实验离线、CPU；不下载模型。语料与结果在`examples/results`。公开SmolLM2基座是可选实验，命令见6.5；本轮实际运行，不是待执行计划。环境、代码身份、失败输出与限制见[VALIDATION.md](VALIDATION.md)、[MODEL_CARD.md](examples/MODEL_CARD.md)。

预算交互只做手算，结果切换读取发布的JSON，没有浏览器训练或远端API。默认模型生成0/4、公开基座0/2均保留；NLL收益不能改写为可靠任务成功。

教学组织参考与许可证见[ATTRIBUTION.md](ATTRIBUTION.md)，科学依据核对见[SOURCE_AUDIT.md](SOURCE_AUDIT.md)。
