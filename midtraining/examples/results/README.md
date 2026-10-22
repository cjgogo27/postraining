# 发布记录

- `lab-reference.json`：默认两seed配比实验与长度接口实验的完整实际记录。
- `corpus-reference.json`：实际生成的160篇原始文档与切分。
- `checks-reference.json`：实际12项CPU检查，包括临时改学习率源码后拒绝恢复。
- `pretrained-reference.json`：固定公开SmolLM2基座的实际适配、原始输出和保存重载对照。

2026-10-23从独立runs目录原样复制；没有填充未运行结果。报告中的code/data身份用于绑定实验，绝对运行路径是本轮环境记录，不要求读者复用该路径。权重与大型缓存不进入Git。

`python midtraining/examples/plot_results.py`从lab-reference读取固定窗口评测生成公开SVG/PNG；绘图可选依赖matplotlib3.9.2，不是默认训练依赖。曲线不是训练batch loss，也没有平滑、挑seed或挑终点。
