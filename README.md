# MaterialGraph

Python 底层节点纹理计算原型，不包含网页界面。节点通过强类型端口连接，可生成 RGBA 图像，并将结果缓存供后续三维预览、烘焙或其他导出流程复用。

## 能力

- 坐标生成与平移、缩放、旋转。
- 纯色、线性渐变、圆形和矩形遮罩。
- 遮罩交集、并集、差集和异或组合。
- 遮罩混合颜色并输出 RGBA PNG。
- 多个分支共享节点时，该节点在一次执行中只计算一次。
- 修改参数后只失效该节点及其下游结果。
- 运行前检查缺失连接、端口类型和循环连接。
- 工程序列化为确定性 JSON，PNG 以临时文件原子替换。
- 批次取消时只保留内存结果，不提交也不覆盖上次有效图像。

## 快速运行

```bash
.venv/bin/python -m materialgraph.cli demo dist/shared-mask
```

生成 `project.json`、`red.png` 和 `blue.png`。两个颜色分支共用同一个经过坐标变换的圆形遮罩。

渲染已保存工程：

```bash
.venv/bin/python -m materialgraph.cli render dist/shared-mask/project.json
```

## Python API

```python
from materialgraph.examples.shared_mask import build_shared_mask_example

graph = build_shared_mask_example(width=64, height=64)
results = graph.execute(["red_output", "blue_output"])
red = results["red_output"].data
blue = results["blue_output"].data
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖共享节点计算次数、局部重算、工程复现、循环报错、类型检查和批次取消原子性。
