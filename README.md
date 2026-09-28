# 材质纹理节点图

当前版本先实现 Python 底层，不包含网页界面。节点图把颜色、二维坐标变换和遮罩组合计算成 RGBA 图片，结果数组以后可直接接到三维预览或烘焙流程。

## 能力

- 类型化端口：image、mask、coordinates 不能错误互连。
- 节点：纯色、UV、坐标缩放/平移/旋转、圆形遮罩、矩形遮罩、遮罩运算、颜色混合。
- 共享节点按内容签名缓存；多个输出共享同一遮罩时只计算一次。
- 修改参数后，只有签名变化的节点及其下游会重算。
- 运行前进行引用、缺失输入和循环连接检查。
- 工程保存为确定性 JSON；重新加载同一工程会生成相同 PNG。
- 批次先完成全部计算，再提交输出；取消时保留上次有效 PNG。

## 运行示例

    python examples/shared_mask.py

示例工程会写到 examples/shared-mask-project.json，两张图片写到 examples/outputs/。

也可以渲染已有工程：

    python -m texturegraph.cli path/to/project.json

## 测试

    python -m pytest
