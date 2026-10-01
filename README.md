# 材质纹理节点图

Python 底层的确定性材质节点图。节点图把颜色、二维坐标变换和遮罩组合计算成 RGBA 图片，支持时间动画、多分辨率输出和可恢复的帧批次。

## 能力

- 类型化端口：image、mask、coordinates 不能错误互连。
- 节点：纯色、UV、坐标缩放/平移/旋转、圆形/矩形遮罩、遮罩运算、颜色混合。
- 时间动画：用关键帧轨道（ParameterTrack，支持 linear/step）修改任意节点参数。
- 缓存签名同时包含依赖版本（CACHE_VERSION）、分辨率、时间相关参数和上游签名：
  - 上一帧的遮罩不会被当作本帧结果；
  - 预览尺寸的结果不会被高分辨率请求复用；
  - 同一帧同一分辨率的多个输出共享遮罩，只计算一次；
  - 修改某时间段或某条支线上的参数，只重算相关帧和下游分支。
- 同一工程可声明三类渲染任务（RenderJob）：静态预览（static）、连续帧（sequence）和高分辨率静帧。
- 帧批次逐帧计算、逐帧原子发布；PNG 内嵌完整依赖签名和像素摘要，取消后未提交的图像不会被误认为已发布。
- 支持取消后续跑（resume）：只有节点、尺寸、时间、参数和上游依赖签名全部匹配且像素摘要完整的 PNG 才会复用；尺寸正确但来自其他任务或旧参数的文件会重渲染。
 - 同一帧多输出共享同尺寸依赖；修改一个共享参数时，仅依赖它的输出重算，独立输出继续按签名复用，CLI 会说明每个输出复用或重算的原因。
- 参数方案（Variant）：同一工程同时保留多套命名参数覆盖，一次提交成批渲染并比较：
  - 未被覆盖的共享输入和中间结果跨方案只算一次（共享 EvaluationCache）；
  - 每张已发布 PNG 的依赖签名带方案 id，即使有效参数、名称和尺寸完全相同，方案之间也不会互相顶替；
  - 一次提交多个方案后可先完成一部分再取消，重开时逐个方案校验已发布输出是否仍有效；
  - 共享输入变化后，比较报告列出受影响的方案和输出，其余方案的已发布版本继续可用；
  - 可只重算选定方案（only / --variants），剩余方案保留旧版本，之后再处理；
  - 某个方案的共享结果损坏时只恢复受影响者，其他方案已成功发布的内容原样保留。
- 工程保存为确定性 JSON（版本 3，含 tracks、jobs 和 variants）；版本 1、2 工程仍可读取渲染。

## 运行示例

    python examples/shared_mask.py
    python examples/variants.py

示例工程包含两个共享同一个移动遮罩的分支，生成：

- examples/outputs/preview/：64x64 静态预览；
- examples/outputs/frames/：64x64 连续帧（遮罩随时间移动）；
- examples/outputs/hires/：512x512 高分辨率静帧。

也可以渲染已有工程：

    python -m texturegraph.cli path/to/project.json
    python -m texturegraph.cli path/to/project.json --variants warm,small
    python -m texturegraph.cli path/to/project.json --compare

--variants 只重算选定方案，其余方案保留已发布版本；--compare 只打印比较页面：每个方案的每个输出是当前输入的结果（current）还是保留的旧版本（OLD VERSION kept），并汇总受影响的方案。

## 参数方案批渲染

```python
from texturegraph import Variant, VariantOutput, compare_variants, render_variant_batch

variants = [
    Variant("base"),
    Variant("warm", overrides={"red": {"color": [0.95, 0.55, 0.10, 1.0]}}),
]
outputs = [VariantOutput("branch_a", "out/{variant}/a.png", 64, 64)]
render_variant_batch(graph, variants, outputs)          # 共享中间结果只算一次
graph.set_param("shared_mask", "radius", 0.3)           # 共享输入变化
for item in compare_variants(graph, variants, outputs): # 谁受影响、谁仍是当前版本
    print(item.variant_id, item.node_id, item.status)
render_variant_batch(graph, variants, outputs, only=["warm"])  # 只重算选定方案
```

## 编程用法

```python
from texturegraph import (
    EvaluationCache, FrameOutput, FrameRequest,
    Keyframe, ParameterTrack, TextureEngine,
)

graph.add_track(ParameterTrack(
    node_id='circle_transform',
    param_name='offset_x',
    keyframes=(Keyframe(0.0, -0.5), Keyframe(1.0, 0.5)),
))

cache = EvaluationCache()
engine = TextureEngine(graph, cache)
frames = [
    FrameRequest(0.0, (FrameOutput('branch_a', 'out/a.0000.png', 64, 64),)),
    FrameRequest(0.5, (
        FrameOutput('branch_a', 'out/a.0001.png', 64, 64),
        FrameOutput('branch_b', 'out/b.0001.png', 64, 64),
    )),
    FrameRequest(0.5, (FrameOutput('branch_a', 'out/a-hires.png', 512, 512),)),
]
engine.render_frames(frames, is_cancelled=lambda: cancelled)
# 取消后再次调用即可续跑；签名和像素摘要仍匹配当前依赖的帧才会跳过。
engine.render_frames(frames, is_cancelled=lambda: cancelled)
```

## 测试

    python -m pytest
