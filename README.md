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
- 帧批次先完成计算，再逐张原子发布；取消后未提交的 PNG 不会被当作结果，已发布的版本保持不变。
- 支持取消后续跑（resume）：PNG 内嵌输出节点、当前完整依赖签名和像素指纹；只有尺寸、节点身份、输入参数、动画时间、上游依赖和像素内容全部匹配才复用。
- 多材质共享输入时，帧内共享节点只计算一次；局部参数变化只重做签名失效的输出，`FrameResult.reused`、`FrameResult.recomputed` 和 `FrameResult.reasons` 会说明依据。
- 工程保存为确定性 JSON（版本 2，含 tracks 和 jobs）；版本 1 工程仍可读取渲染。

## 运行示例

    python examples/shared_mask.py

示例工程包含两个共享同一个移动遮罩的分支，生成：

- examples/outputs/preview/：64x64 静态预览；
- examples/outputs/frames/：64x64 连续帧（遮罩随时间移动）；
- examples/outputs/hires/：512x512 高分辨率静帧。

也可以渲染已有工程：

    python -m texturegraph.cli path/to/project.json

CLI 对每张图输出“复用”或“重算”，并显示尺寸、节点依赖签名、像素指纹或缺失等判定原因。

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
# 取消后再次调用即可续跑；只有与当前输入、参数、时间和依赖完全一致的文件会被跳过。
engine.render_frames(frames, is_cancelled=lambda: cancelled)
```

## 测试

    python -m pytest
