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
- 工程保存为确定性 JSON（版本 2，含 tracks 和 jobs）；版本 1 工程仍可读取渲染。
- 参数方案（variants）：工程可声明多套命名参数覆盖，一次提交整批计算并互相比较：
  - 方案共享图结构，未覆盖的节点签名一致，共享输入和中间结果只计算一次；
  - 每个方案的输出发布在 variants/<方案名>/ 下，PNG 内嵌方案标签，同名同尺寸也不会互相顶替；
  - 批次可中途取消或关闭，重开时逐方案校验已发布输出（签名、像素摘要、标签）是否仍有效；
  - 共享输入变化后，compare 逐方案列出受影响（stale）与仍可用（current）的输出，旧版本保留不删；
  - 可只重算选定方案，其余方案的旧版本继续可用；
  - 某份共享结果损坏时仅受影响输出重算，其他方案已发布内容不受影响。
 - 区域局部重算与发布（``texturegraph.regions``）：方案先整幅发布为带区域清单的基线版本，之后只改一个画面区域：
  - ``plan_update`` 预览每个局部候选必须重算的像素矩形与可沿用区域；``preview_update``/``compare_candidate_impacts`` 按候选图真实像素差异扩张足迹并比较各候选对同一已发布版的影响。
  - 足迹沿下游邻域节点（新增 ``blur_mask`` 盒式模糊，半径 ``radius_px``）向外扩张；拼合后与“整幅重算”逐像素核对，边界不留拼接缝，足迹之外像素与旧发布版逐字节一致。
  - 两个局部候选重叠时预览互相标记 ``overlaps``，只能二选一或用不相交区域重新划定；同一区域被两个候选认领直接拒绝，禁止把不同候选半成品混成正式版。
  - 每个区域带显式 ``input_version``：同尺寸同名但来源不同即为不同来源；``release.json`` 为每块区域记录输入版本、候选 id、参数选择（variant）、依赖签名与像素摘要；光晕伸入邻区时以 ``contributors`` 记录来源。
  - 发布为不可变版本（``releases/<id>/``，旧版永不删除）；取消只在 ``staging/<plan-hash>/`` 留下按完整方案哈希隔离的半成品，续跑只复用签名与摘要都匹配的暂存块。
  - ``inspect_release``/``inspect_all_releases`` 在输入后来变化或共享结果损坏时逐区域列出 current/stale/damaged 及受影响的发布与方案，未依赖它的发布结果仍可使用。

## 运行示例

    python examples/shared_mask.py

示例工程包含两个共享同一个移动遮罩的分支，生成：

- examples/outputs/preview/：64x64 静态预览；
- examples/outputs/frames/：64x64 连续帧（遮罩随时间移动）；
- examples/outputs/hires/：512x512 高分辨率静帧。

也可以渲染已有工程：

    python -m texturegraph.cli path/to/project.json

带方案的工程：

    python -m texturegraph.cli project.json                 渲染全部方案
    python -m texturegraph.cli project.json --variant warm  只重算选定方案
    python -m texturegraph.cli project.json --compare       比较各方案输出状态

工程 JSON 中声明方案：

    "variants": [
      {"id": "warm", "overrides": {"red.color": [0.95, 0.4, 0.1, 1.0]}},
      {"id": "cool", "overrides": {"red.color": [0.1, 0.3, 0.95, 1.0]}}
    ]

区域局部重算示例：

    python examples/region_update.py

编程入口：

```python
from texturegraph import Region, RegionCandidate, RegionPublisher, Variant

publisher = RegionPublisher(graph, "out", 64, 64, root="runs/coating")
regions = [
    Region("left", 0, 0, 32, 64, input_version="photo-v1"),
    Region("right", 32, 0, 32, 64, input_version="photo-v1"),
]
baseline = publisher.publish_baseline(regions, "r0")
candidate = RegionCandidate(
    "move", Region("left", 0, 0, 32, 64, input_version="photo-v2"),
    Variant("move", {("circle", "center_x"): 0.35}),
)
_base, plans = publisher.preview_update([candidate], baseline)
release = publisher.publish_update({"left": candidate}, "r1", baseline, plans)
# 拼合结果在区域内与整幅重算逐像素一致（见 tests/test_regions.py）；
# 来源清单：runs/coating/releases/r1/release.json
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
