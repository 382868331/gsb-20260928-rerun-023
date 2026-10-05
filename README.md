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
- 局部候选与分区发布（RegionWorkspace）：
  - 先 `publish` 不可变基线版本，再用 `create_candidate` 在矩形区域内按参数覆盖计算候选；候选单独存放，旧发布永远不会被候选覆盖；
  - 候选在持久化前与同一覆盖下的完整重算逐像素核对（含 uint8 量化），不一致即报错；
  - 跨界邻域节点（box_blur）按半径向上游反推完整窗口，区域边界像素使用完整 (2r+1) 邻域，与整图重算一致；
  - 登记候选不再拒绝重叠：多个覆盖同一区域的方案可同时保存，`compare_candidates` 并列展示各自参数、逐节点输入版本签名、相对当前发布的像素差异以及方案间重叠区域的像素差异；
  - 只有采用时才要求解决冲突：与待采用矩形真正重叠的未决候选必须显式采用或 `cancel_candidate`，相邻的未决候选原样保留并可按任意顺序逐个采用，登记顺序不决定结果；重叠的已采纳分区只能 `adopt_candidate(..., replace=True)` 替换；
  - 采用只写新版本：候选矩形之外的像素及来源（provenance）逐字节保留，已采纳分区像素独立于发布画布持久化；
  - `save_state` 保存采用范围和逐节点输入版本签名；`RegionWorkspace.open` 恢复状态（不自动重发），`recover_target` 对同名同尺寸输入变化或缓存/文件损坏重新判断影响范围，只局部重算受影响矩形，并再次与完整重算一致；
  - 待处理候选在状态中恢复，像素损坏或签名不符时标记为需要重新判断；`revalidate_candidate` 只重算指定方案（输入变化或瓦片损坏），其他未决选择（含同区域的重叠方案）原样保留，`recover_target` 会按当前输入版本与瓦片完整性逐个重验该目标的未决候选。
  - 依赖闭合组合（`plan_combinations` / `adopt_combination`）：按真实像素写入与邻域读取推导可同时采用的候选组合——写入矩形相交，或一方的读取窗口（按候选自身覆盖下的模糊半径沿依赖链累积外扩）触及另一方写入区，都视为依赖；几何上不重叠但跨模糊光晕互相读取的候选也会明确列出，而不是只检查矩形重叠。报告给出每个候选的读取范围、输入失效标记、依赖对、需要显式选择的冲突簇和可单独采用的独立候选，并比较"保留当前像素最多"与"满足指定区域最多"两种方案；`adopt_combination` 只接受依赖闭合且输入未失效的集合，冲突成员必须显式取舍，未选择区域逐字节保留；原输入改变后相关候选被标记，需 `revalidate_candidate` 重新核对后才能采用。

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
