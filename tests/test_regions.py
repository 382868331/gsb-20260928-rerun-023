import numpy as np
import pytest
from PIL import Image

from texturegraph import (
    EvaluationCache,
    Graph,
    RegionBox,
    RegionError,
    RegionWorkspace,
    judge_affected_range,
    load_region_workspace,
)
from texturegraph.regions import (
    LocalEvaluator,
    evaluate_full,
)


WIDTH = HEIGHT = 24


def build_blur_graph():
    graph = Graph(width=WIDTH, height=HEIGHT)
    graph.add_node('uv', 'uv_coordinates')
    graph.add_node('c1', 'circle_mask', {'center_x': -0.4, 'radius': 0.22})
    graph.add_node('c2', 'circle_mask', {'center_x': 0.4, 'radius': 0.22})
    graph.add_node('m', 'combine_masks', {'operation': 'maximum'})
    graph.add_node('red', 'solid_color', {'color': [0.9, 0.1, 0.1, 1.0]})
    graph.add_node('blue', 'solid_color', {'color': [0.1, 0.1, 0.9, 1.0]})
    graph.add_node('blend', 'blend_colors')
    graph.add_node('blur', 'box_blur', {'radius': 3})
    graph.connect('uv', 'c1', 'coordinates')
    graph.connect('uv', 'c2', 'coordinates')
    graph.connect('c1', 'm', 'a')
    graph.connect('c2', 'm', 'b')
    graph.connect('red', 'blend', 'foreground')
    graph.connect('blue', 'blend', 'background')
    graph.connect('m', 'blend', 'mask')
    graph.connect('blend', 'blur', 'image')
    return graph


def u8(array):
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def test_local_region_matches_full_recompute_pixel_by_pixel():
    graph = build_blur_graph()
    box = (2, 3, 18, 15)
    local = LocalEvaluator(graph, WIDTH, HEIGHT, 0.0, {('c1', 'radius'): 0.33})
    pixels, needs, _ = local.evaluate_region('blur', box)
    reference = evaluate_full(
        graph, 'blur', WIDTH, HEIGHT, 0.0, {('c1', 'radius'): 0.33}
    )
    top, left, bottom, right = box
    assert (u8(pixels) == u8(reference[top:bottom, left:right])).all()
    # blur window was dilated past the region border to gather the halo
    assert needs['blend'][1] < left or needs['blend'][3] > right


def test_two_adjacent_candidates_overlap_must_be_explicitly_chosen(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    split = WIDTH // 2
    workspace.create_candidate(
        'left', (0, 0, HEIGHT, split + 2),
        {('c1', 'radius'): 0.33, ('blur', 'radius'): 3},
    )
    workspace.create_candidate(
        'right', (0, split - 2, HEIGHT, WIDTH),
        {('c2', 'radius'): 0.33, ('blend', 'invert_mask'): True},
    )
    with pytest.raises(RegionError, match='overlaps other pending'):
        workspace.publish_candidate('left')
    # Cancel one explicitly, then adopt the other.
    workspace.discard_candidate('right')
    workspace.publish_candidate('left')

    override = evaluate_full(
        graph, 'blur', WIDTH, HEIGHT, 0.0, {('c1', 'radius'): 0.33}
    )
    base = evaluate_full(graph, 'blur', WIDTH, HEIGHT, 0.0)
    with Image.open(out) as image:
        published = np.asarray(image.convert('RGBA'), dtype=np.uint8)
    # The chosen region carries the left override (halo included).
    assert (published[:, : split + 2] == u8(override[:, : split + 2])).all()
    # The rest keeps the previously published base content and source.
    assert (published[:, split + 2 :] == u8(base[:, split + 2 :])).all()
    labels = workspace.sources()
    assert set(np.unique(labels[:, : split + 2])) == {'left'}
    assert set(np.unique(labels[:, split + 2 :])) == {'base'}


def test_replace_keyword_cancels_overlapping_candidate(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    split = WIDTH // 2
    workspace.create_candidate('left', (0, 0, HEIGHT, split + 1), {('c1', 'radius'): 0.3})
    workspace.create_candidate('right', (0, split, HEIGHT, WIDTH), {('c2', 'radius'): 0.3})
    workspace.publish_candidate('left', replace=['right'])
    assert list(workspace.candidates) == []
    assert list(workspace.zone_boxes()) == ['left']


def test_old_published_region_cannot_be_overwritten(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    workspace.create_candidate('one', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    workspace.publish_candidate('one')
    workspace.create_candidate('two', (0, 5, HEIGHT, 12), {('c1', 'radius'): 0.4})
    with pytest.raises(RegionError, match='cannot be overwritten'):
        workspace.publish_candidate('two')


def test_candidate_goes_stale_after_same_named_same_sized_input_change(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.create_candidate('left', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    graph.set_param('red', 'color', [0.2, 0.9, 0.2, 1.0])
    with pytest.raises(RegionError, match='stale'):
        workspace.publish_candidate('left')


def test_save_and_restore_revives_adopted_range_and_input_version(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    workspace.create_candidate('left', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    workspace.publish_candidate('left')
    workspace.save_state()
    expected = np.asarray(Image.open(out).convert('RGBA'), dtype=np.uint8)

    restored, report = load_region_workspace(graph, 'blur', out, WIDTH, HEIGHT)
    assert report == {'restored': ['left'], 'dropped_stale': [], 'dropped_damaged': []}
    assert restored.zone_boxes() == {'left': (0, 0, HEIGHT, 10)}
    restored_pixels = np.asarray(Image.open(out).convert('RGBA'), dtype=np.uint8)
    assert (restored_pixels == expected).all()


def test_restore_rejudges_range_after_input_version_change(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    workspace.create_candidate('left', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    workspace.publish_candidate('left')
    workspace.save_state()

    graph.set_param('blue', 'color', [0.0, 0.0, 0.0, 1.0])
    restored, report = load_region_workspace(graph, 'blur', out, WIDTH, HEIGHT)
    assert report['dropped_stale'] == ['left']
    assert restored.zones == []


def test_restore_drops_damaged_region_cache(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    workspace.create_candidate('left', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    zone = workspace.publish_candidate('left')
    workspace.save_state()
    with open(zone.source_path, 'r+b') as handle:
        handle.seek(50)
        handle.write(b'\x00\x00\x00')
    _, report = load_region_workspace(graph, 'blur', out, WIDTH, HEIGHT)
    assert report['dropped_damaged'] == ['left']


def test_restore_damaged_state_rejudges_everything(tmp_path):
    graph = build_blur_graph()
    out = str(tmp_path / 'out.png')
    workspace = RegionWorkspace(graph, 'blur', out, WIDTH, HEIGHT)
    workspace.compose()
    workspace.create_candidate('left', (0, 0, HEIGHT, 10), {('c1', 'radius'): 0.3})
    workspace.publish_candidate('left')
    workspace.save_state()
    with open(workspace.state_path, 'w', encoding='utf-8') as handle:
        handle.write('{not json')
    _, report = load_region_workspace(graph, 'blur', out, WIDTH, HEIGHT)
    assert report['restored'] == []
    assert report['dropped_damaged'] == ['*']


def test_affected_range_is_rejudged_for_changed_inputs():
    graph = build_blur_graph()
    box = judge_affected_range(
        graph, 'blur', WIDTH, HEIGHT, 0.0, {('c2', 'radius'): 0.4}
    )
    # Only the right half of the canvas is influenced by the right circle.
    assert box is not None
    top, left, bottom, right = box
    assert left >= WIDTH // 2 - 4 and right == WIDTH
    assert top >= 0 and bottom <= HEIGHT


def test_no_visible_change_means_no_affected_range():
    graph = build_blur_graph()
    assert judge_affected_range(
        graph, 'blur', WIDTH, HEIGHT, 0.0, {('c1', 'softness'): 0.0}
    ) is None
