import json
from pathlib import Path

from examples.shared_mask import build_graph as build_static_graph

from texturegraph import RenderTarget, load_project, render_project, save_project
from examples.moving_texture import build_graph


def test_animated_multiresolution_project_round_trip(tmp_path):
    project_path = tmp_path / "moving.json"
    graph, targets = build_graph()
    save_project(str(project_path), graph, targets)

    document = json.loads(project_path.read_text(encoding="utf-8"))
    assert document["version"] == 2
    assert document["tracks"][0]["param"] == "offset_x"
    assert len(document["outputs"]) == 6

    loaded_graph, loaded_targets, _ = load_project(str(project_path))
    assert isinstance(loaded_targets, list)
    assert loaded_targets == targets
    assert list(loaded_graph.tracks) == list(graph.tracks)

    rendered = render_project(str(project_path))
    assert len(rendered) == len(targets)
    assert any(key.startswith("result:f3:512x512:") for key in rendered)


def test_version_one_project_outputs_remain_supported(tmp_path):
    graph, outputs = build_static_graph()
    document = {
        "version": 1,
        "width": graph.width,
        "height": graph.height,
        "nodes": [
            {"id": node.id, "type": node.type, "params": node.params}
            for node in graph.nodes.values()
        ],
        "connections": [
            {
                "source_node": connection.source_node,
                "source_port": connection.source_port,
                "target_node": connection.target_node,
                "target_port": connection.target_port,
            }
            for connection in graph.connections
        ],
        "outputs": {
            node_id: str(tmp_path / Path(path).name)
            for node_id, path in outputs.items()
        },
    }
    project_path = tmp_path / "version-one.json"
    project_path.write_text(json.dumps(document), encoding="utf-8")

    rendered = render_project(str(project_path))

    assert set(rendered) == set(outputs)
