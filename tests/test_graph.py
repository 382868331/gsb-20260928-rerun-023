import tempfile
import unittest
from pathlib import Path

import numpy as np

from materialgraph.errors import CycleError, PortError, RenderCancelled
from materialgraph.examples.shared_mask import build_shared_mask_example, write_shared_mask_example
from materialgraph.cli import main as cli_main
from materialgraph.project import load_project, save_project
from materialgraph.render import render_batch


class MaterialGraphTests(unittest.TestCase):
    def test_shared_mask_is_computed_once_for_two_branches(self):
        graph = build_shared_mask_example(8, 8)
        graph.execute(["red_output", "blue_output"])
        self.assertEqual(graph.compute_counts["shared_circle"], 1)
        self.assertEqual(graph.compute_counts["moved_coords"], 1)
        self.assertEqual(graph.compute_counts["red_branch"], 1)
        self.assertEqual(graph.compute_counts["blue_branch"], 1)

        graph.execute(["red_output", "blue_output"])
        self.assertEqual(graph.compute_counts["shared_circle"], 1)

    def test_parameter_change_only_invalidates_related_branch(self):
        graph = build_shared_mask_example(8, 8)
        graph.execute(["red_output", "blue_output"])

        graph.set_param("bg_warm", "rgba", [36, 29, 23, 255])
        graph.execute(["red_output", "blue_output"])

        self.assertEqual(graph.compute_counts["shared_circle"], 1)
        self.assertEqual(graph.compute_counts["moved_coords"], 1)
        self.assertEqual(graph.compute_counts["blue_branch"], 1)
        self.assertEqual(graph.compute_counts["red_branch"], 2)

        graph.set_param("shared_circle", "radius", 0.2)
        graph.execute(["red_output", "blue_output"])
        self.assertEqual(graph.compute_counts["shared_circle"], 2)
        self.assertEqual(graph.compute_counts["red_branch"], 3)
        self.assertEqual(graph.compute_counts["blue_branch"], 2)

    def test_saved_project_reproduces_identical_png(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "example"
            graph, output_paths = write_shared_mask_example(directory, 16, 16)
            red_before = output_paths["red_output"].read_bytes()
            blue_before = output_paths["blue_output"].read_bytes()

            save_project(directory / "copy.json", graph, output_paths)
            loaded_graph, loaded_outputs = load_project(directory / "copy.json")
            rerun_directory = Path(temp) / "rerun"
            rerun_paths = {
                node_id: rerun_directory / Path(path).name
                for node_id, path in loaded_outputs.items()
            }
            render_batch(loaded_graph, rerun_paths)

            self.assertEqual(red_before, (rerun_directory / "red.png").read_bytes())
            self.assertEqual(blue_before, (rerun_directory / "blue.png").read_bytes())

    def test_cancelled_batch_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "example"
            _, output_paths = write_shared_mask_example(directory, 8, 8)
            red_before = output_paths["red_output"].read_bytes()
            blue_before = output_paths["blue_output"].read_bytes()

            checks = iter([False, True])
            graph, _ = load_project(directory / "project.json")
            with self.assertRaises(RenderCancelled):
                render_batch(graph, output_paths, lambda: next(checks))

            self.assertEqual(red_before, output_paths["red_output"].read_bytes())
            self.assertEqual(blue_before, output_paths["blue_output"].read_bytes())
            self.assertFalse(list(directory.glob(".*.tmp")))

    def test_cycle_is_reported_before_running(self):
        graph = build_shared_mask_example(4, 4)
        graph.connections[("moved_coords", "coordinates")] = ("shared_circle", "output")
        with self.assertRaises(CycleError):
            graph.validate()
        with self.assertRaises(CycleError):
            graph.execute(["red_output"])

    def test_port_types_are_checked(self):
        graph = build_shared_mask_example(4, 4)
        with self.assertRaises(PortError):
            graph.connect("bg_warm", "red_branch", "mask")

    def test_images_have_expected_mask_regions(self):
        graph = build_shared_mask_example(8, 8)
        results = graph.execute(["red_output", "blue_output"])
        red = results["red_output"].data
        blue = results["blue_output"].data
        self.assertEqual(red.shape, (8, 8, 4))
        self.assertEqual(blue.shape, (8, 8, 4))
        self.assertTrue(np.any(red[..., 0] > 200))
        self.assertTrue(np.any(blue[..., 2] > 200))

    def test_cli_demo(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "cli-demo"
            cli_main(["demo", str(directory)])
            self.assertTrue((directory / "project.json").is_file())
            self.assertTrue((directory / "red.png").is_file())
            self.assertTrue((directory / "blue.png").is_file())


if __name__ == "__main__":
    unittest.main()
