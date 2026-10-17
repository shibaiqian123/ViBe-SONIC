"""Viewer height summaries must support mesh-only and floor-only tiles."""
import ast
import math
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/orcs/cli/view_terrain_motions.py'
# Load the pure geometry helper without a GPU, simulator or task registration.
tree = ast.parse(SOURCE.read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_terrain_top')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), 'exec'), namespace)
top = namespace['_terrain_top']


def box(z=1, half=(1, 2, 0.5), quat=(1, 0, 0, 0)):
    return {'pos': [0, 0, z], 'half': half, 'quat': quat}


def test_mesh_only_stairs():
    assert top({'boxes': [], 'meshes': [
        {'vertices': [[0, 0, 0], [1, 1, 2.5]]},
        {'vertices': [[0, 0, 3.0], [1, 1, 2.0]]},
    ]}) == 3.0


def test_old_box_tiles():
    assert top({'boxes': [box()]}) == 1.5


def test_mixed_geometry():
    assert top({'boxes': [box(z=4)], 'meshes': [{'vertices': [[0, 0, 3]]}]}) == 4.5


def test_floor_and_below_ground_geometry():
    assert top({'boxes': [], 'meshes': []}) == 0.0
    assert top({'boxes': [box(z=-2)]}) == 0.0


def test_rotated_box():
    assert top({'boxes': [box(quat=(math.sqrt(0.5), math.sqrt(0.5), 0, 0))]}) == pytest.approx(3.0)
