"""Material omissions must never hide missing geometry or escape checks."""
import importlib.util
from pathlib import Path

import pytest

Sdf = pytest.importorskip('pxr.Sdf')
spec = importlib.util.spec_from_file_location(
    'grail_usd', Path(__file__).resolve().parents[1] / 'src/orcs/cli/grail_stairs.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def layer_at(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return Sdf.Layer.CreateNew(str(path))


def add_asset(layer, name, kind, path):
    prim = Sdf.PrimSpec(layer, name, Sdf.SpecifierDef, kind)
    attr = Sdf.AttributeSpec(prim, 'inputs:file', Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath(path)


def selection(path='root.usda'):
    return {'samples': [{'files': {'object_usd': path}}]}


def test_shading_paths_skipped_without_modifying_source(tmp_path):
    layer = layer_at(tmp_path / 'root.usda')
    add_asset(layer, 'Metal', 'Shader', '/author/computer/model_metallic.jpg')
    add_asset(layer, 'Mdl', 'Shader', 'OmniPBR.mdl')
    add_asset(layer, 'Color', 'Shader', 'textures/missing.jpg')
    layer.Save()
    original = (tmp_path / 'root.usda').read_bytes()
    assert fetch.usd_dependencies(tmp_path, selection()) == ['root.usda']
    assert (tmp_path / 'root.usda').read_bytes() == original
    assert not layer.dirty


def test_nested_missing_geometry_is_relative_to_own_layer(tmp_path):
    root = layer_at(tmp_path / 'root.usda')
    root.subLayerPaths = ['nested/child.usda']
    root.Save()
    child = layer_at(tmp_path / 'nested/child.usda')
    prim = Sdf.PrimSpec(child, 'Stairs', Sdf.SpecifierDef, 'Xform')
    prim.referenceList.prependedItems = [Sdf.Reference('geometry/steps.usda')]
    child.Save()
    assert fetch.usd_dependencies(tmp_path, selection()) == [
        'nested/child.usda', 'nested/geometry/steps.usda', 'root.usda']


def test_missing_geometry_still_requires_fetch(tmp_path):
    root = layer_at(tmp_path / 'root.usda')
    prim = Sdf.PrimSpec(root, 'Stairs', Sdf.SpecifierDef, 'Xform')
    prim.payloadList.prependedItems = [Sdf.Payload('missing.usda')]
    root.Save()
    assert 'missing.usda' in fetch.usd_dependencies(tmp_path, selection())
    with pytest.raises(FileNotFoundError):
        fetch.check_payload(tmp_path, ['missing.usda'])


def test_external_geometry_rejected(tmp_path):
    root = layer_at(tmp_path / 'root.usda')
    root.subLayerPaths = ['/author/computer/steps.usda']
    root.Save()
    with pytest.raises(ValueError, match='outside source repository'):
        fetch.usd_dependencies(tmp_path, selection())


def test_image_extension_on_non_shader_not_ignored(tmp_path):
    root = layer_at(tmp_path / 'root.usda')
    add_asset(root, 'Geometry', 'Mesh', '/author/computer/height.png')
    add_asset(root, 'Shader', 'Shader', '/author/computer/height.png')
    root.Save()
    with pytest.raises(ValueError, match='outside source repository'):
        fetch.usd_dependencies(tmp_path, selection())
