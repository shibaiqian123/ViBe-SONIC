"""Scene replacement must run on the playback thread, never GUI workers."""
import ast
from pathlib import Path
from queue import Empty, SimpleQueue
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'src/orcs/cli/view_terrain_motions.py'
main = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


class Dropdown:
    value = 'old'

    def on_update(self, callback):
        self.callback = callback
        return callback

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, values):
        self._options = values
        self.value = values[0]
        self.callback(SimpleNamespace(client_id=None))


def setup():
    calls = []
    env = dict(Empty=Empty, selection_updates=SimpleQueue(),
               gui_family=Dropdown(), gui_level=Dropdown(), gui_sample=Dropdown(),
               _levels_of=lambda family: ('level_0.00',), _key=lambda: 'selected',
               _load_tile=lambda key: calls.append(('tile', key)),
               _load_clip=lambda key: calls.append(('clip', key)))
    nodes = [n for n in main.body if isinstance(n, ast.FunctionDef) and (
        n.name == '_apply_pending_selection' or
        any(isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and
            d.value.id in {'gui_family', 'gui_level', 'gui_sample'} for d in n.decorator_list))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
    return env, calls


def test_gui_callbacks_only_enqueue_until_playback_consumes():
    env, calls = setup()
    env['gui_family'].callback(SimpleNamespace(client_id=0))
    assert calls == []
    env['_apply_pending_selection']()
    assert calls == [('tile', 'selected')]
    # Programmatic level change did not enqueue a second rebuild.
    env['_apply_pending_selection']()
    assert calls == [('tile', 'selected')]


def test_repeated_updates_coalesce_to_single_scene_rebuild():
    env, calls = setup()
    for name in ('gui_sample', 'gui_family', 'gui_level', 'gui_family'):
        env[name].callback(SimpleNamespace(client_id=1))
    env['_apply_pending_selection']()
    assert calls == [('tile', 'selected')]


def test_sample_change_does_not_reset_scene():
    env, calls = setup()
    env['gui_sample'].callback(SimpleNamespace(client_id=1))
    assert calls == []
    env['_apply_pending_selection']()
    assert calls == [('clip', 'selected')]


def test_programmatic_updates_do_not_reenter_load():
    env, calls = setup()
    for name in ('gui_family', 'gui_level', 'gui_sample'):
        env[name].callback(SimpleNamespace(client_id=None))
    env['_apply_pending_selection']()
    assert calls == []
