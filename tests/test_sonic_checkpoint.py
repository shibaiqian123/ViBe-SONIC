"""No real weights/network: exercise resolution, cache semantics and process boundaries."""
from __future__ import annotations

import importlib.machinery
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from orcs.cli import sonic_checkpoint as cli

ROOT = Path(__file__).resolve().parents[1]


def args(*values):
    return cli.parser().parse_args(list(values))


@pytest.mark.parametrize('version,filename', [
    ('release', 'sonic_release/last.pt'), ('v1.1', 'sonic_v1_1/last.pt'),
    ('low-latency', 'low_latency/last.pt'),
])
def test_version_hub_mapping_and_snapshot_symlink(tmp_path, monkeypatch, version, filename):
    commit = 'a' * 40
    snapshot = tmp_path / 'models--nvidia--GEAR-SONIC' / 'snapshots' / commit / filename
    blob = tmp_path / 'blob'
    blob.write_bytes(b'fake-weight')
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    calls = []
    hub = types.ModuleType('huggingface_hub')
    def download(**kw):
        calls.append(kw)
        return str(snapshot)
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    path, provenance = cli.resolve_checkpoint(args('--version', version, '--cache-dir', str(tmp_path),
                                                  '--revision', 'my-tag', '--local-files-only'))
    assert path == blob
    assert calls == [{'repo_id': 'nvidia/GEAR-SONIC', 'filename': filename,
                      'cache_dir': str(tmp_path), 'revision': 'my-tag', 'local_files_only': True}]
    assert provenance['resolved_revision'] == commit
    assert provenance['checkpoint_path'] == str(blob)
    assert provenance['version_identity'] == 'official_repository_path'
    assert not (tmp_path / 'last.pt').exists()


def test_sdk_defaults_are_not_overridden(tmp_path, monkeypatch):
    path = tmp_path / 'snapshots' / ('b' * 40) / 'sonic_release/last.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fake')
    calls = []
    hub = types.ModuleType('huggingface_hub')
    hub.hf_hub_download = lambda **kw: calls.append(kw) or str(path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    cli.resolve_checkpoint(args())
    assert calls == [{'repo_id': cli.REPOSITORY, 'filename': 'sonic_release/last.pt', 'local_files_only': False}]


def test_explicit_local_file_wins_without_importing_hub(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    path = tmp_path / 'existing.pt'
    path.write_bytes(b'fake')
    selected, receipt = cli.resolve_checkpoint(args('--version', 'v1.1', '--ckpt', str(path),
                                                  '--revision', 'ignored', '--local-files-only'))
    assert selected == path
    assert receipt['version_identity'] == 'user_declared_not_verified'
    assert receipt['resolved_revision'] is None
    assert receipt['requested_revision'] is None


def test_missing_explicit_path_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    with pytest.raises(FileNotFoundError, match='no fallback'):
        cli.resolve_checkpoint(args('--ckpt', str(tmp_path / 'missing.pt')))


@pytest.mark.parametrize('payload', [b'', b'version https://git-lfs.github.com/spec/v1\noid sha256:a'])
def test_empty_or_pointer_file_is_rejected(tmp_path, payload):
    path = tmp_path / 'last.pt'
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        cli.resolve_checkpoint(args('--ckpt', str(path)))


def test_offline_cache_miss_propagates(monkeypatch):
    hub = types.ModuleType('huggingface_hub')
    def missing(**kw):
        assert kw['local_files_only'] is True
        raise FileNotFoundError('offline cache miss')
    hub.hf_hub_download = missing
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    with pytest.raises(FileNotFoundError, match='offline cache miss'):
        cli.resolve_checkpoint(args('--local-files-only'))


@pytest.mark.parametrize('version', ['v1.1', 'low-latency'])
def test_unadapted_conversion_rejected_before_any_io(tmp_path, monkeypatch, version):
    def fail(*a, **kw):
        raise AssertionError('should not resolve/download/import converter')
    monkeypatch.setattr(cli, 'resolve_checkpoint', fail)
    monkeypatch.setattr(cli, 'find_converter', fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(['--version', version, '--convert', '--out-dir', str(tmp_path / 'output')])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('options', [('--smpl',), ('--out-dir', 'output'), ('--version', 'unknown')])
def test_invalid_arguments(options):
    with pytest.raises(SystemExit) as exc:
        cli.main(list(options))
    assert exc.value.code == 2


def test_converter_is_found_without_importing_mocke(tmp_path, monkeypatch):
    package = tmp_path / 'mocke/src/mocke'
    package.mkdir(parents=True)
    origin = package / '__init__.py'
    origin.write_text('raise AssertionError("must not import")')
    script = tmp_path / 'mocke/scripts/port_sonic_checkpoint.py'
    script.parent.mkdir()
    script.write_text('# fixture')
    spec = importlib.machinery.ModuleSpec('mocke', loader=None, origin=str(origin))
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda name: spec)
    assert cli.find_converter() == script


def test_missing_converter_explains_setup(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(FileNotFoundError, match='sync_deps.sh'):
        cli.find_converter()


def converter_fixture(tmp_path, fail=False):
    script = tmp_path / 'converter.py'
    script.write_text('''import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--ckpt');p.add_argument('--out');p.add_argument('--smpl',action='store_true')
a=p.parse_args()
assert Path(a.ckpt).read_bytes()==b'original'
Path(a.out).write_bytes(b'ported')
''' + ('raise SystemExit(7)\n' if fail else '''if a.smpl:
    Path(a.out).with_name('smpl_ported.pt').write_bytes(b'smpl')
'''))
    ckpt = tmp_path / 'last.pt'
    ckpt.write_bytes(b'original')
    return script, ckpt


def test_release_conversion_outputs_and_no_overwrite(tmp_path):
    script, ckpt = converter_fixture(tmp_path)
    out = tmp_path / 'ported'
    cli.convert_checkpoint(script, ckpt, out, smpl=True, receipt={'selected_version': 'release'})
    assert (out / 'last_ported.pt').read_bytes() == b'ported'
    assert (out / 'smpl_ported.pt').read_bytes() == b'smpl'
    assert json.loads((out / 'conversion.json').read_text())['source']['selected_version'] == 'release'
    assert not (out / '.convert.lock').exists()
    with pytest.raises(FileExistsError):
        cli.convert_checkpoint(script, ckpt, out, smpl=True, receipt={})
    assert ckpt.read_bytes() == b'original'


def test_converter_failure_publishes_no_outputs(tmp_path):
    script, ckpt = converter_fixture(tmp_path, fail=True)
    out = tmp_path / 'ported'
    with pytest.raises(subprocess.CalledProcessError) as exc:
        cli.convert_checkpoint(script, ckpt, out, smpl=True, receipt={})
    assert exc.value.returncode == 7
    assert list(out.iterdir()) == []


def test_receipt_repeated_resolution_is_idempotent(tmp_path):
    receipt = {'selected_version': 'v1.1', 'sha256': 'fake'}
    first = cli.save_source_receipt(tmp_path, receipt)
    assert cli.save_source_receipt(tmp_path, receipt) == first
    assert len(list(tmp_path.iterdir())) == 1


def test_local_main_uses_deps_root_without_changing_task_weights(tmp_path, monkeypatch):
    from orcs.core import paths
    monkeypatch.setattr(paths, 'DEPS_ROOT', tmp_path / 'dependencies')
    ckpt = tmp_path / 'cached.pt'
    ckpt.write_bytes(b'fake')
    cli.main(['--version', 'v1.1', '--ckpt', str(ckpt)])
    receipts = list((paths.DEPS_ROOT / 'checkpoints/sonic/v1.1/sources').glob('*.json'))
    assert len(receipts) == 1
    assert not (paths.DEPS_ROOT / 'checkpoints/sonic/v1.1/ported').exists()


def test_cli_help_and_local_resolution_do_not_import_training_stack(tmp_path):
    ckpt = tmp_path / 'last.pt'
    ckpt.write_bytes(b'fake')
    env = {**os.environ, 'ORCS_DEPS_ROOT': str(tmp_path / 'deps'), 'PYTHONDONTWRITEBYTECODE': '1'}
    code = ('import sys; from orcs.cli.sonic_checkpoint import main; '
            f'main(["--ckpt", {str(ckpt)!r}]); '
            'assert not {"torch", "mjlab", "mocke", "orcs.registration"}.intersection(sys.modules)')
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
    help_result = subprocess.run(['bash', str(ROOT / 'scripts/setup/sonic_checkpoint.sh'), '--help'],
                                env=env, check=True, capture_output=True, text=True)
    assert 'low-latency' in help_result.stdout


def test_main_conversion_failure_preserves_exit_status(tmp_path, monkeypatch):
    from orcs.core import paths
    script, ckpt = converter_fixture(tmp_path, fail=True)
    monkeypatch.setattr(paths, 'DEPS_ROOT', tmp_path / 'deps')
    monkeypatch.setattr(cli, 'find_converter', lambda: script)
    with pytest.raises(SystemExit) as exc:
        cli.main(['--ckpt', str(ckpt), '--convert'])
    assert exc.value.code == 7
    assert not (paths.DEPS_ROOT / 'checkpoints/sonic/release/ported/last_ported.pt').exists()


def test_wheel_packages_and_resolves_console_entry(tmp_path):
    import configparser
    import shutil
    import zipfile

    project = tmp_path / 'project'
    project.mkdir()
    shutil.copytree(ROOT / 'src', project / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    for name in ('pyproject.toml', 'readme.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, project / name)
    if (ROOT / 'LICENSES').is_dir():
        shutil.copytree(ROOT / 'LICENSES', project / 'LICENSES')
    wheels = tmp_path / 'wheels'
    subprocess.run(['uv', 'build', '--wheel', '--no-cache', '--no-build-isolation',
                    '--out-dir', str(wheels), str(project)], check=True, capture_output=True, text=True)
    wheel = next(wheels.glob('*.whl'))
    site = tmp_path / 'site'
    with zipfile.ZipFile(wheel) as archive:
        assert 'orcs/cli/sonic_checkpoint.py' in archive.namelist()
        entry = next(n for n in archive.namelist() if n.endswith('.dist-info/entry_points.txt'))
        config = configparser.ConfigParser()
        config.read_string(archive.read(entry).decode())
        assert config['console_scripts']['orcs-sonic-checkpoint'] == 'orcs.cli.sonic_checkpoint:main'
        archive.extractall(site)
    code = ('import sys; from importlib.metadata import distribution; '
            'ep = next(e for e in distribution("orcs").entry_points if e.name == "orcs-sonic-checkpoint"); '
            'sys.argv=["orcs-sonic-checkpoint", "--help"]; ep.load()()')
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(site), 'PYTHONDONTWRITEBYTECODE': '1'},
                            check=True, capture_output=True, text=True)
    assert 'low-latency' in result.stdout
