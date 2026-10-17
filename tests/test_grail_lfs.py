"""Selected-pointer downloads must not scan unrelated partial-clone blobs."""
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'grail_lfs', Path(__file__).resolve().parents[1] / 'src/orcs/cli/grail_stairs.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def pointer(payload):
    oid = hashlib.sha256(payload).hexdigest()
    return f'version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(payload)}\n'.encode()


def test_selected_paths_only_and_no_tree_scan(tmp_path, monkeypatch):
    paths = ['data/stair_p1/robot/a[1].pkl', 'data/stair_p2/robot/a[1].pkl']
    calls = []
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pointer(b'payload'))
    monkeypatch.setenv('GIT_LFS_SKIP_SMUDGE', '1')
    monkeypatch.setenv('GIT_LFS_SKIP_DOWNLOAD_ERRORS', '1')
    monkeypatch.setattr(fetch, '_git', lambda root, *args: calls.append(args))

    def run(command, **kwargs):
        assert command[-4:-1] == ['lfs', 'smudge', '--']
        assert command[-1] in paths
        assert 'GIT_LFS_SKIP_SMUDGE' not in kwargs['env']
        assert 'GIT_LFS_SKIP_DOWNLOAD_ERRORS' not in kwargs['env']
        assert kwargs['input'] == pointer(b'payload')
        kwargs['stdout'].write(b'payload')
    monkeypatch.setattr(fetch.subprocess, 'run', run)
    fetch.fetch_files(tmp_path, paths, 'pinned-revision')
    assert calls == [('--literal-pathspecs', 'checkout', 'pinned-revision', '--', *paths)]
    assert all((tmp_path / name).read_bytes() == b'payload' for name in paths)


@pytest.mark.parametrize('failure', ['bad_hash', 'command', 'pointer'])
def test_failed_download_preserves_pointer(tmp_path, monkeypatch, failure):
    target = tmp_path / 'a'
    original = pointer(b'correct')
    target.write_bytes(original)

    def run(command, **kwargs):
        if failure == 'command':
            raise subprocess.CalledProcessError(1, command)
        kwargs['stdout'].write(original if failure == 'pointer' else b'wrong!!')
    monkeypatch.setattr(fetch.subprocess, 'run', run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        fetch.materialize_lfs(tmp_path, 'a')
    assert target.read_bytes() == original
    assert not list(tmp_path.glob('.orcs-lfs-*'))


def test_plain_file_needs_no_lfs(tmp_path, monkeypatch):
    (tmp_path / 'meta').write_bytes(b'plain metadata')
    monkeypatch.setattr(fetch.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected LFS call'))
    fetch.materialize_lfs(tmp_path, 'meta')


def test_real_smudge_reuses_cache_offline(tmp_path, monkeypatch):
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip('git-lfs unavailable')
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    payload = b'cached binary\x00payload'
    oid = hashlib.sha256(payload).hexdigest()
    cache = tmp_path / '.git/lfs/objects' / oid[:2] / oid[2:4] / oid
    cache.parent.mkdir(parents=True)
    cache.write_bytes(payload)
    (tmp_path / 'a').write_bytes(pointer(payload))
    monkeypatch.setenv('GIT_LFS_SKIP_SMUDGE', '1')
    fetch.materialize_lfs(tmp_path, 'a')
    assert (tmp_path / 'a').read_bytes() == payload


def test_malformed_pointer_rejected_without_download(tmp_path, monkeypatch):
    (tmp_path / 'a').write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:bad\n')
    monkeypatch.setattr(fetch.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected LFS call'))
    with pytest.raises(ValueError, match='invalid Git LFS pointer'):
        fetch.materialize_lfs(tmp_path, 'a')
