import sys

import pytest

from triton_portcheck import envs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_PORTCHECK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(envs, "wheel_size", lambda version, tag, timeout=6.0: 100 * 1024**2)
    return tmp_path / "home"


def fake_install(commands):
    def _run(command, log):
        commands.append(command)
        if "venv" in command:
            python = envs.env_python(envs.env_path("3.8.0"))
            python.parent.mkdir(parents=True)
            python.write_bytes(b"x" * 2048)

    return _run


def test_home_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("TRITON_PORTCHECK_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert envs.home_dir() == tmp_path / "cache" / "triton-portcheck"
    monkeypatch.setenv("TRITON_PORTCHECK_HOME", str(tmp_path / "explicit"))
    assert envs.envs_dir() == tmp_path / "explicit" / "envs"
    assert envs.env_path("3.8.0").name == "triton-3.8.0"


def test_add_reports_disk_usage_and_installs_pinned_version(home, monkeypatch):
    commands, messages = [], []
    monkeypatch.setattr(envs, "_run", fake_install(commands))
    path = envs.add_env("3.8.0", extra=["numpy"], installer="pip", log=messages.append)
    assert path == home / "envs" / "triton-3.8.0"
    assert commands[0][1:3] == ["-m", "venv"]
    assert commands[1][-2:] == ["triton==3.8.0", "numpy"]
    text = "\n".join(messages)
    assert "needs about 360.0 MiB (wheel is 100.0 MiB)" in text
    assert "free" in text
    assert "on disk; remove with `triton-portcheck env remove 3.8.0`" in text

    messages.clear()
    envs.add_env("3.8.0", log=messages.append)
    assert "already exists" in messages[0]
    assert len(commands) == 2


def test_add_refuses_when_disk_is_short_unless_forced(home, monkeypatch):
    commands = []
    monkeypatch.setattr(envs, "_run", fake_install(commands))
    monkeypatch.setattr(envs, "free_bytes", lambda path: 500 * 1024**2)
    with pytest.raises(envs.EnvError, match="not enough free disk"):
        envs.add_env("3.8.0", log=lambda m: None)
    assert commands == []
    envs.add_env("3.8.0", force=True, installer="pip", log=lambda m: None)
    assert commands


def test_failed_install_leaves_nothing_behind(home, monkeypatch):
    def boom(command, log):
        if "venv" in command:
            envs.env_python(envs.env_path("3.8.0")).parent.mkdir(parents=True)
        else:
            raise envs.EnvError("pip failed")

    monkeypatch.setattr(envs, "_run", boom)
    with pytest.raises(envs.EnvError, match="pip failed"):
        envs.add_env("3.8.0", installer="pip", log=lambda m: None)
    assert not envs.env_path("3.8.0").exists()


def test_invalid_version_is_rejected(home):
    with pytest.raises(envs.EnvError, match="not a Triton version"):
        envs.add_env("latest; rm -rf /", log=lambda m: None)


def test_list_and_remove(home, monkeypatch):
    monkeypatch.setattr(envs, "_run", fake_install([]))
    assert envs.list_envs() == []
    envs.add_env("3.8.0", installer="pip", log=lambda m: None)
    ((version, path, size),) = envs.list_envs()
    assert (version, path, size) == ("3.8.0", home / "envs" / "triton-3.8.0", 2048)
    assert envs.remove_env("3.8.0") == 2048
    assert envs.list_envs() == []
    with pytest.raises(envs.EnvError, match="no environment"):
        envs.remove_env("3.8.0")


def test_base_python_override(monkeypatch):
    monkeypatch.delenv("TRITON_PORTCHECK_PYTHON", raising=False)
    assert str(envs.base_python()) == sys.executable
    monkeypatch.setenv("TRITON_PORTCHECK_PYTHON", "/opt/py/bin/python")
    assert str(envs.base_python()) == "/opt/py/bin/python"


def test_run_surfaces_command_output():
    with pytest.raises(envs.EnvError, match="command failed"):
        envs._run([sys.executable, "-c", "import sys; print('why it broke'); sys.exit(4)"], lambda m: None)
