"""Guard: the dependencies' telemetry stays switched off.

pyannote.audio and onnxruntime both report usage by default, and both decide
whether to do so when they initialise -- so the switches have to be in the
environment before either is imported, and the Windows-only onnxruntime API
call has to come before faster-whisper's VAD creates its session. The
subprocess tests run in a clean interpreter so the parent's own imports and
environment can't mask a regression.
"""
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from noScribe import privacy

REPO = str(Path(__file__).resolve().parent.parent)


def _without_comments(text):
    """Source with comment lines dropped, so a check for a call can't be
    satisfied by a comment that merely mentions it."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _run(code):
    # Every switch starts out "on", so the test also proves they are forced
    # rather than only defaulted.
    env = {**os.environ, "PYTHONPATH": REPO,
           "PYANNOTE_METRICS_ENABLED": "true", "OTEL_SDK_DISABLED": "false",
           "ORT_DISABLE_TELEMETRY": "0"}
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)


def test_importing_the_package_forces_telemetry_off():
    # Worker children re-import the package to reach their entrypoint, so this
    # covers them as well as the GUI and CLI.
    res = _run(
        "import os, noScribe; "
        "wrong = {k: os.environ.get(k) "
        "for k, v in noScribe.privacy.TELEMETRY_OFF_ENV.items() "
        "if os.environ.get(k) != v}; "
        "raise SystemExit('telemetry still on: %s' % wrong if wrong else 0)"
    )
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.skipif(importlib.util.find_spec("pyannote.audio") is None,
                    reason="pyannote.audio not installed")
def test_pyannote_sees_its_metrics_disabled():
    res = _run(
        "import noScribe; "
        "from pyannote.audio.telemetry.metrics import is_metrics_enabled; "
        "raise SystemExit('pyannote metrics enabled' if is_metrics_enabled() else 0)"
    )
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.skipif(importlib.util.find_spec("opentelemetry.sdk") is None,
                    reason="opentelemetry-sdk not installed")
def test_opentelemetry_hands_out_noop_tracers():
    res = _run(
        "import noScribe; "
        "from opentelemetry.sdk.trace import TracerProvider; "
        "from opentelemetry.trace import NoOpTracer; "
        "tracer = TracerProvider().get_tracer('noScribe-test'); "
        "raise SystemExit(0 if isinstance(tracer, NoOpTracer) "
        "else 'live tracer: %r' % tracer)"
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_disable_onnxruntime_telemetry_calls_the_api(monkeypatch):
    calls = []
    stub = types.ModuleType("onnxruntime")
    stub.disable_telemetry_events = lambda: calls.append(True)
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    privacy.disable_onnxruntime_telemetry()
    assert calls == [True]


def test_disable_onnxruntime_telemetry_without_onnxruntime(monkeypatch):
    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    privacy.disable_onnxruntime_telemetry()


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None,
                    reason="onnxruntime not installed")
def test_disable_onnxruntime_telemetry_against_the_real_package():
    # Catches the API being renamed or dropped in a future onnxruntime.
    privacy.disable_onnxruntime_telemetry()


@pytest.mark.parametrize("module, first_vad_use", [
    ("main.py", "get_speech_timestamps(audio_array"),
    ("whisper_mp_worker.py", "vad_filter=True"),
])
def test_onnxruntime_is_silenced_before_the_vad_runs(module, first_vad_use):
    source = _without_comments((Path(REPO) / "noScribe" / module).read_text(encoding="utf-8"))
    call = "privacy.disable_onnxruntime_telemetry()"
    assert call in source, f"{module} never silences onnxruntime"
    assert first_vad_use in source, f"{module} no longer contains {first_vad_use!r}"
    assert source.index(call) < source.index(first_vad_use), \
        f"{module} runs the VAD before silencing onnxruntime"
