"""Switch off the usage telemetry that noScribe's dependencies send by default.

noScribe itself collects nothing, but two libraries it runs on every
transcription do, unless told otherwise:

* pyannote.audio (>= 4.0) sends an OpenTelemetry trace to otel.pyannote.ai
  each time the diarization pipeline and its models are loaded and each time
  it is applied -- with the audio duration and the number of speakers.
* onnxruntime, which faster-whisper uses for its Silero VAD, reports a
  ProcessInfo event (device and hardware ids, machine model, OS, memory) to
  Microsoft over HTTPS on Linux and macOS from 1.29 on, and logs events
  through ETW into the Windows diagnostic pipeline on Windows.

This module has to stay stdlib-only at import time: the package __init__
imports it, and every multiprocessing "spawn" worker re-imports the package.
"""
import os

# Forced rather than setdefault: an inherited "true" must not switch any of
# them back on. All of them are read when the library initialises, so they must
# be in place before pyannote.audio or onnxruntime is first imported -- the
# package __init__ applies them before any submodule is loaded, and spawn
# workers inherit them from the parent's environment on top of that.
TELEMETRY_OFF_ENV = {
    # pyannote.audio only fills this from its own config.yaml (which defaults
    # to true) when the variable is not set yet.
    "PYANNOTE_METRICS_ENABLED": "false",
    # A second lock for pyannote: with the OpenTelemetry SDK disabled, its
    # TracerProvider hands out no-op tracers, so nothing reaches the exporter
    # even if a later pyannote release stops honouring the variable above.
    "OTEL_SDK_DISABLED": "true",
    # onnxruntime's Linux/macOS telemetry. It fires while the native module
    # initialises, so an API call after the import would be too late.
    "ORT_DISABLE_TELEMETRY": "1",
}


def disable_telemetry():
    os.environ.update(TELEMETRY_OFF_ENV)


def disable_onnxruntime_telemetry():
    """Silence onnxruntime's ETW events on Windows, which ORT_DISABLE_TELEMETRY
    does not cover. Call it before faster-whisper's VAD is first used: that is
    where onnxruntime gets imported and its inference session created.
    """
    try:
        import onnxruntime
    except ImportError:
        # faster-whisper reports the missing package itself when the VAD loads.
        return
    onnxruntime.disable_telemetry_events()
