"""Newcomer-facing contracts for the GGML launcher."""
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cpp_ggml" / "scripts" / "run_ggml.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("run_ggml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_horse_demo_has_a_real_image_and_two_pixel_boxes():
    launcher = load_launcher()

    image, boxes, labels = launcher.demo_request("horse")

    assert image == ROOT / "assets" / "reel" / "images" / "horse.jpg"
    assert image.is_file()
    assert boxes == [[470, 130, 650, 630], [90, 310, 1010, 875]]
    assert labels == ["person", "horse"]


def test_backend_selection_prefers_cuda_then_vulkan_then_cpu():
    launcher = load_launcher()

    assert launcher.select_quickstart_backend(lambda _: True) == "cuda"
    assert launcher.select_quickstart_backend(lambda name: name == "vulkaninfo") == "vulkan"
    assert launcher.select_quickstart_backend(lambda _: False) == "cpu"


def test_doctor_report_marks_missing_tools_and_recommends_cpu():
    launcher = load_launcher()

    report = launcher.doctor_report(lambda _: False)

    assert report["recommended_backend"] == "cpu"
    assert report["tools"]["cmake"] is False
    assert report["tools"]["python3"] is False
    assert report["assets"]["horse_demo"] is True
    assert report["assets"]["default_test_data"] is True


def test_quickstart_resolves_backend_before_running_self_test(monkeypatch):
    launcher = load_launcher()
    commands = []
    monkeypatch.setattr(sys, "argv", ["run_ggml.py", "--quickstart", "--self-test"])
    monkeypatch.setattr(launcher, "select_quickstart_backend", lambda _: "cpu")
    monkeypatch.setattr(launcher, "run", lambda command: commands.append(command))

    assert launcher.main() == 0
    assert commands[-1][-2:] == ["--backend", "cpu", "--self-test"][-2:]


def test_result_summary_reports_latency_and_prediction_count(tmp_path):
    launcher = load_launcher()
    output = tmp_path / "result.json"
    output.write_text(json.dumps({
        "latency_ms": 12.5,
        "predictions": [{"predicate": "riding"}, {"predicate": "on"}],
    }))

    assert launcher.result_summary(output) == "完成：2 条关系，延迟 12.50 ms"
