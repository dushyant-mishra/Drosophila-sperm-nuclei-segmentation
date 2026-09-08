"""Report warnings must not open modal dialogs in unattended runs.

The report generators are reachable from batch and multi-sample study runs. A
modal Tk dialog raised there blocks until a human dismisses it, which can stall
an overnight cohort run indefinitely, and it also pops dialogs onto the
operator's screen during automated test runs. Warnings are therefore modal only
when an interactive Tk session already exists.
"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_report_notify_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headless_run_logs_instead_of_opening_a_dialog(capsys):
    saturn = load_saturn_v571()

    shown = saturn.notify_report_warning("Reporting Warning", "PDF failed")

    assert shown is False
    captured = capsys.readouterr().out
    assert "Reporting Warning" in captured
    assert "PDF failed" in captured


def test_dialog_is_used_when_an_interactive_session_exists(monkeypatch):
    saturn = load_saturn_v571()
    tkinter = __import__("tkinter")
    calls = []

    monkeypatch.setattr(
        "tkinter.messagebox.showwarning",
        lambda title, message: calls.append((title, message)),
    )
    monkeypatch.setattr(tkinter, "_default_root", object(), raising=False)

    shown = saturn.notify_report_warning("Reporting Warning", "PDF failed")

    assert shown is True
    assert calls == [("Reporting Warning", "PDF failed")]


def test_dialog_failure_never_propagates(monkeypatch):
    """A broken display must not turn a report warning into a crash."""
    saturn = load_saturn_v571()
    tkinter = __import__("tkinter")

    def explode(_title, _message):
        raise RuntimeError("no display")

    monkeypatch.setattr("tkinter.messagebox.showwarning", explode)
    monkeypatch.setattr(tkinter, "_default_root", object(), raising=False)

    assert saturn.notify_report_warning("Reporting Warning", "PDF failed") is False


def test_batch_pdf_failure_does_not_open_a_dialog(tmp_path, monkeypatch):
    """Regression: the synthetic-PDF-failure test used to pop a real dialog."""
    import pandas as pd

    saturn = load_saturn_v571()
    opened = []
    monkeypatch.setattr(
        "tkinter.messagebox.showwarning",
        lambda title, message: opened.append(title),
    )

    class BrokenPdf:
        def __init__(self, *_args, **_kwargs):
            raise OSError("synthetic PDF failure")

    monkeypatch.setattr(saturn, "PdfPages", BrokenPdf)
    try:
        saturn.generate_batch_report(
            tmp_path,
            pd.DataFrame(),
            pd.DataFrame(),
            {"xy": 1.0, "z": 1.0},
            generate_pptx=False,
        )
    except RuntimeError:
        pass

    assert opened == [], "batch report generation must not open a modal dialog"
