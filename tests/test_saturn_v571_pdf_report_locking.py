"""PDF report writing must survive a report file being open in a viewer.

On Windows a PDF viewer holds an exclusive lock on the file it is displaying.
matplotlib opens its output lazily on the first ``savefig``, so the resulting
PermissionError previously surfaced deep inside report rendering and discarded
the report for an otherwise completed analysis run. These tests pin the
fallback behaviour that keeps the output.
"""

import ctypes
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_pdf_lock_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writable_existing_path_is_returned_unchanged(tmp_path):
    saturn = load_saturn_v571()
    target = tmp_path / "batch_report.pdf"
    target.write_bytes(b"%PDF-1.4\n")

    assert saturn.resolve_writable_pdf_path(str(target)) == str(target)
    # The probe must not damage the existing report.
    assert target.read_bytes() == b"%PDF-1.4\n"


def test_missing_path_is_returned_and_leaves_no_stray_file(tmp_path):
    saturn = load_saturn_v571()
    target = tmp_path / "new_report.pdf"

    assert saturn.resolve_writable_pdf_path(str(target)) == str(target)
    # Probing must not leave an empty placeholder behind.
    assert not target.exists()


def test_locked_path_falls_back_instead_of_raising(tmp_path, monkeypatch, capsys):
    saturn = load_saturn_v571()
    target = tmp_path / "batch_report.pdf"
    target.write_bytes(b"%PDF-1.4\n")

    real_open = open

    def deny_append(path, mode="r", *args, **kwargs):
        if str(path) == str(target) and "a" in mode:
            raise PermissionError(13, "Permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", deny_append)
    resolved = saturn.resolve_writable_pdf_path(str(target))

    assert resolved != str(target)
    assert resolved.endswith(".pdf")
    assert Path(resolved).parent == tmp_path
    assert "open in another program" in capsys.readouterr().out


def test_unrelated_os_error_does_not_trigger_rename(tmp_path, monkeypatch):
    """A missing directory is not a viewer lock and must not be renamed away."""
    saturn = load_saturn_v571()
    target = tmp_path / "missing_dir" / "batch_report.pdf"

    resolved = saturn.resolve_writable_pdf_path(str(target))

    # The real failure should surface later with its own context, unchanged.
    assert resolved == str(target)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows share-mode locking")
def test_real_exclusive_lock_falls_back_then_recovers(tmp_path):
    """Adversarial: reproduce the exact lock a PDF viewer takes."""
    saturn = load_saturn_v571()
    target = tmp_path / "specimen_group_comparison.pdf"
    target.write_bytes(b"%PDF-1.4\n")

    generic_read = 0x80000000
    open_existing = 3
    no_sharing = 0
    handle = ctypes.windll.kernel32.CreateFileW(
        str(target), generic_read, no_sharing, None, open_existing, 0x80, None
    )
    assert handle != -1, "could not acquire exclusive lock for the test"
    try:
        resolved = saturn.resolve_writable_pdf_path(str(target))
        assert resolved != str(target)
        # The fallback must actually be writable, which is the whole point.
        with open(resolved, "wb") as handle_out:
            handle_out.write(b"%PDF-1.4 fallback\n")
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

    # Once the viewer closes the file, the standard name is used again.
    assert saturn.resolve_writable_pdf_path(str(target)) == str(target)
