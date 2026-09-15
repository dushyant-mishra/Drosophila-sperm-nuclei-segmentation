"""Overlay review must be read-only in behaviour, not merely in a label.

`WORKFLOW-GUI-PRIMARY-001` requires that overlay review cannot mutate state while
the manual-correction workflow is unaccepted. That was previously asserted by
checking a warning string appears in the source, which proves only that a label
exists. These tests exercise the click handler and inspect the GUI class for any
route into the correction API.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"

# Writing a correction is the state change that must be unreachable from the GUI
# while the workflow is unaccepted.
CORRECTION_ENTRY_POINTS = {
    "materialize_reviewed_false_detection_revision",
    "retrack_false_detection_corrections",
    "append_correction_event",
    "create_correction_base_manifest",
    "materialize_false_detection_revision",
}


class _Label:
    def __init__(self):
        self.text = ""

    def config(self, **kwargs):
        self.text = kwargs.get("text", self.text)


class _Mode:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Event:
    def __init__(self, axes, button=1, xdata=10.0, ydata=20.0):
        self.inaxes = axes
        self.button = button
        self.xdata = xdata
        self.ydata = ydata


class _Gui:
    """The minimum surface on_click touches, so no display is required."""

    def __init__(self, mode):
        self.ax = object()
        self.mode_var = _Mode(mode)
        self.roi_active = False
        self.roi_points = []
        self.lbl_roi = _Label()
        self.renders = 0

    def render(self):
        self.renders += 1

    def reset_roi(self, redraw=True):
        self.roi_points = []
        self.roi_active = False


@pytest.fixture(scope="module")
def gui_class():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "saturn_v571_overlay_readonly_test", PIPELINE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpermGUI


def test_review_mode_click_mutates_nothing(gui_class):
    gui = _Gui("review")
    gui_class.on_click(gui, _Event(gui.ax, button=1))
    gui_class.on_click(gui, _Event(gui.ax, button=3))

    assert gui.roi_points == []
    assert gui.roi_active is False
    assert gui.renders == 0
    assert gui.lbl_roi.text == ""


def test_view_mode_click_mutates_nothing(gui_class):
    gui = _Gui("view")
    gui_class.on_click(gui, _Event(gui.ax, button=1))
    assert gui.roi_points == []
    assert gui.renders == 0


def test_the_mutation_check_can_actually_detect_a_mutation(gui_class):
    """Control: the same call in ROI mode must change state.

    Without this, the read-only tests above would pass even if on_click did
    nothing at all in every mode.
    """
    gui = _Gui("roi")
    gui_class.on_click(gui, _Event(gui.ax, button=1))

    assert len(gui.roi_points) == 1
    assert gui.renders == 1
    assert "building" in gui.lbl_roi.text


def test_a_click_outside_the_axes_is_ignored_in_every_mode(gui_class):
    for mode in ("review", "view", "roi"):
        gui = _Gui(mode)
        gui_class.on_click(gui, _Event(axes=object(), button=1))
        assert gui.roi_points == []
        assert gui.renders == 0


def test_no_gui_method_reaches_the_correction_api():
    """A read-only notice is worthless if a control still calls the writer."""
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    gui_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "SpermGUI"
    )
    called = set()
    for node in ast.walk(gui_class):
        if isinstance(node, ast.Call):
            target = node.func
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr if isinstance(target, ast.Attribute) else ""
            )
            if name in CORRECTION_ENTRY_POINTS:
                called.add(name)
    assert called == set(), (
        "the GUI reaches the correction API while manual correction is "
        f"unaccepted: {sorted(called)}"
    )


def test_the_read_only_notice_is_still_shown_to_the_operator():
    """Behaviour is the requirement; the notice explains it to the user."""
    source = PIPELINE.read_text(encoding="utf-8")
    assert "Overlay review is read-only" in source
