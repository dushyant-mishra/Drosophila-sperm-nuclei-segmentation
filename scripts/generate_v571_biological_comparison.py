"""Fail-closed v5.7.1 entry point for specimen-level biological reports."""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.saturn_v571_gui_services import production_audit_gate_state

try:
    from scripts.generate_v57_biological_comparison import main as _generate_report_main
except ModuleNotFoundError:  # Direct execution places this directory on sys.path.
    from generate_v57_biological_comparison import main as _generate_report_main


def _require_explicit_group_direction(arguments):
    reference_present = any(
        value == "--reference-group" or value.startswith("--reference-group=")
        for value in arguments
    )
    comparison_present = any(
        value == "--comparison-group" or value.startswith("--comparison-group=")
        for value in arguments
    )
    if not reference_present or not comparison_present:
        raise SystemExit(
            "Saturn v5.7.1 requires explicit --reference-group and "
            "--comparison-group values from validated study roles. Group names "
            "are never interpreted as genotype semantics."
        )


def _require_production_audit_gate(project_root=PROJECT_ROOT):
    ready, detail = production_audit_gate_state(project_root)
    if not ready:
        raise SystemExit(
            "Saturn v5.7.1 biological report generation is blocked by the "
            f"scientific audit gate. {detail}"
        )


def _argument_value(arguments, name):
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return ""


def _require_complete_cohort(arguments):
    study_output = _argument_value(arguments, "--study-output")
    if not study_output:
        raise SystemExit("Saturn v5.7.1 requires --study-output for cohort validation.")
    pipeline_path = PROJECT_ROOT / "sperm_segmentation_saturnv5.7.1.py"
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_report_validation", pipeline_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Saturn v5.7.1 cohort validator could not be loaded.")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    try:
        pipeline._validated_complete_study_report_inputs(Path(study_output))
    except Exception as exc:
        raise SystemExit(
            "Saturn v5.7.1 biological report generation requires a fully "
            f"complete validated cohort. {type(exc).__name__}: {exc}"
        ) from exc


def _main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    _require_explicit_group_direction(arguments)
    _require_production_audit_gate()
    _require_complete_cohort(arguments)
    _generate_report_main(arguments + ["--metric-profile", "concise_v571"])


if __name__ == "__main__":
    _main()
