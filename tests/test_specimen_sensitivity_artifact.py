import importlib.util


def _module():
    spec = importlib.util.spec_from_file_location("artifact", "scripts/generate_specimen_sensitivity_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rows_keep_excluded_specimens_and_mark_denominators():
    rows = _module().build_rows([
        {"sample_id": "a", "group": "WT", "status": "complete",
         "qc_analysis_population_unet_track_count": "10", "qc_technical_failure_track_count": "2",
         "body_width_available_fraction": "0.75", "roi_area_um2": "20", "sampled_roi_volume_um3": "40",
         "normalization_valid": "True", "normalization_warning": ""},
        {"sample_id": "b", "group": "KJ", "status": "excluded"},
    ])
    assert len(rows) == 2
    assert rows[0]["analysis_included"] is True
    assert rows[0]["width_missing"] is True
    assert rows[0]["area_denominator_valid"] and rows[0]["volume_denominator_valid"]
    assert rows[0]["all_reconstructed_count"] == 12
    assert rows[0]["technical_exclusion_fraction"] == 2 / 12
    assert rows[0]["primary_count_per_1000_um2"] == 500.0
    assert rows[0]["all_reconstructed_count_per_100000_um3"] == 30000.0
    assert rows[1]["analysis_included"] is False
    assert rows[1]["area_denominator_valid"] is False


def test_artifact_declares_specimen_unit_and_suppresses_inference(tmp_path):
    module = _module()
    source = tmp_path / "qc.csv"
    source.write_text("sample_id,group,status\na,WT,complete\n", encoding="utf-8")
    out_csv, out_json = tmp_path / "sensitivity.csv", tmp_path / "sensitivity.json"
    module.write_artifact(source, out_csv, out_json)
    import json
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["analysis_unit"] == "biological_specimen"
    assert payload["inference_performed"] is False
    assert payload["included_specimen_count"] == 1
    assert payload["excluded_or_missing_specimen_count"] == 0
    assert "sampled ROI volume is not anatomical organ volume" in (
        payload["scenario_interpretation"]["normalization"]
    )
