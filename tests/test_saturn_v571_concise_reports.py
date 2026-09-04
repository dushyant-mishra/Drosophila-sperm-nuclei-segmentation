import importlib.util
from pathlib import Path
import zipfile

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    path = ROOT / "sperm_segmentation_saturnv5.7.1.py"
    spec = importlib.util.spec_from_file_location("saturn_v571_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_tracks():
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "technical_valid": [True, True, False],
            "representative_body_length_um": [8.0, 10.0, 20.0],
            "representative_body_width_um": [2.0, 2.5, 5.0],
            "length_body_width_ratio": [4.0, 4.0, 4.0],
            "representative_section_tortuosity": [1.0, 1.1, 2.0],
            "tortuosity_3d": [1.8, 2.0, 3.0],
            "projection_z_extent_um": [9.0, 11.0, 22.0],
            "observed_slab_effective_thickness_um": [3.0, 3.2, 6.0],
            "observed_slice_mask_volume_um3": [30.0, 40.0, 100.0],
            "pitch_deg": [5.0, 10.0, 30.0],
            "taper_ratio": [1.1, 1.2, 2.5],
            "nearest_neighbor_um": [4.0, 5.0, 8.0],
        }
    )


def test_concise_pdf_and_ppt_contain_only_actionable_biological_measurements(tmp_path):
    saturn = load_saturn()
    pdf_path = Path(saturn.generate_concise_biologist_pdf(tmp_path, primary_tracks()))
    ppt_path = Path(saturn.generate_concise_biologist_pptx(tmp_path))

    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 2
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    with zipfile.ZipFile(ppt_path) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        ppt_text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in slide_names
        ).lower()
    assert len(slide_names) == 2

    required = (
        "estimated unique nuclei",
        "representative-section length",
        "apparent body-mask width",
        "length / body width",
        "representative-section tortuosity",
    )
    forbidden = (
        "projection + z",
        "effective thickness",
        "slab",
        "volume",
        "pitch",
        "taper",
        "nearest neighbor",
        "p90",
        "iqr",
        "legacy",
        "density",
    )
    for label in required:
        assert label in pdf_text
    for label in forbidden:
        assert label not in pdf_text
        assert label not in ppt_text


def test_concise_report_uses_only_technical_valid_population(tmp_path):
    saturn = load_saturn()
    pdf_path = Path(saturn.generate_concise_biologist_pdf(tmp_path, primary_tracks()))
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    assert "Estimated unique nuclei (coverage-sensitive)\n2" in text
    assert "20.0" not in text
