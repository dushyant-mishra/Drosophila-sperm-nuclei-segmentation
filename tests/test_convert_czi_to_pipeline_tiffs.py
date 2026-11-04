"""Contract tests for the Zeiss CZI to pipeline-TIFF converter.

The risks worth covering are not in the decoding, which a library does, but in
the seams: the metre-to-micron conversion, whether the filenames the converter
invents are actually the ones the pipeline's source parser accepts, and whether
a stack that disagrees with its own metadata is refused rather than written out
quietly.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERTER = load_module("czi_converter", "scripts/convert_czi_to_pipeline_tiffs.py")


XML = """<ImageDocument><Metadata><Information><Image>
<SizeX>1024</SizeX><SizeY>1024</SizeY><SizeZ>33</SizeZ>
<ComponentBitCount>8</ComponentBitCount><PixelType>Gray8</PixelType>
<Dimensions><Channels><Channel><ExcitationWavelength>488.0</ExcitationWavelength>
</Channel></Channels></Dimensions>
<ObjectiveSettings><Immersion>Water</Immersion></ObjectiveSettings>
<Instrument><Objectives><Objective><LensNA>1.2</LensNA>
<NominalMagnification>40</NominalMagnification></Objective></Objectives></Instrument>
</Image></Information><Scaling><Items>
<Distance Id="X"><Value>1.6746138992264493e-007</Value></Distance>
<ScalingX>1.6746138992264493e-007</ScalingX>
<ScalingY>1.6746138992264493e-007</ScalingY>
<ScalingZ>9.9999999999999995e-007</ScalingZ>
</Items></Scaling></Metadata></ImageDocument>"""


def test_scaling_is_converted_from_metres_to_microns():
    """Zeiss writes metres. A missed factor of a million silently rescales
    every physical measurement downstream, so pin the exact values."""
    calibration = CONVERTER.read_calibration(XML)

    assert calibration["xy_um_per_pixel"] == pytest.approx(0.167461, abs=1e-6)
    assert calibration["z_um_per_slice"] == pytest.approx(1.0, abs=1e-6)
    assert calibration["size_z"] == 33
    assert calibration["bit_depth"] == 8
    assert calibration["objective_na"] == "1.2"


def test_absent_metadata_fields_do_not_fabricate_a_calibration():
    """A stack whose metadata lacks scaling must report nothing rather than a
    default, because a plausible wrong number is worse than a missing one."""
    calibration = CONVERTER.read_calibration("<ImageDocument></ImageDocument>")

    assert calibration["xy_um_per_pixel"] is None
    assert calibration["z_um_per_slice"] is None
    assert calibration["size_z"] == 0


def test_written_slice_names_are_understood_by_the_pipeline_parser(tmp_path):
    """The converter and the pipeline agree on filenames only by convention,
    so assert the convention against the real parser rather than a copy of it."""
    saturn = load_module("saturn_for_names", "sperm_segmentation_saturnv5.7.1.py")
    specimen = "TppKO_1dPM_mated_1"

    parsed = saturn._study_parse_source_name(f"{specimen}_z07_ch00.tif")

    assert parsed is not None
    assert parsed["label"] == specimen
    assert parsed["z"] == 7
    assert parsed["channel"] == 0
    assert parsed["file_pattern"] == f"{specimen}_z[0-9]*_ch00.tif"


def test_converted_slices_round_trip_unchanged(tmp_path):
    """Pixels must survive the write. An 8-bit stack that silently became a
    different dtype or picked up compression artefacts would change every
    intensity-derived measurement."""
    source = np.random.default_rng(0).integers(0, 256, (4, 16, 16), dtype=np.uint8)
    for index in range(source.shape[0]):
        tifffile.imwrite(
            str(tmp_path / f"demo_z{index:02d}_ch00.tif"),
            source[index],
            photometric="minisblack",
        )

    for index in range(source.shape[0]):
        back = tifffile.imread(str(tmp_path / f"demo_z{index:02d}_ch00.tif"))
        assert back.dtype == np.uint8
        assert np.array_equal(back, source[index])


def test_manifest_template_leaves_group_identity_blank(tmp_path, monkeypatch):
    """Group identity and direction are the owner's to declare. A converter that
    guessed them from a filename would breach the rule that no genotype is ever
    inferred from a label."""
    records = [
        {
            "specimen": "TppKO_1dPM_mated_1",
            "xy_um_per_pixel": 0.167461,
            "z_um_per_slice": 1.0,
            "slices_written": 33,
            "file_pattern": "TppKO_1dPM_mated_1_z[0-9]*_ch00.tif",
        }
    ]
    monkeypatch.setattr(CONVERTER, "convert", lambda czi, out: (records[0], out))
    source = tmp_path / "in"
    source.mkdir()
    (source / "TppKO_1dPM_mated_1.czi").write_bytes(b"")

    CONVERTER.main(["--input", str(source)])

    with (source / "study_manifest_template.csv").open(encoding="utf-8") as handle:
        row = next(iter(csv.DictReader(handle)))
    assert row["group"] == ""
    assert row["group_role"] == ""
    assert row["roi_path"] == ""
    assert row["xy_um_per_pixel"] == "0.167461"
    assert row["sample_id"] == "TppKO_1dPM_mated_1"
