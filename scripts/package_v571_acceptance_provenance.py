"""Package compact, portable v5.7.1 calibration and replay provenance."""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILES = (
    "runtime_parameters.json",
    "runtime_environment.json",
    "analysis_profile_used.json",
    "calibration_used.json",
    "microscope_metadata_used.xml",
    "roi_mask_source.npy",
    "settings_manifest.json",
)

CHANNEL_PATTERN = re.compile(r"(?:^|[_ .-])(?:ch|c)(?P<channel>\d+)(?:[_.-]|$)", re.I)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def resolve_source_channel_manifest(payload):
    resolved = dict(payload)
    records = []
    for source_record in payload.get("ordered_source_images", []):
        record = dict(source_record)
        match = CHANNEL_PATTERN.search(str(record.get("name", "")))
        if match is None:
            channel = record.get("channel")
            if channel is None:
                raise ValueError(
                    "Source channel cannot be resolved from manifest or filename: "
                    f"{record.get('name', '')}"
                )
            source = "source_manifest:explicit_channel"
        else:
            channel = int(match.group("channel"))
            source = "filename:explicit_channel"
        if int(channel) != 0:
            raise ValueError(
                f"Acceptance source must resolve to channel 0: {record.get('name')}"
            )
        record["channel"] = 0
        record["channel_resolution_source"] = source
        record["channel_selection_rule"] = "accepted source channel is ch00"
        records.append(record)
    resolved["channel_selection_rule"] = "all accepted source images resolve to ch00"
    resolved["channel_selection_source"] = (
        "archived manifest field cross-checked against filename when available"
    )
    resolved["ordered_source_images"] = records
    return resolved


def completed_samples(study_output):
    for sample_dir in sorted((Path(study_output) / "samples").iterdir()):
        attempts = sorted(sample_dir.glob("attempt_*"))
        complete = [path for path in attempts if (path / "sample_complete.json").is_file()]
        if complete:
            yield sample_dir.name, complete[-1]


def write_replay_zip(destination, members):
    member_records = []
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for archive_name, source in members:
            source = Path(source)
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            content = source.read_bytes()
            archive.writestr(info, content)
            member_records.append(
                {
                    "archive_name": archive_name,
                    "source_path": str(source.resolve()),
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    return member_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-output", required=True)
    parser.add_argument("--tracking-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay_root = Path(args.tracking_replay).resolve()
    specimens = []
    replay_members = []

    for sample_id, attempt in completed_samples(args.study_output):
        settings_source = attempt / "settings"
        settings_destination = output / "specimens" / sample_id / "settings"
        settings_destination.mkdir(parents=True, exist_ok=True)
        retained = []
        for name in SETTINGS_FILES:
            source = settings_source / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing required provenance file: {source}")
            destination = settings_destination / name
            shutil.copy2(source, destination)
            retained.append(
                {
                    "role": name,
                    "repository_path": repository_path(destination),
                    "sha256": sha256(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )

        original_source_manifest = settings_source / "source_image_manifest.json"
        if not original_source_manifest.is_file():
            raise FileNotFoundError(
                f"Missing required provenance file: {original_source_manifest}"
            )
        original_destination = settings_destination / "source_image_manifest_original.json"
        shutil.copy2(original_source_manifest, original_destination)
        resolved_destination = settings_destination / "source_image_manifest.json"
        resolved_payload = resolve_source_channel_manifest(
            json.loads(original_source_manifest.read_text(encoding="utf-8"))
        )
        resolved_destination.write_text(
            json.dumps(resolved_payload, indent=2) + "\n", encoding="utf-8"
        )
        for role, destination in (
            ("source_image_manifest_original", original_destination),
            ("source_image_manifest_resolved", resolved_destination),
        ):
            retained.append(
                {
                    "role": role,
                    "repository_path": repository_path(destination),
                    "sha256": sha256(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )

        source_2d = next(attempt.glob("spermatid_measurements_*.csv"))
        tracked = replay_root / f"{sample_id}_production_morphology_neutral_tracked.csv"
        tracks = replay_root / f"{sample_id}_production_morphology_neutral_tracks.csv"
        for role, source in (
            ("source_2d_detections", source_2d),
            ("tracked_detections", tracked),
            ("track_summary", tracks),
        ):
            if not source.is_file():
                raise FileNotFoundError(f"Missing replay file: {source}")
            replay_members.append((f"{sample_id}/{role}.csv", source))

        calibration = json.loads(
            (settings_destination / "calibration_used.json").read_text(
                encoding="utf-8"
            )
        )
        specimens.append(
            {
                "sample_id": sample_id,
                "xy_um_per_pixel": calibration["xy_um_per_pixel"],
                "z_um_per_slice": calibration["z_um_per_slice"],
                "calibration_status": calibration["status"],
                "settings": retained,
            }
        )

    replay_zip = output / "tracking_replay_inputs_outputs.zip"
    replay_records = write_replay_zip(replay_zip, replay_members)
    checkpoint = ROOT / "model_checkpoints" / "v571_model_c_dual_head_epoch003.pt"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_repository_path": repository_path(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "specimens": specimens,
        "replay_archive_repository_path": repository_path(replay_zip),
        "replay_archive_sha256": sha256(replay_zip),
        "replay_members": replay_records,
    }
    (output / "acceptance_provenance_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output / "acceptance_provenance_manifest.json")


if __name__ == "__main__":
    main()
