"""Build v5 of the illustrated workflow by extending v4 rather than rebuilding it.

v4 documents the workflow from ROI-aware preprocessing through study management,
and that description is still accurate. What changed in 2026-09 is how width,
area, volume and merged objects are measured. This script opens v4, retitles it,
and appends a section covering those changes in plain language for a biologist,
with figures generated from real measurements.

Usage:
    python scripts/build_v571_workflow_v5_document.py
"""

import argparse
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Saturn_V5.7_Illustrated_Technical_Workflow_v4.docx"
FIGURES = ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"
OUTPUT = ROOT / "Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx"

BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x6B, 0x72, 0x80)


def heading(document, text, level=1):
    paragraph = document.add_heading(text, level=level)
    for run in paragraph.runs:
        run.font.color.rgb = BLUE
    return paragraph


def body(document, text):
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def bullet(document, text):
    return document.add_paragraph(text, style="List Bullet")


def figure(document, filename, caption, width=6.6):
    path = FIGURES / filename
    if not path.is_file():
        raise FileNotFoundError(f"figure missing: {path}")
    document.add_picture(str(path), width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    run.font.size = Pt(8.5)
    run.font.italic = True
    run.font.color.rgb = GREY
    paragraph.paragraph_format.space_after = Pt(14)


def callout(document, text):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.font.color.rgb = BLUE
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(10)


def build(source=SOURCE, output=OUTPUT):
    document = Document(str(source))

    # Retitle in place so the document announces itself correctly.
    for paragraph in document.paragraphs:
        if paragraph.style.name == "Title":
            for run in paragraph.runs:
                run.text = ""
            paragraph.runs[0].text = "Saturn v5.7.1 Illustrated Technical Workflow"
            break

    document.add_page_break()
    heading(document, "13. What Changed in v5.7.1: Measuring Width From the Signal")
    body(
        document,
        "Everything above still describes how the workflow runs. This section "
        "covers what changed in September 2026, and it matters because it "
        "changes what the width and count numbers mean. The short version is "
        "that the computer was measuring the outline it had drawn around each "
        "nucleus rather than the nucleus itself, and it was treating some "
        "clumps of nuclei as single objects."
    )

    heading(document, "13.1 What a nucleus actually looks like", level=2)
    body(
        document,
        "These are real sperm nuclei from one optical slice. Each panel is the "
        "same size. The top row is the raw image; the bottom row adds the "
        "outline the computer drew and the line along which the nucleus is "
        "measured."
    )
    figure(
        document, "v5_fig00_clean_examples.png",
        "Figure 13.1. Six clean, well-separated nuclei. The orange outline sits "
        "noticeably outside the bright glow of the nucleus.",
    )
    callout(
        document,
        "That gap between the orange outline and the bright glow is the whole "
        "problem.",
    )

    heading(document, "13.2 Two ways of measuring width disagree", level=2)
    body(
        document,
        "We can measure width two ways. The old way measures across the outline "
        "the computer drew. The new way measures across the actual brightness "
        "of the nucleus, taking the width at half of its peak brightness. The "
        "two do not agree, and they do not disagree by the same amount in every "
        "sample, which is the part that breaks a comparison between genotypes."
    )
    figure(
        document, "v5_fig01_mask_versus_signal.png",
        "Figure 13.2. The outline reports roughly twice the width of the glow, "
        "and the factor differs between the two samples.",
    )
    body(
        document,
        "If the outline were simply too wide by a fixed amount, we could correct "
        "for it. It is not fixed: it was 2.12 times too wide in one sample and "
        "2.29 in the other. That difference between samples is larger than the "
        "difference we would be trying to detect between genotypes."
    )

    heading(document, "13.3 Why the outline is too wide", level=2)
    body(
        document,
        "This is not a fault in the model. The model was trained on outlines "
        "drawn by hand, and those hand-drawn outlines are about two and a half "
        "times wider than the nucleus appears in the image. A very thin, faint "
        "object is hard to trace tightly by hand, so the training outlines are "
        "generous. The model learned that convention faithfully."
    )
    figure(
        document, "v5_fig02_annotation_convention.png",
        "Figure 13.3. The model reproduces the width it was taught, not the "
        "width of the nucleus.",
    )

    heading(document, "13.4 What the microscope can and cannot resolve", level=2)
    body(
        document,
        "There is a hard limit underneath all of this. At the settings used, one "
        "pixel is 0.378 microns across, while a sperm nucleus is roughly half a "
        "micron wide. A nucleus is therefore only about one and a third pixels "
        "wide, which is too few to measure a width properly. The Z step is not "
        "the problem and should be left as it is."
    )
    figure(
        document, "v5_fig03_sampling_limit.png",
        "Figure 13.4. Sideways sampling is the limit. Below about one micron, "
        "real differences in width are squashed into a much smaller measured "
        "difference.",
    )
    callout(
        document,
        "Comparing genotypes against each other is valid. Quoting an absolute "
        "nucleus diameter in microns is not.",
    )
    body(
        document,
        "For future imaging, increasing the zoom to about 3 would make a nucleus "
        "span roughly five pixels instead of one, which would turn width into a "
        "measurement rather than a comparison. Recording in 12-bit rather than "
        "8-bit and using line averaging of two to four would also help. Laser "
        "power, gain, pinhole and zoom must stay identical across every specimen "
        "in a study."
    )

    heading(document, "13.5 A second, more sensitive measure", level=2)
    body(
        document,
        "Alongside the width of the glow we now also record its total "
        "brightness. Blurring spreads light out but does not destroy it, so "
        "total brightness reflects how much material is present and does not "
        "get squashed the way width does. It is far more sensitive to a real "
        "change in size."
    )
    figure(
        document, "v5_fig04_two_measures.png",
        "Figure 13.5. Total brightness is far better at detecting a real change, "
        "but it is also fooled by a brighter stain. Width is not.",
    )
    body(
        document,
        "Because this study has no separate staining control, total brightness "
        "cannot be used on its own to claim a biological difference: a brighter "
        "stain would look the same as a bigger nucleus. It is recorded as a "
        "supporting check. When it agrees with the width measurement that is "
        "reassuring, and when it disagrees that is a signal to look closer."
    )

    heading(document, "13.6 Nuclei that were being counted as one", level=2)
    body(
        document,
        "Separately, the software was treating some groups of touching nuclei as "
        "a single object. The old rule only called something a clump if it was "
        "longer than 20 microns, but a typical nucleus here is about 8 microns, "
        "so two stuck end to end come to roughly 16 and slipped under the "
        "threshold. The rule now looks at the shape instead: if the skeleton of "
        "an object branches, or if a cut across it shows two separate peaks of "
        "brightness, it is treated as more than one nucleus and separated."
    )
    figure(
        document, "v5_fig05_merge_correction.png",
        "Figure 13.6. The old length cut-off sat above where clumps of two "
        "nuclei actually fall, so almost none were caught.",
    )
    callout(
        document,
        "Counts rose by about 10 percent, and by more in KJ than in WT, so the "
        "old undercounting was not affecting both groups equally.",
    )
    body(
        document,
        "An object is only separated when the shape gives a reason to. If the "
        "pieces that would result do not look like plausible nuclei, the object "
        "is left exactly as it was. Nothing is deleted and no pixel is added or "
        "removed."
    )

    heading(document, "13.7 When a width cannot be measured", level=2)
    body(
        document,
        "About a fifth of objects are skipped for width, most often because a "
        "neighbouring nucleus is too close for a clean measurement. Skipping "
        "them is deliberate: a contaminated measurement is worse than none. The "
        "risk is that skipping could quietly favour one genotype, so it was "
        "checked across all 35 specimens."
    )
    figure(
        document, "v5_fig06_availability_bias.png",
        "Figure 13.7. The same fraction is skipped in both groups, so the "
        "comparison is not distorted by it.",
    )

    heading(document, "13.8 What this means for interpreting results", level=2)
    bullet(document,
           "Width is a comparison between groups. It is not an absolute "
           "measurement of how wide a nucleus is, and should never be quoted as "
           "one.")
    bullet(document,
           "Length, counts, tracking and Z extent are unaffected by the width "
           "problem and can be read as before.")
    bullet(document,
           "Counts are about 10 percent higher than in earlier reports because "
           "clumped nuclei are now separated. Older count numbers should not be "
           "compared directly against new ones.")
    bullet(document,
           "Total brightness is a supporting check only, because this study has "
           "no staining control.")
    bullet(document,
           "None of this has been through independent review yet. The software "
           "blocks the biological comparison report until it has.")

    body(
        document,
        f"Generated {date.today().isoformat()}. Every figure in this section was "
        "produced from real measurements on the KJ and WT specimens and is "
        "recorded in the project audit evidence."
    )

    document.save(str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    written = build(Path(args.source), Path(args.output))
    print(f"wrote {written}  ({written.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
