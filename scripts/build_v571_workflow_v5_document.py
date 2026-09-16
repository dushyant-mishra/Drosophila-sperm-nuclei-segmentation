"""Build v5 of the illustrated workflow: a full rewrite, not an append.

v4 described the workflow well but is now out of date in ways that change what
the numbers mean. It measured width across the outline the model draws rather
than across the nucleus, it counted clumped nuclei as single objects, and its
whole demonstration came from a different, coarser dataset than the KJ and WT
study this project is actually about.

v5 keeps v4's section order, because that order is good, and rewrites the
content for a biologist reading it to interpret results. Every figure and every
number comes from the KJ and WT specimens.

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
NEW_FIGS = ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"
OLD_FIGS = ROOT / "docs" / "v5_7_illustrated_workflow" / "figures"
OUTPUT = ROOT / "Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx"

BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x6B, 0x72, 0x80)
WARN = RGBColor(0xB9, 0x1C, 0x1C)


def title(document, text, subtitle):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = BLUE
    sub = document.add_paragraph()
    run = sub.add_run(subtitle)
    run.font.size = Pt(11)
    run.font.color.rgb = GREY
    sub.paragraph_format.space_after = Pt(16)


def heading(document, text, level=1):
    paragraph = document.add_heading(text, level=level)
    for run in paragraph.runs:
        run.font.color.rgb = BLUE
    return paragraph


def body(document, text):
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(7)
    return paragraph


def bullet(document, text):
    return document.add_paragraph(text, style="List Bullet")


def numbered(document, text):
    return document.add_paragraph(text, style="List Number")


def callout(document, text, colour=BLUE):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.font.color.rgb = colour
    paragraph.paragraph_format.space_before = Pt(7)
    paragraph.paragraph_format.space_after = Pt(11)


def figure(document, folder, filename, caption, width=6.5):
    path = folder / filename
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


def table(document, header, rows, widths=None):
    grid = document.add_table(rows=1, cols=len(header))
    grid.style = "Light Grid Accent 1"
    for cell, text in zip(grid.rows[0].cells, header):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
    for values in rows:
        cells = grid.add_row().cells
        for cell, text in zip(cells, values):
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.size = Pt(9)
    document.add_paragraph().paragraph_format.space_after = Pt(10)
    return grid


def build(output=OUTPUT):
    document = Document()

    title(
        document,
        "Saturn v5.7.1 Illustrated Technical Workflow",
        "How Drosophila sperm nuclei are found, measured and counted, and how to "
        "read the results. Written for the biologist using the output.",
    )

    callout(
        document,
        "New in this version: width is measured directly from each nucleus's own "
        "fluorescence rather than from the detected outline, touching nuclei are "
        "separated into individual objects, and a study can compare one reference "
        "group against several mutant or rescue lines at once.",
    )

    table(
        document,
        ["This document describes", "Value"],
        [
            ["Pipeline", "Saturn v5.7.1"],
            ["Specimens illustrated", "KJ-01 and WT-01, from the current study"],
            ["Image size", "1024 x 1024 pixels per slice"],
            ["Pixel size", "0.378 um across, 0.346 um between slices"],
            ["Microscope", "Leica DMI8-CS confocal, 40x / 1.30 oil, pinhole 1.0 Airy"],
            ["Detection", "Dual-head 2.5D neural network"],
        ],
    )

    # ------------------------------------------------------------------ 1
    heading(document, "1. What Saturn Does")
    body(
        document,
        "Saturn takes a confocal stack through a Drosophila seminal vesicle and "
        "reports how many sperm nuclei are present and what shape they are. These "
        "nuclei are long, thin and tightly packed, which makes them genuinely "
        "difficult to count by hand: the same nucleus appears on several slices, "
        "and neighbouring nuclei frequently touch. Saturn handles both."
    )
    bullet(document, "You draw a region of interest once, and everything outside "
                     "it is ignored, so bright tissue elsewhere cannot skew the result.")
    bullet(document, "A trained neural network finds the nuclei on each slice, "
                     "using the slices above and below for context.")
    bullet(document, "Each nucleus is reduced to a centre line, and its length, "
                     "width and curvature are measured from that line.")
    bullet(document, "Detections are linked through depth, so one nucleus spanning "
                     "five slices is counted once rather than five times.")
    bullet(document, "Touching nuclei are recognised and separated into individual "
                     "objects.")
    bullet(document, "Specimens are analysed independently, and group comparisons "
                     "treat the specimen as the replicate.")
    callout(
        document,
        "The one rule worth remembering: the specimen is the replicate. Thousands "
        "of nuclei from one animal are still one animal.",
    )

    # ------------------------------------------------------------------ 2
    heading(document, "2. Loading a Stack and Drawing the Region")
    body(
        document,
        "The interface opens a stack, lets you move through the slices, and lets "
        "you draw or reload the region of interest. It is worth checking the "
        "region on an early, a middle and a late slice, because the tissue shifts "
        "through depth."
    )
    figure(document, NEW_FIGS, "v5_fig10_region_across_slices.png",
           "Figure 2.1. The same drawn region applied to four consecutive slices "
           "of specimen KJ-01. Pixels outside the red boundary take no part in "
           "the analysis.", width=6.8)

    # ------------------------------------------------------------------ 3
    heading(document, "3. Preparing Each Slice")
    body(
        document,
        "Each slice is normalised using brightness statistics gathered from inside "
        "the region only, then denoised and background-corrected. Gathering those "
        "statistics inside the region is what keeps a bright piece of tissue just "
        "outside it from setting the thresholds and making real nuclei look dim."
    )
    figure(document, NEW_FIGS, "v5_fig11_processing_stages.png",
           "Figure 3.1. One slice of KJ-01 from raw image through to the "
           "individual nuclei the network identifies.", width=6.2)

    # ------------------------------------------------------------------ 4
    heading(document, "4. Finding the Nuclei")
    body(
        document,
        "Detection uses a neural network that reads three slices at once: the one "
        "being analysed plus the ones above and below. That context helps it "
        "recognise a nucleus that is only partly in focus. It predicts two things, "
        "and the second is what makes dense packing tractable."
    )
    figure(document, NEW_FIGS, "v5_fig12_network_context.png",
           "Figure 4.1. The three slices the network reads, how confident it is "
           "that a nucleus is present, and where the dense core of each nucleus "
           "lies. The core prediction is what separates two touching nuclei.",
           width=6.9)
    body(
        document,
        "Detection does not require a nucleus to look normal. An unusually long, "
        "short, wide or bent nucleus is detected and measured like any other, and "
        "simply carries a note that it is unusual. This matters for mutant work: "
        "a pipeline that quietly discarded unusual shapes would hide exactly what "
        "you are looking for."
    )

    # ------------------------------------------------------------------ 5
    heading(document, "5. Measuring a Nucleus")
    body(
        document,
        "Once a nucleus has been found, three things are measured from it: how "
        "long it is, how curved it is, and how wide it is. All three come from "
        "the same starting point, the centre line running down the middle of the "
        "nucleus, and all three are measured on the image itself rather than on "
        "the outline the network drew around it."
    )
    figure(document, NEW_FIGS, "v5_fig20_how_measured.png",
           "Figure 5.1. Length and width on one typical nucleus from KJ-01. "
           "Left: length runs along the centre line from tip to tip, with the "
           "straight tip-to-tip distance shown for comparison. Middle: width is "
           "sampled on cuts at right angles to the centre line, taken along the "
           "middle three quarters of the nucleus. Right: the brightness across "
           "the marked cut, measured at half its peak.",
           width=6.9)

    heading(document, "5.1 Length, and why it is not the straight-line distance", level=2)
    body(
        document,
        "Saturn thins each nucleus down to a one-pixel centre line and then walks "
        "along it from one tip to the other, adding up the distance step by step. "
        "Measuring along the centre line rather than straight from tip to tip "
        "matters because these nuclei are often gently bent: a straight-line "
        "measurement would report a bent nucleus as shorter than an identical "
        "straight one. In the example above the two differ by about four percent."
    )
    body(
        document,
        "Dividing the centre-line length by the straight tip-to-tip distance gives "
        "the curvature. A perfectly straight nucleus scores 1.00, and the more "
        "bent it is the higher the number climbs. Because it is one length divided "
        "by another, curvature has no units and is unaffected by calibration."
    )

    heading(document, "5.2 Width, measured from the nucleus rather than its outline", level=2)
    body(
        document,
        "At every point along the middle of the centre line, Saturn samples the "
        "raw brightness on a line at right angles to it. The background is read "
        "from the quiet ends of that same line, so no separate background region "
        "and no expansion of the mask is involved. The width is then the distance "
        "between the two points where the brightness has fallen to half its peak. "
        "That distance is measured on every cut along the middle three quarters "
        "of the nucleus, typically sixteen of them and never fewer than five, and "
        "the middle value is what gets reported, so a single odd cut cannot move "
        "the result."
    )
    body(
        document,
        "This measures the nucleus, not the outline around it. The outline still "
        "has a job to do: it says which pixels belong to this nucleus, so that a "
        "neighbour a micron away cannot contribute to the profile. But how "
        "generously or tightly that outline was drawn does not change the number, "
        "which is what makes the measurement stable."
    )
    figure(document, NEW_FIGS, "v5_fig_hero_neighbourhood.png",
           "Figure 5.2. Five neighbouring nuclei in a crowded field of KJ-01, each "
           "measured on its own perpendicular cut. The closest pair are under two "
           "microns apart, and they still give separate, closely agreeing widths.",
           width=6.9)
    figure(document, NEW_FIGS, "v5_fig00_clean_examples.png",
           "Figure 5.3. Six well-separated nuclei with the measurement applied. "
           "Widths cluster tightly between about 0.63 and 0.67 microns.", width=6.9)
    callout(
        document,
        "These nuclei are thinner than the microscope can fully resolve, so width "
        "is the right measure for comparing one group against another, and the "
        "wrong one for quoting an absolute diameter. Section 7 says which of "
        "Saturn's numbers fall into each category.",
    )

    # ------------------------------------------------------------------ 6
    heading(document, "6. Counting: One Nucleus, Not Five Slices")
    body(
        document,
        "A confocal stack cuts each nucleus into several optical slices, so the "
        "same nucleus is found again and again on consecutive planes. Adding up "
        "the per-slice detections would therefore overstate the count several-fold. "
        "Saturn instead joins the detections that belong together and counts the "
        "joined objects."
    )
    figure(document, NEW_FIGS, "v5_fig13_through_depth.png",
           "Figure 6.1. The same small area of KJ-01 across five consecutive "
           "slices. The same nuclei recur on plane after plane.", width=6.9)

    heading(document, "6.1 How detections are joined through depth", level=2)
    body(
        document,
        "Moving from one slice to the next, Saturn compares every detection on the "
        "new slice against every detection on the previous one and picks the set "
        "of pairings that fits best overall, rather than greedily matching each "
        "nucleus to its nearest neighbour in turn. A pairing is judged on how far "
        "the nucleus has moved, how much its footprint overlaps the previous "
        "one, whether it still points in the same direction, and how strongly "
        "the network supports it. Since a nucleus shifts by only a few tenths of "
        "a micron between neighbouring slices, these agreements are decisive in "
        "practice."
    )
    body(
        document,
        "If a nucleus is missing from a single slice, perhaps because it dipped "
        "briefly out of focus, the link can be carried across that one gap and "
        "resumed on the next slice. Nothing is invented for the missing slice: it "
        "contributes no measurement, and the gap is recorded. A gap of two or more "
        "slices ends the object, and what follows is treated as a separate nucleus."
    )
    figure(document, NEW_FIGS, "v5_fig21_joining_depth.png",
           "Figure 6.2. Left: individual nuclei of KJ-01 followed through five "
           "consecutive slices, each coloured on every plane it appears on, "
           "with the detections judged to be the same nucleus linked by a "
           "dashed line. Right: the movement between neighbouring slices sits "
           "far inside the limit allowed, and joining turns 26,651 slice-level "
           "detections into 5,766 counted nuclei in this specimen.", width=6.9)
    body(
        document,
        "Deliberately, size and shape play no part in the decision. Length, width "
        "and area are all given zero weight when a study is run in comparative "
        "mode, so that a mutant whose nuclei are a different size cannot end up "
        "joined more or less readily than the wild type. Joining is decided on "
        "where the object is, not on what it looks like, which keeps the count "
        "independent of the phenotype being measured."
    )
    body(
        document,
        "The joined object is what carries the reported measurements. Its length "
        "and width are taken from the one slice on which it appeared largest, so a "
        "plane that caught only its edge cannot dilute the value, and the slice "
        "used is recorded alongside the number. Its depth extent comes from how "
        "many slices it spanned."
    )

    heading(document, "6.2 Separating touching nuclei", level=2)
    body(
        document,
        "Because these nuclei are so tightly packed, a detected outline sometimes "
        "encloses two or more of them. Saturn now recognises this from the shape "
        "of the object: if its skeleton branches, or a cut across it shows two "
        "separate brightness peaks, it is treated as more than one nucleus and "
        "separated into its parts."
    )
    figure(document, NEW_FIGS, "v5_fig05_merge_correction.png",
           "Figure 6.3. Recognising clumps from their shape rather than from "
           "length alone identifies far more of them, which makes the count more "
           "accurate.", width=6.9)
    body(
        document,
        "The separation is conservative by design. An object is only divided when "
        "its shape gives an objective reason, and if the resulting pieces would "
        "not be plausible nuclei it is left exactly as it was. Nothing is deleted, "
        "and no pixel is added or removed. The practical effect is that counts are "
        "around ten percent higher than earlier versions reported, and closer to "
        "the true number."
    )

    heading(document, "6.3 Only clean measurements are reported", level=2)
    body(
        document,
        "A width is only reported when the brightness is seen to fall to half "
        "its peak on both sides while still inside the nucleus. Where it does "
        "not, usually because a neighbour sits inside the profile window, "
        "Saturn reports no width for that object rather than a contaminated "
        "one. This affects about one object in five in this densely packed "
        "tissue. Length and counting are unaffected and the object is still "
        "counted; only the width is withheld."
    )
    figure(document, NEW_FIGS, "v5_fig06_availability_bias.png",
           "Figure 6.4. Left: why a width is withheld. In three cases out of "
           "four the signal has not fallen to half by the edge of the nucleus, "
           "which is what a close neighbour does. Right: the same proportion is "
           "withheld in both groups, so comparisons between them stay balanced.")

    # ------------------------------------------------------------------ 7
    heading(document, "7. Every Number Saturn Reports, and How to Use It")
    body(
        document,
        "Saturn reports rather more than length, width and count. This section "
        "lists what is in the output tables and, for each one, says how far it can "
        "be pushed. Three categories are used throughout."
    )
    table(
        document,
        ["Category", "What it means"],
        [
            ["Absolute",
             "The number is a real physical quantity. Quote it directly, plot it, "
             "put it in a paper."],
            ["Comparison",
             "Valid for comparing one group against another under matched "
             "acquisition, but not to be quoted as a physical value on its own."],
            ["Diagnostic",
             "A technical quality check. Useful for deciding whether a run went "
             "well; not a biological result."],
        ],
    )

    heading(document, "7.1 Size and shape of each nucleus", level=2)
    table(
        document,
        ["Measure", "What it is", "Use"],
        [
            ["Length", "Distance along the centre line, on the slice where the "
                       "nucleus appeared largest, in microns.", "Absolute"],
            ["Curvature", "Centre-line length divided by straight tip-to-tip "
                          "distance. 1.00 is straight.", "Absolute"],
            ["3D curvature", "The same ratio applied to the path the nucleus's "
                             "centre point traces through the stack.",
             "Absolute"],
            ["Signal width", "Width of the nucleus's fluorescence at half its peak, "
                             "in microns.", "Comparison"],
            ["Outline width", "Width of the outline the network drew, in microns.",
             "Diagnostic"],
            ["Length / width", "Slenderness. Its width term carries the same limit "
                               "as width itself.", "Comparison"],
            ["Taper", "Largest slice area divided by smallest, across the slices "
                      "the nucleus spans.", "Comparison"],
            ["Depth span", "How far the nucleus reaches through the stack, in "
                           "microns.", "Comparison"],
            ["Slices spanned", "How many optical slices the nucleus was found on.",
             "Absolute"],
            ["Projection + depth extent", "Lateral extent and depth span combined "
                                          "into one distance.", "Comparison"],
            ["Footprint area", "Length multiplied by width, in square microns.",
             "Comparison"],
            ["Mask volume", "Outlined area on each slice, added up through depth, "
                            "in cubic microns.", "Comparison"],
            ["Signal volume", "Length times signal width on each slice, added "
                              "up through depth, in cubic microns.", "Comparison"],
        ],
    )
    body(
        document,
        "Length, curvature and slice count are absolute because each is either a "
        "distance the microscope resolves comfortably or a ratio in which the "
        "calibration cancels out. Anything with width in it inherits width's "
        "limit: these nuclei are thinner than the microscope resolves, so the "
        "measured width is an upper bound. It responds to real differences "
        "between groups, which is what makes it useful, but the number itself "
        "should not be read as a diameter."
    )

    heading(document, "7.2 Orientation and neighbourhood", level=2)
    table(
        document,
        ["Measure", "What it is", "Use"],
        [
            ["Yaw", "The direction the nucleus points within the image plane, in "
                    "degrees.", "Absolute"],
            ["Pitch", "How steeply it tilts through the stack, in degrees.",
             "Comparison"],
            ["Nearest neighbour", "Centre-to-centre distance to the closest "
                                  "other nucleus, in microns.", "Absolute"],
        ],
    )
    body(
        document,
        "Yaw and nearest-neighbour distance are ordinary in-plane measurements. "
        "Pitch mixes an in-plane distance with a depth distance, and depth is "
        "sampled and resolved differently from the image plane, so pitch is best "
        "compared between groups rather than quoted."
    )

    heading(document, "7.3 Counts and densities, per specimen", level=2)
    table(
        document,
        ["Measure", "What it is", "Use"],
        [
            ["Nuclei counted", "Number of reconstructed nuclei passing the quality "
                               "checks in that specimen.", "Absolute"],
            ["Nuclei per 1,000 um2", "Count divided by the area of the region you "
                                     "drew.", "Comparison"],
            ["Nuclei per 100,000 um3", "Count divided by the volume the stack "
                                       "sampled.", "Comparison"],
        ],
    )
    body(
        document,
        "The count is a true count of what was imaged. The densities divide it by "
        "how much tissue was in the field, which makes them comparable across "
        "specimens only when the region was drawn on the same basis and the stacks "
        "cover a similar depth. Keep those consistent and the densities are the "
        "more informative of the two."
    )

    heading(document, "7.4 Quality flags", level=2)
    table(
        document,
        ["Flag", "What it means", "Use"],
        [
            ["Morphology note", "The nucleus is unusually long, short, wide or "
                                "bent. It is still counted and measured.",
             "Diagnostic"],
            ["Suspected clump", "The object looks like more than one nucleus.",
             "Diagnostic"],
            ["Width withheld", "No clean brightness profile was available, usually "
                               "because of a close neighbour.", "Diagnostic"],
            ["Network confidence", "How strongly the network called this object.",
             "Diagnostic"],
            ["Link rejected", "A candidate join was considered and refused, with "
                              "the reason recorded.", "Diagnostic"],
            ["Profile brightness", "Total signal across the nucleus. Depends on "
                                   "staining as well as on the nucleus.",
             "Diagnostic"],
        ],
    )
    body(
        document,
        "Flags are never used to silently delete an object. They travel with it in "
        "the output so you can decide what to do, and the technical-valid table "
        "already applies the standard choices."
    )
    callout(
        document,
        "Rule of thumb: lengths, angles, ratios and counts can be quoted. Anything "
        "involving width, depth or density is for comparing groups that were "
        "imaged the same way.",
    )

    # ------------------------------------------------------------------ 8
    heading(document, "8. Getting the Most From Your Microscope")
    body(
        document,
        "Saturn reads the calibration directly from the Leica metadata, so it "
        "adapts automatically to whatever settings you use. A few acquisition "
        "choices make a large difference to how much it can extract."
    )
    figure(document, NEW_FIGS, "v5_fig03_sampling_limit.png",
           "Figure 8.1. The spacing between slices is already well matched to the "
           "microscope. Increasing the zoom is what would add the most detail.")
    bullet(document, "The slice spacing of 0.346 microns is already well matched "
                     "to the objective. Leave it as it is.")
    bullet(document, "Raising the zoom by about three times would put roughly "
                     "five pixels across each nucleus instead of fewer than "
                     "two, which is what would turn width into an absolute "
                     "measurement rather than a comparative one.")
    bullet(document, "Recording at 12 bits rather than 8 costs nothing in time and "
                     "gives a finer brightness profile.")
    bullet(document, "Line averaging of two to four improves the profile at higher "
                     "zoom, and bleaches less than frame averaging.")
    bullet(document, "Keep laser power, gain, pinhole and zoom identical across "
                     "every specimen in a study, so the groups stay comparable.")

    # ------------------------------------------------------------------ 9
    heading(document, "9. What You Get Out")
    table(
        document,
        ["File", "What it is", "Use it for"],
        [
            ["The PDF report", "Summary of one specimen or one comparison",
             "Start here."],
            ["track_summary_technical_valid", "One row per reconstructed nucleus",
             "The main table for analysis."],
            ["specimen_summary", "One row per specimen",
             "Group comparisons. This is the unit of analysis."],
            ["spermatid_measurements", "One row per slice-level detection",
             "Checking detection. Not a count of nuclei."],
            ["metric_interpretation_limits", "How each measure should be read",
             "Worth a look before quoting a number."],
            ["technical_qc folder", "Diagnostics and annotations",
             "Troubleshooting."],
        ],
    )
    body(
        document,
        "Every width value carries a short note on how to read it, in the tables, "
        "on the axis labels and in the report text, so the number stays "
        "interpretable when it is copied somewhere else. Section 7 lists which "
        "measures can be quoted directly and which are for comparing groups."
    )

    # ------------------------------------------------------------------ 10
    heading(document, "10. Comparing Groups")
    body(
        document,
        "Each specimen is processed independently and contributes one row to the "
        "comparison. A study has one reference group and as many comparison groups "
        "as you need, so wild type against a single mutant, or against a mutant and "
        "a rescue line together, are both supported. Which group is the reference "
        "comes from the study manifest rather than from the group's name, so the "
        "direction of every comparison is explicit."
    )
    body(
        document,
        "Because several measures are tested at once, p-values are corrected for "
        "multiple testing. Two corrected values are reported: one accounting for "
        "testing several measures within a comparison, and one accounting for "
        "testing a single measure against several groups. The second becomes "
        "relevant as soon as a study includes more than one mutant line."
    )
    callout(
        document,
        "Inferential statistics need at least three specimens per group. Below "
        "that the report is descriptive and says so.",
    )

    # ------------------------------------------------------------------ 11
    heading(document, "11. Reproducibility")
    body(
        document,
        "Every run records what produced it: the image files, the region, the "
        "calibration read from the microscope metadata, the software version, the "
        "network checkpoint and its fingerprint, and the settings used. Archiving "
        "these with the study means a result can be reproduced exactly, which is "
        "what makes it defensible later."
    )

    # ------------------------------------------------------------------ 12
    heading(document, "12. How to Read the Results")
    bullet(document, "Counts, lengths, angles and curvature are direct "
                     "measurements and can be reported as they stand. Section 7 "
                     "gives the full list.")
    bullet(document, "Width is best used to compare groups, since these nuclei "
                     "sit close to the microscope's resolution limit.")
    bullet(document, "Counts are about ten percent higher than earlier versions "
                     "because touching nuclei are now separated, so do not mix old "
                     "and new counts in one comparison.")
    bullet(document, "The specimen is the replicate. Report how many animals were "
                     "used, not how many nuclei.")
    bullet(document, "A new microscope, magnification, stain or genotype is worth "
                     "a visual check of the overlays before the results are relied on.")

    # ------------------------------------------------------------------ 13
    heading(document, "13. Running a New Dataset")
    numbered(document, "Keep one folder per specimen, with the microscope metadata alongside.")
    numbered(document, "Use identical acquisition settings for every specimen.")
    numbered(document, "Draw one region per specimen and check it on an early, a "
                       "middle and a late slice.")
    numbered(document, "Confirm the pixel size and slice spacing were read correctly.")
    numbered(document, "Run one specimen first and look at the overlays before "
                       "committing to the whole cohort.")
    numbered(document, "Run the study manager so each specimen gets its own output.")
    numbered(document, "Read the quality-control summary before comparing groups.")
    numbered(document, "Use the reconstructed nucleus count and the specimen-level "
                       "table for analysis.")
    numbered(document, "Archive the manifest, settings, checkpoint identifier and "
                       "software version with the study.")

    # ------------------------------------------------------------------ appendix
    heading(document, "Appendix. Terms Used Here")
    table(
        document,
        ["Term", "What it means"],
        [
            ["Region of interest", "The area you draw. Everything outside is ignored."],
            ["Slice", "One optical plane. A nucleus spans several."],
            ["Detection", "One nucleus found on one slice."],
            ["Reconstructed nucleus", "Detections on consecutive slices joined into one object. This is what is counted."],
            ["Centre line", "The line down the middle of a nucleus, used for length."],
            ["Outline", "The boundary the network draws around a nucleus."],
            ["Width", "The width of the nucleus's fluorescence at half its peak."],
            ["Morphology note", "A flag that a nucleus is unusual. It is kept and measured."],
            ["Joining", "Deciding that detections on neighbouring slices are the same nucleus."],
            ["Representative slice", "The slice on which a nucleus appeared largest. Its length and width are taken from there."],
            ["Curvature", "Centre-line length divided by straight tip-to-tip distance. 1.00 is straight."],
        ],
    )

    body(
        document,
        f"Generated {date.today().isoformat()} from the Saturn v5.7.1 pipeline. "
        "Every figure and every number in this document comes from the KJ-01 and "
        "WT-01 specimens of the current study."
    )

    document.save(str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    written = build(Path(args.output))
    print(f"wrote {written}  ({written.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
