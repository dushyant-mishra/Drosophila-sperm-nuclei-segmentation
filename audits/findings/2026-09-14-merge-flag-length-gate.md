# Finding: the merge flag is gated behind a length threshold that almost never fires

Date: 2026-09-14
Raised by: user observation on visual evidence panels, quantified by Claude
Status: open, not yet remediated
Affects: `PIPELINE-V571-PRODUCTION-001` (accepted), `POP-SHORTTRACK-001`,
and any count-based biological conclusion

## What was observed

Reviewing the intensity-width visual evidence, the user pointed out that in
`kj_sv_40xx0.75-1_widest_mask`, `w1118_sv_feb_40xx0.75-1_widest_mask` and
`w1118_sv_feb_40xx0.75-1_narrowest_signal`, several objects are enclosed in one
mask and counted as a single nucleus.

Two of those are confirmed joined structures on objective topology:

| instance | length | branch nodes | official merge flag |
|---|---:|---:|---|
| `kj_sv_40xx0.75-1` 222 | 18.91 um | 19 | **False** |
| `w1118_sv_feb_40xx0.75-1` 184 | 6.24 um | 8 | **False** |
| `w1118_sv_feb_40xx0.75-1` 210 | 7.53 um | 0 | False |

Instance 210 is topologically clean, with two endpoints, no branch nodes, and a
length at the population median. It cannot be confirmed as a merge from the data;
the second object visible in that crop lies outside the mask boundary and appears
to be a correctly excluded neighbour. The finding rests on 222 and 184.

## Root cause

`sperm_segmentation_saturnv5.7.1.py`, instance measurement:

```python
"suspected_multi_object_merge": bool(
    geodesic_um > 20.0 and raw_branch_count > 0
),
```

A structure is flagged as a merge only when it is **both** longer than 20 um
**and** branched. Branching is the objective evidence; length is the weak signal.
The conjunction means objective merge evidence is discarded whenever the object
is shorter than 20 um.

## Scale, measured on plane 35 of one specimen per group, 559 instances

| | count | share |
|---|---:|---:|
| branched, any degree | 41 | 7.33% |
| longer than 20 um | 3 | 0.54% |
| **both, so officially flagged** | **1** | **0.18%** |
| branched but under 20 um, counted as one nucleus | 40 | 7.16% |

Median instance length is 7.91 um, so a 20 um gate is roughly 2.5 times the
typical nucleus and almost never opens. Of the four instances with eight or more
branch nodes, which cannot plausibly be a skeletonisation spur, three are missed.

Branch-node distribution: 92.7% unbranched, 1.4% with one or two branch nodes
where a spur is plausible, 5.2% with three to seven, 0.7% with eight or more.

The group rates are close: 7.19% in KJ against 7.12% in WT. A roughly balanced
bias limits the distortion of a between-group count comparison, but it does not
make the absolute count correct, and the balance is measured on one plane of one
specimen per group and must not be assumed to hold across the cohort.

## Why the profile merge detector does not close the gap

`intensity_profile_suspected_merge` catches 18 of the 41 branched instances. It
tests whether a cross-section is bimodal, so it sees nuclei lying side by side
across the normal and cannot see two joined end to end, which are unimodal at
every cross-section. It is complementary to the branch-node evidence, not a
replacement for it.

## Tension with the design ledger

`audits/V5_7_1_DESIGN_DECISIONS.md` states that length above 20 um is not
sufficient evidence to split an object and that objective fusion or merge
evidence is required before a technical intervention. Branching is exactly that
objective evidence. Requiring it to co-occur with a length threshold inverts the
intent: the weak signal has become mandatory and the strong one is ignored
without it.

## What has deliberately not been changed

Nothing. Merge handling determines `estimated_unique_nuclei`, which is a primary
biological metric, so changing it changes counts. That is a decision for the
owner and requires an independent audit, and the accepted pipeline claim may need
a superseding run rather than a silent amendment.

## Options for remediation

1. Flag on objective branch evidence regardless of length, keeping the 20 um
   review band as a separate annotation. Most consistent with the ledger.
2. Flag on branch evidence above a spur-tolerant threshold, for example three or
   more branch nodes, so a single skeletonisation spur does not reclassify a
   valid nucleus.
3. Combine branch evidence with the profile detector, since they catch different
   merge topologies, and treat either as sufficient.

A flag alone does not correct the count. Whether flagged merges are excluded from
`estimated_unique_nuclei`, split into their constituent objects, or reported as a
separate population is a further decision.
