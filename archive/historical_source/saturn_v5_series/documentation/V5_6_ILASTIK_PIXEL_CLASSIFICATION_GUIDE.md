# Saturn v5.6 Ilastik Pixel Classification Guide

This guide prepares a manual ilastik Pixel Classification pilot for sperm nucleus candidate support. The GUI is required only for initial labeling and classifier correction. A saved `.ilp` project can later run headlessly.

## Manual Steps

1. Open ilastik.
2. Create a Pixel Classification project.
3. Add the six training images as separate 2D datasets from `scratch\v5_6_ilastik_pilot\training_images`.
4. Select intensity, edge, Hessian, and structure-tensor features at small and medium scales.
5. Create the four classes in this exact order:
   1. `sperm_nucleus`
   2. `structured_tissue_edge`
   3. `punctum_or_ring`
   4. `diffuse_background`
6. Draw sparse labels for bright nuclei, faint nuclei, curved nuclei, dense parallel nuclei, structured tissue boundaries, transverse broad structures, bright puncta and rings, and diffuse granular background.
7. Enable Live Update.
8. Correct false classifications with small targeted strokes.
9. Check every training image.
10. Save the project as `sperm_nucleus_classifier_v1.ilp`.
11. Apply the trained project to the six evaluation images in `scratch\v5_6_ilastik_pilot\evaluation_images`.
12. Export class probabilities, not Simple Segmentation.

## Important

Probability channel `0` is treated as `sperm_nucleus` only if the class order above is preserved. The Saturn importer must not guess channel order. If metadata are missing, provide an explicit nucleus channel.

Do not use Saturn detections as ground truth labels. Raw images remain the source for morphology measurements.
