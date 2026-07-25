# Active Utilities

This directory contains only utilities required by Saturn v5.7:

- `tune_parameters_Saturnv5_7.py`: parameter tuning and self-check entry point.
- `saturn_unet25d_bridge.py`: lazy PyTorch loading and ROI-tiled U-Net
  probability inference.

Historical tuners, standalone audit scripts, v5.6 experimental modules, and
old build launchers are preserved in:

`archive/historical_source/saturn_v5_series/`

Keep active runtime modules at this level because the GUI and tuner use these
stable paths directly.
