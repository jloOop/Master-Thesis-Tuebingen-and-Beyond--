# Notebooks

| Notebook | Purpose | Runs without lab hardware? | Data included? | Expected output |
|---|---|---:|---:|---|
| 01_chip_characterization.ipynb | IVC/Ic characterization | Yes | needs manual check | Ic+/Ic−, A(Icoil) |
| 02_sine_drive_rectification.ipynb | Vdc(Iac) from measured IVC | Yes | Yes after adding CSV | Vdc curve |
| 03_gaussian_drive_rectification.ipynb | Gaussian/noise-drive calculation | Yes | needs manual check | Vdc/Pout/Pin/η vs σ |
| 04_noise_signal_tests.ipynb | Noise tests/readout prep | Partly | needs manual check | PSD/noise diagnostics |


This folder contains the main clean notebook entry points for inspecting the thesis-era YBCO Josephson-ratchet analysis workflow.

| Notebook | Purpose | Notes |
|---|---|---|
| `01_chip_characterization.ipynb` | Chip / IVC / critical-current characterization | Main entry point for cryogenic I-V and critical-current asymmetry analysis. |
| `02_sine_drive_rectification.ipynb` | Deterministic sine-drive rectification | Uses measured asymmetric IVCs to evaluate quasistatic rectification behavior. |
| `03_gaussian_drive_rectification.ipynb` | Quasistatic Gaussian/noise-drive rectification | Calculates stochastic-drive rectification, power, and efficiency from measured IVCs. |
| `04_noise_signal_tests.ipynb` | Noise-signal tests and preparation | Supporting exploratory notebook for follow-up signal/noise analysis. |

## Reproducibility note

These notebooks are intended primarily as public, inspectable analysis workflows. Some cells may depend on local paths, saved data files, or thesis-era data organization. If a notebook references original laboratory paths, treat it as historical/reproducibility-context material rather than a standalone software package.

## Interpretation

The notebooks support analysis and documentation claims. They should not be used to claim production software engineering, full automated measurement-control ownership, or completed final direct noise-driven Josephson-ratchet operation.
