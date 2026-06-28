# YBCO Josephson diode / Josephson-ratchet portfolio

Public portfolio archive for superconducting-device R&D material connected to the Tübingen YBCO Josephson diode / Josephson-ratchet project. The repository documents thesis-era notebooks, selected analysis workflows, experimental review material, cryogenic/instrumentation notes, and follow-up measurement-chain preparation.

This is a research and documentation archive, not a polished instrument-control package or production software library.

## Start here

| Entry point | What it shows | Best for |
|---|---|---|
| [`notebooks/01_chip_characterization.ipynb`](notebooks/01_chip_characterization.ipynb) | Chip / IVC / device-characterization workflow | Superconducting-device analysis |
| [`notebooks/02_sine_drive_rectification.ipynb`](notebooks/02_sine_drive_rectification.ipynb) | Deterministic sine-drive rectification from measured IVCs | Device-performance analysis |
| [`notebooks/03_gaussian_drive_rectification.ipynb`](notebooks/03_gaussian_drive_rectification.ipynb) | Quasistatic Gaussian/noise-drive rectification calculations | Noise-drive modeling / data analysis |
| [`notebooks/04_noise_signal_tests.ipynb`](notebooks/04_noise_signal_tests.ipynb) | Noise-signal tests and measurement-chain context | Instrumentation / readout preparation |
| [`docs/experimental-rd-highlights/`](docs/experimental-rd-highlights/) | Selected experimental review slides and project context | Recruiter-readable project overview |
| [`docs/instrumentation/`](docs/instrumentation/) | Cryogenic dipstick and noise-chain documentation | Applied physics / instrumentation |
| [`docs/future-noise-operation/`](docs/future-noise-operation/) | Setup-level preparation for future direct noise-operation studies | Low-noise readout preparation |
| [`docs/paper/`](docs/paper/) | Publication connection and citation | Publication-backed evidence |
| [`docs/thesis/`](docs/thesis/) | MSc thesis PDF | Full scientific background |

## What this repository demonstrates

- Cryogenic IVC / I-V analysis for YBCO Josephson-ratchet devices.
- Magnetic-field / coil-current dependent critical-current extraction, including \(I_{c+}\), \(I_{c-}\), and critical-current asymmetry.
- Deterministic AC-drive rectification analysis: \(V_{dc}(I_{ac})\).
- Loaded-ratchet performance analysis: output power \(P_{out}\), input power \(P_{in}\), and thermodynamic efficiency \(\eta\).
- Quasistatic Gaussian/noise-drive rectification calculations from measured asymmetric IVCs.
- Measurement-chain reasoning around DAQ/ADC hardware, waveform generator, current source, voltage preamplifier, resistive loads, FFT/PSD spectra, and cryogenic dipstick configurations.
- Public documentation of superconducting-device R&D context, selected figures, notebooks, and thesis-era material.

## Scientific context

Josephson ratchets use asymmetric critical currents in a Josephson junction to rectify an applied zero-mean drive into a directed DC voltage. In this project, the device platform was a YBCO thin-film Josephson diode / ratchet based on in-line Josephson-junction geometry and focused He+ beam project context.

The published collaborative paper reports a YBCO Josephson diode operating as a high-efficiency ratchet with critical-current asymmetry around 7, rectified voltage up to 212 µV, output power up to 0.2 nW, thermodynamic efficiency up to about 75%, and an essential area around 1 µm².

## Related publication

Christoph Schmid*, Alireza Jozani*, Reinhold Kleiner, Dieter Koelle, Edward Goldobin, “Josephson diode fabricated by focused-helium-ion-beam irradiation,” *Physical Review Applied* **24**, 014041 (2025).  
DOI: `10.1103/vqhx-16ss`  
arXiv: `2408.01521`  
\* Equal contribution.

## Repository layout

```text
notebooks/
  01_chip_characterization.ipynb
  02_sine_drive_rectification.ipynb
  03_gaussian_drive_rectification.ipynb
  04_noise_signal_tests.ipynb

docs/
  experimental-rd-highlights/
  future-noise-operation/
  instrumentation/
  paper/
  thesis/

figures/
legacy/
src/josephson_ratchet/
```

## Reproducibility and hardware-dependence

This repository combines analysis notebooks, documentation, figures, thesis-era material, and preserved legacy scripts. Some notebooks can be inspected as analysis workflows; some scripts require original laboratory hardware, measurement files, or lab-specific configuration and are included as archival evidence rather than plug-and-play software.

## Contribution and scope boundary

This repository documents my thesis-era contribution to superconducting-device analysis, Josephson-ratchet performance interpretation, measurement-chain reasoning, and experimental R&D documentation.

It does **not** claim:

- sole fabrication ownership;
- sole He-FIB operation;
- full cleanroom process ownership;
- completed final direct noise-driven Josephson-ratchet operation;
- production instrument-control software;
- commercial or industrial device-production ownership.

For final published device results, cite the Physical Review Applied paper. For portfolio review, use this repository as supporting evidence of superconducting-device analysis, cryogenic I-V interpretation, measurement-chain reasoning, and physics-based Python/Jupyter workflows.





