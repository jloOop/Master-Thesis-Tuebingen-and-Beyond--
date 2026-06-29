





# Master Thesis Tübingen and Beyond — YBCO Josephson Ratchet Portfolio

This repository is a public superconducting-device R&D portfolio archive connected to my MSc research at the University of Tübingen and follow-up documentation around YBCO Josephson diode / Josephson-ratchet analysis.

It is intended as a compact entry point for inspecting the analysis, notebooks, figures, thesis material, publication context, and measurement-chain documentation associated with the project.

## Technical focus

This repository documents analysis and supporting material for:

- YBCO Josephson diode / Josephson-ratchet devices;
- cryogenic I-V / IVC interpretation;
- magnetic-field / coil-current dependent critical-current asymmetry;
- deterministic AC-drive rectification;
- quasistatic Gaussian/noise-drive rectification calculations;
- loaded-ratchet output power and thermodynamic efficiency;
- measurement-chain and noise-source reasoning;
- cryogenic instrumentation and dipstick documentation.

## Start here

| Path | What to inspect | Why it matters |
|---|---|---|
| `notebooks/01_chip_characterization.ipynb` | IVC / chip characterization and critical-current analysis | Core device-characterization workflow |
| `notebooks/02_sine_drive_rectification.ipynb` | Deterministic sine-drive rectification from measured IVCs | Shows how measured transport curves are converted into rectification curves |
| `notebooks/03_gaussian_drive_rectification.ipynb` | Quasistatic Gaussian/noise-drive calculations | Shows stochastic-drive analysis based on measured asymmetric IVCs |
| `notebooks/04_noise_signal_tests.ipynb` | Noise-signal tests / exploratory preparation | Supporting setup and signal-analysis context |
| `docs/paper/` | Publication connection and citation | Links the repository to the published Josephson-diode/ratchet work |
| `docs/thesis/` | MSc thesis PDF | Full scientific and experimental thesis context |
| `docs/experimental-rd-highlights/` | Selected experimental review slides | Compact visual summary of device geometry, I-V data, rectification, and fabrication context |
| `docs/instrumentation/` | Cryogenic measurement and dipstick documentation | Hardware-facing measurement-chain context |
| `docs/future-noise-operation/` | Setup-level noise-chain preparation | Follow-up preparation for controlled noise-operation studies |
| `legacy/` | Preserved original notebooks and scripts | Archival material from the original workflow |
| `src/josephson_ratchet/` | Reusable/helper code area | Lightweight source location for analysis utilities and instrumentation helpers |

## Scientific and experimental context

The project concerns YBCO Josephson diode / Josephson-ratchet devices based on asymmetric critical currents. In the ratchet regime, an unbiased AC or stochastic drive can be rectified into a DC voltage when the device is biased at an asymmetric working point.

The related publication reports YBCO Josephson diodes / ratchets with critical-current asymmetry around `A ≈ 7`, rectified voltage up to `212 µV`, output power up to `0.2 nW`, thermodynamic efficiency up to about `75%`, and an essential device area around `1 µm²`.

Related publication:

> C. Schmid*, A. Jozani*, R. Kleiner, D. Koelle, and E. Goldobin,  
> **“YBa₂Cu₃O₇ Josephson diode fabricated by focused-helium-ion-beam irradiation,”**  
> *Physical Review Applied* **24**, 014041 (2025).  
> DOI: `10.1103/vqhx-16ss`  
> `*` Equal contribution.

## Representative workflows

### 1. Cryogenic IVC and critical-current asymmetry

The chip-characterization workflow examines cryogenic I-V / IVC data, extracts positive and negative critical currents, and identifies magnetic-field / coil-current working points with strong critical-current asymmetry.

### 2. Deterministic AC-drive rectification

The sine-drive workflow uses measured asymmetric IVCs to calculate rectified DC voltage curves under quasistatic sinusoidal drive and loaded-ratchet operation.

### 3. Gaussian/noise-drive calculations

The Gaussian/noise-drive workflow evaluates quasistatic rectification, output power, input power, and efficiency from measured IVCs under stochastic-drive assumptions.

### 4. Measurement-chain preparation

The instrumentation and future-noise-operation folders document setup-level noise-source identification, ADC / generator / current-source / voltage-amplifier configurations, resistor and dipstick tests, and cryogenic measurement hardware context.

## Repository map

```text
.
├── README.md
├── PROJECT_SCOPE.md
├── notebooks/
│   ├── 01_chip_characterization.ipynb
│   ├── 02_sine_drive_rectification.ipynb
│   ├── 03_gaussian_drive_rectification.ipynb
│   └── 04_noise_signal_tests.ipynb
├── docs/
│   ├── paper/
│   ├── thesis/
│   ├── experimental-rd-highlights/
│   ├── instrumentation/
│   └── future-noise-operation/
├── figures/
├── legacy/
└── src/josephson_ratchet/
```

## Reproducibility status

This is a thesis-era and publication-adjacent research archive, not a polished commercial software package.

| Material | Status |
|---|---|
| Clean notebooks in `notebooks/` | Main inspection entry points; rerun status depends on local data paths and bundled data availability |
| Documents in `docs/` | Stable public context for thesis, publication, instrumentation, and follow-up setup preparation |
| `legacy/` scripts/notebooks | Preserved historical material; not intended as a clean API |
| Hardware/acquisition scripts | Lab-context material; may require original instruments, paths, and configuration |
| `src/josephson_ratchet/` | Reusable/helper-code area; inspect file-level docstrings before reuse |

## Scope and contribution boundaries

This repository supports claims about superconducting-device analysis, cryogenic I-V / IVC interpretation, critical-current asymmetry, rectification calculations, loaded-ratchet output/input power, thermodynamic efficiency, measurement-chain reasoning, and cryogenic/instrumentation documentation.

It should **not** be used to claim:

- sole fabrication ownership;
- sole He-FIB operation;
- full cleanroom process ownership;
- completed final direct noise-driven Josephson-ratchet device operation;
- production software engineering;
- commercial instrument-control software;
- custom low-noise amplifier design;
- industrial quantum-device production ownership.

The strongest safe framing is:

> Public superconducting-device R&D portfolio archive documenting YBCO Josephson diode/ratchet analysis, cryogenic I-V interpretation, rectification and efficiency calculations, measurement-chain reasoning, and cryogenic/instrumentation documentation.



