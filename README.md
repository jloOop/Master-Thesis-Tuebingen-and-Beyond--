# Master Thesis Tübingen and Beyond — YBCO Josephson Ratchet Portfolio

Public superconducting-device R&D portfolio archive connected to MSc-era YBCO Josephson diode / Josephson-ratchet work at the University of Tübingen and later setup-level readout/instrumentation documentation.

This repository is meant as a compact inspection point for cryogenic I-V / IVC analysis, `Ic(H)` / `Ic(Icoil)` asymmetry extraction, AC-drive and quasistatic Gaussian/noise-drive rectification calculations from measured IVCs, loaded-ratchet `Pout` / `Pin` / `η` analysis, measurement-chain reasoning, and cryogenic dipstick documentation.

## Visual overview

| Device characterization | Rectification / stochastic-drive calculation | Readout and instrumentation |
|---|---|---|
| <img src="figures/01_ivc_ic_asymmetry_overview.png" width="360"> | <img src="figures/02_sine_drive_rectification.png" width="360"> | <img src="figures/04_noise_chain_fft_psd_summary.png" width="360"> |
| `Ic(Icoil)`, asymmetric IVC, `Vdc(Iac)` | measured IVC → rectification curve | ADC/noise-chain FFT/PSD diagnostic |
| <img src="figures/03_gaussian_noise_drive_calculation.png" width="360"> | <img src="figures/05_dc_dipstick_cad_render.png" width="360"> | See `docs/instrumentation/` |
| quasistatic Gaussian/noise-drive calculation | DC cryogenic dipstick CAD/rendering | setup-level documentation |

## Key outputs and symbols

- **Device:** YBCO Josephson diode / Josephson ratchet using in-line Josephson-junction geometry.
- **Core measured objects:** cryogenic IVC / I-V curves; `Ic+`, `Ic−`; `Ic(H)` / `Ic(Icoil)`; rectified `Vdc(Iac)`.
- **Publication-reported figures of merit:** critical-current asymmetry `A ≈ 7`, rectified voltage up to `212 µV`, output power up to `0.2 nW`, thermodynamic efficiency up to about `75%`, essential device area around `1 µm²`.
- **Power/efficiency quantities:** `Pout = Vdc · Idc`, `Pin = (1/T)∫V(t)I(t)dt`, `η = -Pout/Pin`.
- **Noise boundary:** Gaussian/noise-drive material here is quasistatic calculation and setup/readout preparation; it is not a claim that a final direct external-noise Josephson-ratchet experiment was completed.

## Start here

| Path | What to inspect | Why it matters |
|---|---|---|
| `figures/README.md` | visual index | fastest portfolio entry point |
| `notebooks/README.md` | notebook purpose and rerun status | separates clean analysis notebooks from legacy lab scripts |
| `notebooks/01_chip_characterization.ipynb` | IVC / chip characterization | cryogenic I-V and critical-current asymmetry workflow |
| `notebooks/02_sine_drive_rectification.ipynb` | deterministic sine-drive rectification | measured IVC → `Vdc(Iac)` |
| `notebooks/03_gaussian_drive_rectification.ipynb` | quasistatic Gaussian/noise-drive calculation | measured IVC → stochastic-drive `Vdc`, `Pout`, `Pin`, `η` |
| `notebooks/04_noise_signal_tests.ipynb` | exploratory noise-signal tests | supporting setup/noise-analysis context |
| `docs/paper/` | publication context | final scientific citation and DOI |
| `docs/thesis/` | MSc thesis | full thesis-era scientific context |
| `docs/experimental-rd-highlights/` | selected review slides | device geometry, IVC, rectification, fabrication context |
| `docs/instrumentation/` | noise-chain and dipstick documentation | measurement-chain / instrumentation evidence |
| `docs/future-noise-operation/` | setup-level noise-readout preparation | planned follow-up operation context, not final device-operation proof |
| `PROJECT_SCOPE.md` | claim boundaries | what this repo can and cannot support |
| `EVIDENCE_FOR_RECRUITERS.md` | recruiter/hiring-manager evidence map | safe public evidence summary |

## Publication connection

C. Schmid*, A. Jozani*, R. Kleiner, D. Koelle, and E. Goldobin, **“YBa₂Cu₃O₇ Josephson diode fabricated by focused-helium-ion-beam irradiation,”** *Physical Review Applied* **24**, 014041 (2025), DOI `10.1103/vqhx-16ss`.

`*` Equal contribution.

## Representative workflows

### 1. Cryogenic IVC and critical-current asymmetry

The chip-characterization workflow examines cryogenic I-V / IVC data, extracts positive and negative critical currents, and identifies magnetic-field / coil-current working points with strong critical-current asymmetry.

### 2. Deterministic AC-drive rectification

The sine-drive workflow uses measured asymmetric IVCs to calculate rectified DC voltage curves under quasistatic sinusoidal drive and loaded-ratchet operation.

### 3. Gaussian/noise-drive calculations

The Gaussian/noise-drive workflow evaluates quasistatic rectification, output power, input power, and efficiency from measured IVCs under stochastic-drive assumptions.

### 4. Measurement-chain preparation

The instrumentation and future-noise-operation folders document setup-level noise-source identification, ADC / generator / current-source / voltage-amplifier configurations, resistor and dipstick noise diagnostics, and cryogenic measurement hardware context.

## Repository map

```text
.
├── README.md
├── PROJECT_SCOPE.md
├── EVIDENCE_FOR_RECRUITERS.md
├── requirements.txt
├── figures/
│   ├── README.md
│   ├── 01_ivc_ic_asymmetry_overview.png
│   ├── 02_sine_drive_rectification.png
│   ├── 03_gaussian_noise_drive_calculation.png
│   ├── 04_noise_chain_fft_psd_summary.png
│   └── 05_dc_dipstick_cad_render.png
├── notebooks/
│   ├── README.md
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
├── legacy/
└── src/josephson_ratchet/
```

## Minimal local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The notebooks are public inspection and research-analysis material. Some notebook cells may still depend on thesis-era data paths or local saved data. Legacy acquisition scripts may require original lab hardware and are not a maintained instrument-control package.

## Scope and contribution boundaries

This repository supports claims about superconducting-device analysis, cryogenic I-V / IVC interpretation, critical-current asymmetry, rectification calculations, loaded-ratchet output/input power, thermodynamic efficiency, measurement-chain reasoning, and cryogenic/instrumentation documentation.

It should **not** be used to claim sole fabrication ownership, sole He-FIB operation, full cleanroom process ownership, completed final direct noise-driven Josephson-ratchet operation, production software engineering, commercial instrument-control software, custom low-noise amplifier design, full RF/microwave engineering ownership, certified electronics design, full cryostat design, or industrial quantum-device production ownership.












# Master Thesis Tübingen and Beyond — YBCO Josephson Ratchet Portfolio

This repository is a public superconducting-device R&D portfolio archive connected to my MSc-era YBCO Josephson diode/ratchet work at the University of Tübingen. It documents cryogenic I–V/IVC analysis, Ic(H)/Ic(Icoil) asymmetry extraction, AC-drive and quasistatic Gaussian-drive rectification calculations from measured IVCs, loaded-ratchet Pout/Pin/η analysis, and later setup-level readout/noise-chain identification and cryogenic dipstick documentation.

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

The instrumentation and future-noise-operation folders document setup-level noise-source identification, ADC / generator / current-source / voltage-amplifier configurations, resistor and dipstick noise diagnostics, and cryogenic measurement hardware context.

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









