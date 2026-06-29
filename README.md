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









