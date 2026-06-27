# YBCO Josephson Diode / Josephson Ratchet — Research Portfolio

This repository is a public research portfolio and archival workspace for my Tübingen MSc-era work on **YBa$_2$Cu$_3$O$_7$ (YBCO) Josephson diode / Josephson ratchet devices**, together with selected follow-up material on **cryogenic measurement chains**, **noise-source identification**, and **instrumentation documentation**.

It is intended to document superconducting-device analysis, Python/Jupyter data-analysis workflows, cryogenic I--V interpretation, rectification calculations, and laboratory instrumentation context. It is **not** a polished production software package, a complete instrument-control platform, or a claim of sole fabrication ownership.

---

## Scientific context

A Josephson diode / ratchet is a superconducting device with asymmetric critical currents in opposite current directions. In the Tübingen YBCO implementation, an in-line Josephson-junction geometry was used so that the device could rectify a zero-mean applied drive into a finite dc voltage. The associated publication reports a high critical-current asymmetry, deterministic ac-drive rectification, loaded operation, output-power extraction, thermodynamic-efficiency analysis, and quasistatic Gaussian/noise-drive calculations.

This repository contains supporting portfolio material around that work: notebooks, selected documentation, experimental review slides, instrument/readout notes, cryogenic-dipstick material, and preserved legacy scripts from the original project workflow.

**Main publication:**

> C. Schmid\*, A. Jozani\*, R. Kleiner, D. Koelle, and E. Goldobin,  
> **“YBa$_2$Cu$_3$O$_7$ Josephson diode fabricated by focused-helium-ion-beam irradiation,”**  
> *Physical Review Applied* **24**, 014041 (2025).  
> DOI: `10.1103/vqhx-16ss`  
> \* Equal contribution.

The paper should be cited for the final peer-reviewed scientific results. This repository is supporting research-portfolio material.

---

## Start here

| Purpose | Entry point | What to inspect |
|---|---|---|
| Publication context | [`docs/paper/`](docs/paper/) | Paper citation and relation of the repository to the published work. |
| MSc thesis document | [`docs/thesis/`](docs/thesis/) | Thesis-era project document. |
| Clean notebook entry points | [`notebooks/`](notebooks/) | Chip characterization, sine-drive rectification, Gaussian/noise-drive calculations, and noise-signal tests. |
| Experimental R&D overview | [`docs/experimental-rd-highlights/`](docs/experimental-rd-highlights/) | Device geometry, I--V characterization, critical-current asymmetry, rectification context, and laboratory review material. |
| Measurement-chain and cryogenic instrumentation | [`docs/instrumentation/`](docs/instrumentation/) | Noise-chain documentation, cryogenic dipstick material, PCB/mechanical documentation, and setup notes. |
| Future noise-operation preparation | [`docs/future-noise-operation/`](docs/future-noise-operation/) | Setup-level external-noise characterization prepared for later direct noise-operation studies. |
| Historical scripts and notebooks | [`legacy/`](legacy/) | Preserved original scripts/notebooks from the research workflow. |

---

## Repository structure

```text
.
├── docs/
│   ├── experimental-rd-highlights/
│   │   └── selected experimental review slides and README
│   ├── future-noise-operation/
│   │   └── setup-level noise-characterization material and summary CSV
│   ├── instrumentation/
│   │   ├── dc-dipstick/
│   │   └── noise-characterization/
│   ├── paper/
│   │   └── publication citation and project connection
│   └── thesis/
│       └── MSc thesis PDF
├── notebooks/
│   ├── 01_chip_characterization.ipynb
│   ├── 02_sine_drive_rectification.ipynb
│   ├── 03_gaussian_drive_rectification.ipynb
│   └── 04_noise_signal_tests.ipynb
├── src/
│   └── josephson_ratchet/
│       └── instruments/
├── legacy/
│   ├── original_notebooks/
│   └── original_root_scripts/
├── figures/
└── README.md
```

Some folders contain historical or hardware-dependent material. The repository is therefore best read as a **research archive and portfolio**, not as a standalone software package that can reproduce every laboratory measurement without the original instruments, wiring, and data files.

---

## Representative workflows

### 1. Cryogenic I--V and critical-current analysis

The device-characterization workflow starts from low-temperature current--voltage characteristics and critical-current extraction. These data are used to identify the diode working point, quantify non-reciprocity through asymmetric positive and negative critical currents, and connect measured transport curves to ratchet behavior.

Relevant entry points:

- [`notebooks/01_chip_characterization.ipynb`](notebooks/01_chip_characterization.ipynb)
- [`docs/experimental-rd-highlights/`](docs/experimental-rd-highlights/)
- [`docs/paper/`](docs/paper/)

### 2. Deterministic ac-drive rectification

For quasistatic sinusoidal driving, a measured asymmetric I--V curve can be used to compute the time-averaged rectified voltage under an applied ac current. This connects the superconducting-device transport data to rectification curves, output-power estimates, and efficiency analysis.

Relevant entry point:

- [`notebooks/02_sine_drive_rectification.ipynb`](notebooks/02_sine_drive_rectification.ipynb)

### 3. Gaussian/noise-drive rectification calculations

The repository includes calculations for quasistatic Gaussian/noise-drive rectification based on measured or modeled I--V characteristics. These calculations are useful for understanding the expected ratchet response to externally supplied stochastic drives.

This should not be confused with a completed final direct noise-driven Josephson-ratchet experiment. The final direct follow-up noise-operation experiment is **not claimed as completed** in this repository.

Relevant entry point:

- [`notebooks/03_gaussian_drive_rectification.ipynb`](notebooks/03_gaussian_drive_rectification.ipynb)

### 4. Noise-source identification and readout-chain preparation

The follow-up measurement-chain work is organized around isolating external noise contributions before making device-level claims. The relevant configurations include ADC/PC baseline checks, voltage-amplifier and current-source configurations, waveform-generator contribution, resistive loads, shorted dipstick measurements, and connected-dipstick tests. FFT/PSD spectra and gain-bandwidth reasoning are used to identify readout limitations and possible improvement paths.

Relevant entry points:

- [`notebooks/04_noise_signal_tests.ipynb`](notebooks/04_noise_signal_tests.ipynb)
- [`docs/future-noise-operation/`](docs/future-noise-operation/)
- [`docs/instrumentation/noise-characterization/`](docs/instrumentation/noise-characterization/)

### 5. Cryogenic dipstick and instrumentation documentation

The instrumentation material documents DC cryogenic dipstick design and measurement-infrastructure context, including CAD/mechanical documentation, PCB-compatible mounting/readout configurations, wiring/assembly context, and laboratory setup notes.

Relevant entry points:

- [`docs/instrumentation/`](docs/instrumentation/)
- [`docs/instrumentation/dc-dipstick/`](docs/instrumentation/dc-dipstick/)

---

## Technologies and methods represented

- **Superconducting-device physics:** YBCO Josephson junctions, Josephson diode behavior, Josephson ratchets, critical-current asymmetry, rectification, loaded operation, output power, thermodynamic efficiency.
- **Cryogenic/device characterization:** low-temperature I--V analysis, magnetic-field-dependent critical currents, device-performance extraction.
- **Python/Jupyter analysis:** I--V post-processing, rectification calculations, Gaussian/noise-drive calculations, plotting, and data inspection.
- **Measurement-chain analysis:** DAQ/ADC context, waveform generator, current source, voltage preamplifier, resistive loads, FFT/PSD spectra, gain-bandwidth reasoning, noise-source isolation.
- **Instrumentation documentation:** cryogenic dipstick design, PCB-compatible device mounting/readout configuration, CAD/mechanical documentation, wiring and assembly context.

---

## What I personally use this repository to demonstrate

This repository supports the following evidence-backed claims:

- superconducting-device R&D context around YBCO Josephson diode / Josephson ratchet devices;
- cryogenic I--V and critical-current asymmetry analysis;
- deterministic ac-drive rectification calculations from measured device characteristics;
- quasistatic Gaussian/noise-drive rectification calculations;
- output-power and thermodynamic-efficiency interpretation;
- setup-level external-noise identification for planned follow-up noise-operation studies;
- measurement-chain reasoning involving ADC, generator, current source, voltage preamplifier, resistors, wiring, and cryogenic dipstick configurations;
- cryogenic instrumentation documentation and dipstick-related hardware support;
- public technical communication through notebooks, documentation, preserved scripts, and project summaries.

The strongest supported public claims are **device-analysis**, **physics-based data analysis**, **measurement-chain preparation**, and **instrumentation documentation**. Fabrication and He-FIB should be described as collaborative project context unless a specific personal step is separately documented.

---

## Safe-claim boundaries

This repository **does not** claim:

- sole fabrication ownership;
- sole He-FIB operation;
- full cleanroom process ownership;
- industrial quantum-device production ownership;
- custom low-noise amplifier design;
- a complete production-ready instrument-control package;
- production software engineering;
- that all notebooks/scripts run without the original lab environment;
- that the final direct noise-driven Josephson-ratchet experiment was completed.


---

## Reproducibility and data policy

This repository is partially reproducibility-oriented but not a fully self-contained reproduction package for every experiment.

- The notebooks are intended to make the analysis logic and scientific workflow inspectable.
- Some notebooks may require saved/example data files or paths from the original project environment.
- Hardware-acquisition scripts and legacy instrument-control material may require the original laboratory instruments and wiring configuration.
- Large or private raw measurement files may be absent.
- The peer-reviewed paper remains the authoritative source for final reported device figures of merit.

Before using any result from this repository in an application, presentation, or derivative work, check the corresponding notebook, folder README, and publication context.

---

## Role-family relevance

This repository is most relevant for:

- quantum hardware / superconducting-device R&D;
- cryogenic device characterization;
- applied physics and instrumentation;
- measurement-chain and low-noise-readout preparation;
- physics-based data analysis;
- experimental R&D communication.

It can also support scientific-computing applications when the role values research notebooks, data analysis, and simulation from measured device characteristics. For generic software-engineering roles, this repository should be framed only as research-code / analysis-workflow evidence, not as production software.

---

## Citation

If this repository is useful for understanding the project context, cite the peer-reviewed publication for the final scientific result:

```bibtex
@article{SchmidJozani2025YBCOJosephsonDiode,
  title   = {YBa$_2$Cu$_3$O$_7$ Josephson diode fabricated by focused-helium-ion-beam irradiation},
  author  = {Schmid, Christoph and Jozani, Alireza and Kleiner, Reinhold and Koelle, Dieter and Goldobin, Edward},
  journal = {Physical Review Applied},
  volume  = {24},
  pages   = {014041},
  year    = {2025},
  doi     = {10.1103/vqhx-16ss}
}
```

---

## Contact

**Alireza Jozani**  
Physics PhD Candidate, University of Tübingen  
GitHub: [`github.com/jloOop`](https://github.com/jloOop)





<sup>&ast;</sup> Equal contribution.
