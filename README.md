# YBCO Josephson diode / ratchet: thesis code and analysis archive

This repository contains thesis-era analysis notebooks, measurement scripts,
experimental review material, and instrumentation documentation connected to my
MSc project at the University of Tübingen on YBa2Cu3O7 Josephson diode/ratchet
devices fabricated by focused helium-ion-beam irradiation.

The project contributed to:

Christoph Schmid<sup>*</sup>, Alireza Jozani<sup>*</sup>, Reinhold Kleiner,
Dieter Koelle, Edward Goldobin, "Josephson diode fabricated by
focused-helium-ion-beam irradiation," Physical Review Applied 24, 014041
(2025).

<sup>*</sup>Equal contribution.

## Start here

For experimental and industry readers, start with:

- [`docs/experimental-rd-highlights/`](docs/experimental-rd-highlights/)

This folder collects selected experimental review material from the Tübingen
Josephson-ratchet project, including device geometry, pre-fabrication context,
I-V and critical-current characterization, AC/noise rectification context, and
cryogenic measurement infrastructure.

## What is here

- chip and I-V characterization notebooks;
- deterministic sine-drive ratchet simulations from measured IVCs;
- quasistatic Gaussian/noise-drive rectification calculations;
- voltage/current acquisition and noise-characterization legacy scripts;
- experimental R&D review slides;
- future noise-operation preparation notes;
- cryogenic dipstick and instrumentation documentation;
- thesis PDF and preserved legacy scripts from the original workflow.

## Important note on reproducibility

The analysis notebooks can be inspected and partly re-run from saved or example
data. The hardware-acquisition scripts require the original laboratory setup,
including NI-DAQ channels, VISA-connected waveform generator, current-source
calibration, voltage-amplifier gain, and cryogenic measurement wiring.

## Repository layout

- `notebooks/`: cleaned notebook entry points for chip characterization,
  sine-drive rectification, Gaussian/noise-drive rectification, and noise tests.
- `docs/experimental-rd-highlights/`: selected experimental review material.
- `docs/future-noise-operation/`: setup-level noise-characterization material
  prepared for future direct noise-driven Josephson-ratchet operation.
- `docs/instrumentation/`: cryogenic measurement hardware and dipstick
  documentation.
- `docs/paper/`: publication context and citation information.
- `docs/thesis/`: MSc thesis PDF and thesis-related material.
- `legacy/`: preserved original scripts and notebooks from the thesis workflow.
- `src/josephson_ratchet/`: placeholder/package location for reusable analysis
  code.
- `figures/`: selected output figures or documentation figures.

## Scope

This is a historical and reproducibility-oriented archive, not a polished
instrument-control package. Some scripts are hardware-specific and are preserved
to document the original experimental workflow.
