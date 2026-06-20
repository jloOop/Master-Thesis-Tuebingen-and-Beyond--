# YBCO Josephson diode / ratchet: thesis code and analysis archive

This repository contains thesis-era analysis notebooks, measurement scripts,
and reproducibility material connected to my MSc project at the University of
Tübingen on YBa2Cu3O7 Josephson diode/ratchet devices fabricated by focused
helium-ion-beam irradiation.

The project contributed to:

Christoph Schmid*, Alireza Jozani*, Reinhold Kleiner, Dieter Koelle,
Edward Goldobin, "Josephson diode fabricated by focused-helium-ion-beam
irradiation," Physical Review Applied 24, 014041 (2025).
*Equal contribution.

## What is here

- chip and I-V characterization notebooks;
- deterministic sine-drive ratchet simulations from measured IVCs;
- quasistatic Gaussian/noise-drive rectification calculations;
- voltage/current acquisition and noise-characterization scripts;
- thesis PDF and review slides;
- preserved legacy scripts from the original thesis workflow.

## Important note on reproducibility

The analysis notebooks can be inspected and partly re-run from saved or
example data. The hardware-acquisition scripts require the original laboratory
setup, including NI-DAQ channels, VISA-connected waveform generator, current
source calibration, voltage amplifier gain, and cryogenic measurement wiring.

## Repository layout

- `notebooks/`: cleaned notebook entry points.
- `src/josephson_ratchet/`: reusable analysis code.
- `scripts/`: command-line reproduction/acquisition scripts.
- `configs/`: example hardware and analysis configurations.
- `docs/`: thesis/paper mapping and instrumentation notes.
- `legacy/`: preserved original scripts and notebooks.
- `data/`: small example or processed data only.

## AI-assistance note

Some scripting was AI-assisted during the thesis/research workflow. All
scientific interpretation, validation, device analysis, and conclusions were
reviewed by the author and collaborators.
