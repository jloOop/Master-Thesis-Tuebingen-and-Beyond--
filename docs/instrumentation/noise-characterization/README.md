# External noise identification and noise-operation preparation

This folder documents setup-level noise checks for Josephson-ratchet noise
operation.

The slide deck `external_noise_identification_may2025.pdf` is a May 2025
presentation by Alireza Jozani. It connects the Josephson-ratchet/noise-operation
motivation to practical identification of external noise sources in the
measurement chain.

## Scientific context

The associated Physical Review Applied paper demonstrates deterministic
sine-drive rectification, loaded ratchet operation, and quasistatic Gaussian-noise
rectification based on measured IVCs. This folder documents the complementary
experimental task of identifying the laboratory noise contributions that would
matter for direct noise-driven operation.

## Covered setup components

- ADC baseline noise;
- voltage preamplifier and current-source contribution;
- waveform generator contribution;
- resistor thermal-noise estimates;
- shorted-dipstick and connected-dipstick configurations;
- suggested low-noise amplifier options.

## Relation to code

Relevant legacy scripts and notebooks include:

- `legacy/original_root_scripts/IV_Script_ForNoise.py`
- `legacy/original_root_scripts/understanding_random_signal_v9_legacy.py`
- `notebooks/04_noise_signal_tests.ipynb`

A small extracted summary table is stored in:

- `data/processed/noise-budget/noise_source_summary_from_slides.csv`

The extracted CSV is a presentation-level summary, not a replacement for raw
measurement traces.
