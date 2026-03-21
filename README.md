# pyvital

Open source Python library for biosignal analysis. Provides signal processing utilities and medical algorithm implementations for vital signs data.

## Installation

```bash
pip install pyvital
```

## Core Functions

The `pyvital` package provides signal processing utilities directly at the package level:

```python
import pyvital

# Interpolate NaN values
data = pyvital.interp_undefined(raw_data)

# QRS detection (Pan-Tompkins algorithm)
r_peaks = pyvital.detect_qrs(ecg_data, srate=500)

# Blood pressure / pleth peak detection
minlist, maxlist = pyvital.detect_peaks(abp_data, srate=100)

# Bandpass filter
filtered = pyvital.band_pass(data, srate=500, fl=5, fh=15)

# Resampling
resampled = pyvital.resample_hz(data, srate_from=500, srate_to=100)
```

## Filters

Each filter module implements a `run(inp, opt, cfg)` function and a `cfg` dict describing its inputs, outputs, and parameters.

| Module | Description |
|--------|-------------|
| `abp_hpi` | Hypotension Prediction Index from arterial blood pressure |
| `abp_ppv` | Pulse Pressure Variation from arterial blood pressure |
| `ecg_annotator` | ECG waveform annotation using wavelets |
| `ecg_beat_noise_detector` | Beat/noise classification using deep learning |
| `ecg_classifier` | ECG rhythm and beat classification |
| `ecg_hrv` | Heart Rate Variability analysis |
| `ecg_mtwa` | Microvolt T-Wave Alternans detection |
| `ecg_qrs_detector` | R-peak detection |
| `eeg_fft` | EEG frequency analysis (band powers, SEF, MF) |
| `nirs_cox` | Cerebral oximetry autoregulation index (COx) |
| `pkpd_3comp` | Pharmacokinetic 3-compartment model |
| `pleth_dpop` | Delta POP from plethysmography |
| `pleth_ptt` | Pulse Transit Time |
| `pleth_pvi` | Pleth Variability Index |
| `pleth_spi` | Surgical Pleth Index |
| `resp_compliance` | Respiratory compliance |
| `sv_dlapco` | Stroke volume estimation (DLAPCO) |

## Filter Server

pyvital includes a built-in HTTP server (Sanic) that exposes filters as REST endpoints:

```bash
python -m pyvital [filter_folder] [port]
```

- `GET /` returns the list of available filters and their configurations.
- `POST /<module_name>` runs a filter with gzip-compressed JSON input.

## License

MIT
