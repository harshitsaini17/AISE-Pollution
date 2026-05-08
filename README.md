# PM2.5 Concentration Forecasting with Wavelet Neural Operators

## Overview
India’s air quality remains critically degraded; fine particulate matter (PM2.5, particles < 2.5 μm) is associated with approximately 1.67 million premature deaths annually. Traditional chemical transport models rely on computationally expensive differential equations, whereas data-driven models can learn spatiotemporal dynamics directly from simulations.

This repository presents a deep-learning framework to **forecast hourly PM2.5 concentrations** over a **140 × 124** spatial grid covering India, using WRF-Chem simulation data for 2016.

**Problem formulation**
- **Input:** 10 hours of meteorological, emission, and atmospheric features
- **Output:** Next 16 hours of PM2.5 predictions at every grid cell
- **Dataset:** WRF-Chem simulation data (2016)

---

## Model Architecture
The model is a **Wavelet Neural Operator (WNO)** that replaces Fourier spectral convolutions with **2D Haar Discrete Wavelet Transforms**, enabling multi-scale spatial reasoning.

```
Input → Conv1×1 Encoder → [WNO Block ×4] → GELU → Conv1×1 Decoder → + Residual → Output
```

Each **WNO Block** performs:
1. Group Normalization
2. Haar DWT (LL / LH / HL / HH sub-bands)
3. Spectral mixing (group-wise 3×3 conv across sub-bands)
4. Inverse Haar DWT
5. Spatial mixing (depthwise 5×5 conv)
6. Pointwise conv + GELU + residual

The decoder predicts a **delta** added to the last observed PM2.5 state, enabling the model to learn **changes** rather than absolute values.

**Training strategy**
- 3-model ensemble with seeds `[0, 42, 2026]`
- Stochastic Weight Averaging (SWA)
- Cosine annealing learning rate schedule
- Combined loss: MSE + spatial gradient penalty
- Horizon weighting for multi-step forecasts

---

## Repository Structure
```
AISE-Pollution/
├── configs/                        # YAML configuration files
│   ├── prepare_dataset.yaml        # Dataset preparation settings
│   ├── train.yaml                  # Training hyperparameters
│   └── infer.yaml                  # Inference settings
├── models/
│   └── baseline_model.py           # WNO model definition
├── scripts/
│   ├── prepare_dataset.py          # Stage 1 — preprocessing & normalization
│   ├── train.py                    # Stage 2 — multi-seed ensemble training
│   ├── infer.py                    # Stage 3 — ensemble inference
│   ├── data_exploration.py         # EDA utilities
│   ├── results_analysis.py         # Result evaluation
│   └── interpret_results.py        # Result interpretation & visualization
├── src/
│   └── utils/                      # Helper modules (optimizer, config loader, metrics)
├── notebooks/                      # Jupyter notebooks for experimentation
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Scripts Overview
The pipeline consists of three core stages followed by three analysis utilities.

### Main Pipeline
| Script | Description |
|---|---|
| `prepare_dataset.py` | Loads raw `.npy` files, computes derived features (wind speed, ventilation coefficient, rain mask), normalizes variables, and writes processed datasets. |
| `train.py` | Trains three WNO models with different seeds, applies SWA, and saves checkpoints. |
| `infer.py` | Runs ensemble inference, averages predictions, converts to physical units, clips negatives to zero, and saves `preds.npy`. |

### Analysis & Visualization
| Script | Description |
|---|---|
| `data_exploration.py` | Seasonal PM2.5 maps, emission hotspots, and time-series at critical grid cells. |
| `results_analysis.py` | Forecast RMSE by horizon, spatial error maps, and predicted vs. actual curves. |
| `interpret_results.py` | Statistical summaries, spatial heatmaps, distributions, hotspot/coldspot maps, and anomaly analysis. |

---

## Requirements
All dependencies are listed in `requirements.txt`.

| Package | Purpose |
|---|---|
| `torch>=2.0` | Deep learning framework (training, inference, GPU) |
| `triton` | Torch compiler backend |
| `numpy` | Array operations |
| `scipy` | Scientific computing |
| `pandas` | Data handling |
| `h5py` | HDF5 I/O |
| `netCDF4` | Reading WRF-Chem data |
| `xarray` | Labelled multi-dimensional arrays |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML utilities and metrics |
| `joblib` | Parallel utilities |
| `tqdm` | Progress bars |
| `PyYAML` | YAML config loading |

Install with:
```bash
pip install -r requirements.txt
```

---

## Reproducibility (Kaggle Notebook)
Open the Kaggle notebook and run all cells in order:
[**Group ID 46 — PRML Project**](https://www.kaggle.com/code/tahseen123/group-id-46-prml-project)

---

## Notes
- Full pipeline runtime: **~3–4 hours** on a Kaggle P100 GPU
- `/kaggle/working/` is persistent (up to 20 GB)
- `/kaggle/temp/` is non-persistent and suitable for intermediate artifacts

---

## Presentation
**Project slides:** https://canva.link/gz3uqqwwhhx3w1p
