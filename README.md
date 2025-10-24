# XSPEDS: X‑ray Single‑Photon Energy‑Dispersive Spectroscopy

End‑to‑end pipeline for converting raw CCD frames (HDF5) into a counts‑per‑eV spectrum using:

1) Dynamic single‑photon cleaning & clustering,  
2) Geometric mapping from CCD pixels to energy via Bragg‑derived iso‑energy conics, and  
3) energy‑normalized lineouts with Poisson uncertainty bands.





## Quick Overview

- **Context:**  
  XSPEDS processes raw CCD camera frames collected from an X-ray diffraction setup,  
  where the X-ray source produces a characteristic spectrum with prominent germanium Lα (~1188 eV) and Lβ (~1218 eV) peaks.  
  Each frame is effectively a *snapshot* of that spectrum, showing small clusters of pixels created by individual X-ray photons hitting the detector.  
  Because of the detector’s tilt and the Bragg diffraction geometry, these peaks appear as **curved ridges** on the CCD. See *XSPEDS_project.pdf* for more background on the experimental setup and physical motivation. 

  <p align="center">
    <img src="figure/raw_frame_example.png" width="420">
    <br>
    <em>Example raw CCD frame (2048 by 2048) showing photon clusters. The curved Lα/Lβ ridges are visible.</em>
  </p>



- **Input:**  
  HDF5 stack of 2048×2048 CCD frames (Princeton FrameV2 layout).  
  The first three columns are dropped to remove edge artefacts.

- **Output:**  
  Energy spectrum (counts / eV) and an optional figure with Wiener-smoothed uncertainty bands.

- **Core idea:**  
  Use the two known germanium emission lines to fit the detector geometry,  
  generate iso-energy conics on the CCD, and sum photon hits along those conics normalized  
  by the local energy dispersion (eV per pixel).

---
### Quantitative results

- **Spectral performance (Ge Lα ≈ 1188 eV):**  
  FWHM ≈ 2.0 eV and SNR ≈ 16–22 after baseline correction  
  *(SNR defined relative to the pybaselines-corrected background)*  

- **Geometric calibration (Lα/Lβ ridge fit):**  
  Sub-pixel residuals (< 0.01 px) in fitted geometry parameters  

See *Section 3.4* of the project report (PDF) for detailed validation and figures.



**Project status:**  
The main algorithm and functionality were completed in April 2025 as part of my Oxford computational project.  
Subsequent commits involve only formatting, documentation, and readability improvements. The underlying logic and results remain unchanged from the original implementation.

---




**Requirements:**

`numpy`, `scipy`, `h5py`, `matplotlib`, `pybaselines`, `matplotlib`

All dependencies are also listed in `requirements.txt`; install via:
```bash
pip install -r requirements.txt







## Data expectations

- **File:** Example `sxro6416-r0504.h5`, with Princeton FrameV2 groups:
  ```
  Configure:0000/Run:0000/CalibCycle:{0000,0001,...}/
    Princeton::FrameV2/SxrEndstation.0:Princeton.0/data
  ```
- Each “CalibCycle” provides a 2048×2048 frame (the loader collects all cycles into a stack).
- On load, the first 3 columns are dropped (dataset‑specific spike/edge artefacts).

---



## What the pipeline does

### 1) Load
Reads frames from the HDF5 tree into a `(N, H, W)` stack (`float64`) and drops the first 3 columns. Frames are expected to be **2048×2048**. Update the HDF5 path in `load_image_data` if your dataset layout differs.

### 2) Cleaning & Clustering (SPC)
- **Per‑batch pedestal modeling (default 5 rows):** fit a Gaussian to the ADU histogram region dominated by background.
- **Adaptive threshold:** choose a dynamic threshold; values below it are zeroed (“scrubbed”).
- **BFS‑style clustering:** identify photon shapes (1–4 pixels typical) and collect per‑frame **photon maps** plus **cluster metadata**.

### 3) Mapping (instrument calibration)
- **Ridge extraction:** across row batches, sum columns and detect two reference ridges (e.g., Ge Lα & Lβ), smoothed with a Gaussian filter (`MAP_SMOOTH_SIGMA`).
- **Conic geometry:** photons of energy \(E\) form a cone (half‑angle derived from Bragg’s law). Intersection with a tilted CCD plane yields conics (ellipse/hyperbola); near the vertex they are well approximated by parabolas.
- **Parameter fit:** fit parabolas to both ridges and solve for distance **d** and tilt **θ_z** by minimizing (1) scatter residuals, (2) **focal‑length residual** via \(F=1/4A\), and (3) **vertex‑gap residual**. Uses global **differential evolution** with an optional seed, followed by local least‑squares.

### 4) Lineout (physics output)
- Generate **iso‑energy conics** using the fitted geometry and \(alpha(E)\).
- **Sum photon counts** within ±`TOLERANCE_PX` laterally around each conic.
- **Normalize to counts/eV:** divide by the local energy bin width \(W(E)=(2\,	ext{tol}+1)\,\mathrm{d}E/\mathrm{d}p\) because dispersion varies strongly across the band.
- **Uncertainties:** Poisson shot noise (σ≈√N). Optional **Wiener** smoothing for display and shaded ±`ERROR_BAND_K`·σ bands.

---

## Configuration reference
Note to self- mention where these configs affect

| Key | Meaning | Typical impact |
|---|---|---|
| `ROW_BATCH_SIZE` | Rows per pedestal/threshold fit | Larger → smoother thresholds; smaller → more local adaptation |
| `MAP_ALPHA1_DEG`, `MAP_ALPHA2_DEG` | Known half‑angles (90° − Bragg angle) | Encodes physics link to energy; ensure consistency with chosen lines |
| `E_STEP` | Energy grid step | Smaller → finer spectrum (slower). Paper uses 0.1 eV |
| `TOLERANCE_PX` | Lateral half‑width around each conic | Tune for best SNR without FWHM inflation (±2 px is a solid default) |

---

## Outputs

- **Console log:** geometry fit, counts summary, Lα FWHM and SNR, runtime.
- **Figure (optional):** spectrum with ±k·σ bands; save via `SAVE_FIG_PATH`.

---
