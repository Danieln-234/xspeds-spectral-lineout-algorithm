
**Author:** Anonymous
**Date:**  25/03/2025

#X-ray Single-Photon Energy-Dispersive Spectroscopy

See paper 'XSPEDS' for details and background
## Overview
This repository implements XSPEDS to get a spectral lineout of L shell emission of Germanium. This is in the X-ray regime.
It achieves this by combining single-photon-counting with Bragg spectroscopy mapping.

It also outputs the uncertainty bounds as well as the spectral resolution for the 1188 eV peak.


The main file is b803.py. It is a notebook that can can be run in order.
## Requirements

The data used is from sxro6416-r0504.h5

Libraries needed:
  - `NumPy`
  - `Pandas`
  - `SciPy`
  - `collections`
  - `pybaseines`
  - `matplotlib`

