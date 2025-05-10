# Turbulent Flow Analysis using JHTDB

This repository contains Python scripts and reports developed as part of coursework on turbulent flows. The analysis leverages high-resolution data from the Johns Hopkins Turbulence Database (JHTDB) to investigate classical turbulence theory through spectral analysis, structure functions, flow topology, and Lagrangian particle tracking.

## 📁 Contents

The repository is organized around three major modules corresponding to assignments in the coursework:

### 1. Energy Spectrum & Structure Functions

* ✅ **Parseval's Theorem**: Verified for the $u$-velocity component.
* 📊 **1D & 2D Energy Spectra**: Computed using Fourier transforms; spectral slopes compared with Kolmogorov's prediction.
* 🔁 **Structure Functions**: Computed up to order 7 and analyzed for Extended Self-Similarity (ESS) and deviation from Kolmogorov's predictions.
* 📈 **Kolmogorov's 4/5 Law**: Empirically validated using third-order structure functions.

### 2. Turbulence Statistics & Coherent Structures

* 🔄 **Velocity Gradient Tensor**: Computed across 3D grid; eigenvalue PDFs plotted.
* 🔷 **Q-R Invariants**: Calculated to analyze flow topology using joint distributions and scatter plots.
* 📌 **Flow Classification**: Flow structures categorized (e.g., unstable focus, saddle points) based on the discriminant of the velocity gradient tensor.
* 🧮 **Continuity Check**: Verifies incompressibility using finite difference approximations.

### 3. Lagrangian Turbulence Analysis

* 🔀 **Particle Trajectories**: Simulated for various time spans and particle counts.
* 🔍 **Trapping Phenomenon**: Visualized particle entrapment in coherent structures.
* 📏 **Mean Square Displacement**: Analyzed to observe transition from ballistic to diffusive regimes.
* 🌪️ **Turbulent Diffusivity**: Estimated from the linear region of MSD.
* 🔂 **Richardson Dispersion**: Pair dispersion statistics evaluated and compared across initial separations.
* 🧮 **Lyapunov Exponent**: Computed from early exponential separation behavior.
