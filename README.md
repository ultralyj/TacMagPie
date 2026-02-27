# TacMagPie

**TacMagPie** is a general-purpose real-time simulator for soft magnetic tactile sensors (sMagTac), combining physically consistent elastomer deformation modeling with accurate magnetic flux computation.

> **Paper:** *TacMagPie: Fast, Physically Consistent Simulation of Soft Magnetic Tactile Sensors*

---

## Overview

Soft magnetic tactile sensors offer compelling advantages for robotic manipulation — rich 3D contact force measurement, high durability, and a compact form factor. Yet simulation has remained a bottleneck: finite element methods (FEM) are physically rigorous but too slow for real-time use, while analytical approximations fail to generalize across geometries and contact scenarios.

TacMagPie closes this gap with two core components:

- **MLS-MPM** (Moving Least Squares Material Point Method) for hyperelastic modeling of silicone elastomers
- **DMM** (Differential Magnetic Microparticle Method) for real-time resolution of magneto-elastic coupling

The result is a simulator that runs in **under 1 ms per step** (3–4 orders of magnitude faster than FEM), while maintaining high physical fidelity (**PSNR 26.21 dB, SSIM 0.68**) against real-world sensor measurements.

## Repository Structure
* tacmagpie: source code of the proposed simulator;
* model: STL files for simulation environment;
* hardware: hardware design and embedded code (stm32) for real-world sensor reference
* evaluation: implementation of mass-spring and FEM methods, comparative experiments and dataset.

## Requirements
```
conda create -n tacmagpie python=3.10
conda activate tacmagpie
pip install taichi numpy open3d mujoco
pip install glob glfw pillow matplotlib # for evaluation
```

## Quick Start

```bash
# Run with auto-generated spherical indenter
cd tacmagpie
python MagPie.py
```


## MuJoCo Co-simulation

TacMagPie supports real-time joint input from MuJoCo through `MuJoCoJointController`, enabling bidirectional coupling between robot simulation and tactile response:

```python
coming soon...
```


## Citation

```bibtex
@article{tacmagpie2026,
  title   = {TacMagPie: Fast, Physically Consistent Simulation of Soft Magnetic Tactile Sensors},
  year    = {2026}
}
```
