
# 🌌 Quantum Cosmology & Astrophysics Unified Suite (QCAUS)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quantum-cosmology-astrophysics-unified-suite.streamlit.app/)
[![License](https://img.shields.io/badge/License-Dual%20License-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A unified computational framework integrating 8 interconnected open-source projects that explore the quantum nature of the universe – from dark matter solitons to quantum-corrected cosmology, magnetar QED, and stealthdarkleakage detection. 

**[Launch QCAUS App](https://qcaustfordmodel.streamlit.app/)**

---

## 🔭 Projects Overview

### 1. QCI AstroEntangle Refiner
**FDM Soliton Physics + Photon-DarkPhoton Entanglement Overlay**

| Feature | Description |
|---------|-------------|
| **FDM Soliton** | Fuzzy Dark Matter soliton core: ρ(r) = ρ₀ [sin(kr)/(kr)]² |
| **PDP Entanglement** | Photon-DarkPhoton kinetic mixing: ℒ_mix = (ε/2) F_μν F'^μν |
| **Image Processing** | Upload FITS, JPEG, PNG images; apply quantum overlays |
| **Annotated Comparison** | Before/after views with scale bars and metrics |
| **Radar-Style Overlay** | Green speckles (FDM) + Blue halos (PDP) for stealth detection visualization |

### 2. Magnetar QED Explorer
**Strong-Field Quantum Electrodynamics in Magnetar Magnetospheres**

| Feature | Description |
|---------|-------------|
| **Dipole Field** | B = B₀ (R/r)³ (2 cosθ, sinθ) |
| **Vacuum Polarization** | Euler-Heisenberg effect: ΔL = (α/45π) (B/B_crit)² |
| **Dark Photon Conversion** | P_conversion = ε² (1 - e^{-B²/m²}) |
| **Interactive Visualization** | Real-time parameter adjustment for B-field, mixing angle |

### 3. Primordial Photon-DarkPhoton Entanglement
**Von Neumann Evolution in the Expanding Universe**

| Feature | Description |
|---------|-------------|
| **Von Neumann Equation** | i∂ρ/∂t = [H_eff, ρ] |
| **Entanglement Entropy** | S = -Tr(ρ log ρ) |
| **Mixing Probability** | |⟨ψ_d|ψ_γ⟩|² |
| **Time Evolution** | Simulates photon-dark photon oscillation in early universe |

### 4. QCIS – Quantum Cosmology Integration Suite
**Quantum-Corrected Cosmological Perturbations**

| Feature | Description |
|---------|-------------|
| **Quantum-Corrected Power Spectrum** | P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) |
| **Non-Gaussianity** | f_NL parameter for primordial fluctuations |
| **Spectral Index** | n_q for quantum corrections |
| **ΛCDM Comparison** | Side-by-side comparison with standard cosmology |

---

## 📊 Key Metrics & Outputs

| Project | Key Outputs | Download Format |
|---------|-------------|-----------------|
| **QCI AstroEntangle** | Annotated comparison, radar-style overlay, FDM soliton map, PDP entanglement map | PNG, JSON |
| **Magnetar QED** | B-field map, QED polarization, dark photon conversion | PNG, NPZ |
| **Primordial Entanglement** | Entropy evolution, mixing probability time series | PNG, JSON, NPY |
| **QCIS** | Power spectra, quantum enhancement ratio | PNG, JSON, NPY |

---

## 🧪 Supported Image Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **FITS** | .fits, .fit | Astronomical images (Hubble, JWST, Chandra) |
| **JPEG** | .jpg, .jpeg | Standard images |
| **PNG** | .png | Lossless images |
| **BMP** | .bmp | Bitmap images |

---

## 📥 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/tlcagford/QCI_AstroEntangle_Refiner.git
cd QCI_AstroEntangle_Refiner

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
astropy>=5.3.0  # For FITS support
Pillow>=10.0.0  # For image processing
```

---

## 📖 Physics References

### FDM Soliton
The Fuzzy Dark Matter soliton is the ground state solution of the Schrödinger-Poisson equation for ultra-light bosons (axions, ~10⁻²² eV):

$$\rho(r) = \rho_0 \left[\frac{\sin(kr)}{kr}\right]^2$$

### Photon-DarkPhoton Kinetic Mixing
The interaction between photons and dark photons is described by:

$$\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}$$

where ε is the kinetic mixing angle.

### Von Neumann Evolution
The density matrix evolution for entangled systems:

$$i\partial_t\rho = [H_{\text{eff}}, \rho]$$

### Quantum-Corrected Power Spectrum
The matter power spectrum with quantum corrections:

$$P(k) = P_{\Lambda\text{CDM}}(k) \times \left(1 + f_{\text{NL}}\left(\frac{k}{k_0}\right)^{n_q}\right)$$

---

## 🚀 Quick Start Examples

### QCI AstroEntangle Refiner
```python
# Load image and apply FDM + PDP overlays
from qci_astro import process_qci_astro
enhanced, soliton, pdp = process_qci_astro(image_data, omega=0.5, fringe=1.0, soliton_scale=1.0)
```

### Magnetar QED Explorer
```python
# Compute magnetar field and dark photon conversion
from magnetar_qed import process_magnetar
B_mag, qed, dark_photons = process_magnetar(r_grid, theta_grid, B0=1e15, mixing=0.1)
```

### Primordial Entanglement
```python
# Simulate photon-dark photon entanglement evolution
from primordial_entanglement import process_primordial_entanglement
entropy, mixing = process_primordial_entanglement(omega=0.7, dark_mass=1e-9, mixing=0.1)
```

### QCIS Power Spectra
```python
# Compute quantum-corrected power spectrum
from qcis import process_qcis
P_quantum = process_qcis(k_vals, f_nl=1.0, n_q=0.5)
```

---

## 📊 Example Outputs

### Annotated Comparison (Abell-1689)
- **Before**: Standard HST/JWST data with scale bar (100 kpc)
- **After**: FDM Soliton + PDP Entanglement overlays
- **Metrics**: Maximum Mixing Ratio, Minimum Entropy, FDM Value (kpc)

### Radar-Style Overlay
- **Green**: FDM Soliton (dark matter density)
- **Blue**: PDP Entanglement (dark photon field)
- **Red**: Original astrophysical signal

### Magnetar Field Maps
- **B-Field**: Dipole field structure (B ∝ r⁻³)
- **QED Polarization**: Euler-Heisenberg vacuum effects
- **Dark Photons**: Conversion probability maps

### Power Spectra
- **ΛCDM**: Standard cosmology baseline
- **Quantum**: Corrected spectrum with f_NL and n_q parameters

---

## 📄 Citation

If you use QCAUS in your research, please cite:

```bibtex
@software{Ford2026QCAUS,
  author = {Ford, Tony E.},
  title = {Quantum Cosmology \& Astrophysics Unified Suite (QCAUS)},
  year = {2026},
  url = {https://github.com/tlcagford/QCI_AstroEntangle_Refiner},
  doi = {10.5281/zenodo.xxxxxxx}
}
```

---

## 📜 License

This project is released under a **Dual License**:

- **Academic / Non-Commercial Use**: Free for research, education, and personal projects
- **Commercial Use**: Requires a separate license. Please contact the author for details.

See the `LICENSE` file for full terms.

---

## 📧 Contact

**Tony E. Ford**  
Independent Researcher / Astrophysics & Quantum Systems  
GitHub: [@tlcagford](https://github.com/tlcagford)  
Email: tlcagford@gmail.com

---

## 🙏 Acknowledgments

- **NASA/ESA Hubble Space Telescope & JWST** for public FITS data
- **OpenSky Network** for radar data integration
- **FDM, QED, and cosmology communities** for foundational research
- **Streamlit, NumPy, SciPy, Matplotlib, Astropy** for open-source tools

---

## 🔗 Related Projects

| Project | Repository |
|---------|------------|
| **StealthPDPRadar** | [StealthPDPRadar](https://github.com/tlcagford/StealthPDPRadar) |
| **Magnetar QED Explorer** | [Magnetar-Quantum-Vacuum-Engineering](https://github.com/tlcagford/Magnetar-Quantum-Vacuum-Engineering-for-Extreme-Astrophysical-Environments-) |
| **Primordial Entanglement** | [Primordial-Photon-DarkPhoton-Entanglement](https://github.com/tlcagford/Primordial-Photon-DarkPhoton-Entanglement) |
| **QCIS** | [Quantum-Cosmology-Integration-Suite](https://github.com/tlcagford/Quantum-Cosmology-Integration-Suite-QCIS-) |

---

*"Exploring the quantum nature of the universe – from dark matter solitons to quantum-corrected cosmology."*

---

## 📸 Screenshots

### QCI AstroEntangle Refiner
![Annotated Comparison](docs/images/comparison.png)
*Annotated before/after comparison of Abell-1689 with FDM Soliton and PDP Entanglement overlays*

### Magnetar QED Explorer
![Magnetar Fields](docs/images/magnetar.png)
*Magnetar dipole field with quantum vacuum polarization and dark photon conversion*

### Primordial Entanglement
![Entanglement Evolution](docs/images/entanglement.png)
*Von Neumann entropy evolution and photon-dark photon mixing probability*

### QCIS Power Spectra
![Power Spectrum](docs/images/power_spectrum.png)
*Quantum-corrected matter power spectrum vs ΛCDM*

---

## 🚀 Quick Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `app.py`
6. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`
```

---

## 📁 Additional Files to Create

### `requirements.txt`
```txt
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
astropy>=5.3.0
Pillow>=10.0.0
pandas>=2.0.0
```

### `LICENSE` (Dual License)
```txt
Dual License: Academic & Commercial

Academic / Non-Commercial Use:
- Free for research, education, and personal projects
- Attribution required

Commercial Use:
- Requires separate license
- Contact: tlcagford@gmail.com
```
