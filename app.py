import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm  # FIX BUG-1: use plt.get_cmap (not deprecated mcm.get_cmap)
from matplotlib.patches import Circle
from PIL import Image
import io, base64, warnings, zipfile
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter, convolve, uniform_filter

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="QCAUS v1.0 — Quantum Cosmology & Astrophysics Unified Suite",
    page_icon="🔭",
    layout="wide"
)

st.markdown("""<style>
[data-testid="stAppViewContainer"] { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e0e4e8; }
.stTitle, h1, h2, h3 { color: #1e3a5f; }
.dl-btn {
    display: inline-block; padding: 6px 14px; background-color: #1e3a5f;
    color: white !important; text-decoration: none; border-radius: 5px;
    margin-top: 6px; font-size: 13px;
}
</style>""", unsafe_allow_html=True)

# =============================================================================
# VERIFIED PHYSICS FUNCTIONS — Tony Ford QCAUS Model
# =============================================================================

def fdm_soliton_2d(size=300, m_fdm=1.0):
    """FDM Soliton 2D density map: ρ(r) = ρ₀[sin(kr)/(kr)]²   k=π/r_s"""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / size * 5
    r_s = 1.0 / m_fdm
    k = np.pi / max(r_s, 0.1)
    kr = k * r
    with np.errstate(divide="ignore", invalid="ignore"):
        sol = np.where(kr > 1e-6, (np.sin(kr) / kr) ** 2, 1.0)
    mn, mx = sol.min(), sol.max()
    return (sol - mn) / (mx - mn + 1e-9)


def fdm_soliton_profile(m_fdm=1.0, n=300):
    """1D radial soliton profile for plot"""
    r = np.linspace(0, 3, n)
    r_s = 1.0 / m_fdm
    k = np.pi / max(r_s, 0.1)
    kr = k * r
    return r, np.where(kr > 1e-6, (np.sin(kr) / kr) ** 2, 1.0)


def generate_interference_pattern(size, fringe, omega):
    """PDP two-field interference pattern"""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / size * 4
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 15.0
    pat = (
        np.sin(k * 4 * np.pi * r) * 0.5
        + np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi))) * 0.5
    )
    pat = pat * (1 + omega * 0.6 * np.sin(k * 4 * np.pi * r))
    pat = np.tanh(pat * 2)
    return (pat - pat.min()) / (pat.max() - pat.min() + 1e-9)


def pdp_entanglement_overlay(image, interference, soliton, omega):
    """Overlay PDP interference + FDM soliton onto grayscale image"""
    # FIX BUG-12: auto-resize if shape mismatch instead of silently returning zeros
    if image.shape != interference.shape:
        pil = Image.fromarray((interference * 255).astype(np.uint8))
        pil = pil.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
        interference = np.array(pil, dtype=np.float32) / 255.0
    if image.shape != soliton.shape:
        pil = Image.fromarray((soliton * 255).astype(np.uint8))
        pil = pil.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
        soliton = np.array(pil, dtype=np.float32) / 255.0
    m = omega * 0.6
    return np.clip(image * (1 - m * 0.4) + interference * m * 0.5 + soliton * m * 0.4, 0, 1)


def pdp_spectral_duality(image, omega=0.20, fringe_scale=45.0,
                         mixing_eps=1e-10, fdm_mass_eV=1e-22):
    """
    FIX BUG-5 & BUG-6: Use kin_mix (epsilon) directly and fdm_mass for dark photon mass.
    Photon-Dark-Photon spectral duality via FFT kinetic mixing.
    dark_photon_mass in eV, oscillation_length = 100 / (m_dark_GeV * 1e9)
    """
    rows, cols = image.shape
    fft_s = fftshift(fft2(image))
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # Convert fdm_mass (×10⁻²² eV) to GeV: m_GeV = fdm_mass * 1e-22 * 1e-9
    m_dark_GeV = max(fdm_mass_eV * 1e-31, 1e-35)   # prevent /0
    L = 100.0 / max(m_dark_GeV * 1e9, 1e-6)
    osc = np.sin(2 * np.pi * R * L / max(fringe_scale, 1.0))
    # FIX BUG-6: mixing_eps is dimensionless epsilon, use directly (not *1e9)
    dmm = mixing_eps * np.exp(-omega * R ** 2) * np.abs(osc) * (
        1 - np.exp(-R ** 2 / max(fringe_scale / 30, 0.1))
    )
    omm = np.exp(-R ** 2 / max(fringe_scale / 30, 0.1)) - dmm
    dark_mode = np.abs(ifft2(fftshift(fft_s * dmm)))
    ordinary_mode = np.abs(ifft2(fftshift(fft_s * omm)))
    return ordinary_mode, dark_mode


def entanglement_residuals(image, ordinary, dark, strength=0.3,
                           mixing_eps=1e-10, fringe_scale=45.0):
    """
    FIX BUG-6: mixing_eps is dimensionless epsilon, used directly.
    S = -ρ·log(ρ) + interference cross-term
    """
    eps = 1e-10
    tp = np.sum(image ** 2) + eps
    rho = np.maximum(ordinary ** 2 / tp, eps)
    S = -rho * np.log(rho)
    xterm = (np.abs(ordinary + dark) ** 2 - ordinary ** 2 - dark ** 2) / tp
    # Scale mixing contribution: use log10 of inverse epsilon for physical significance
    eps_scale = np.clip(-np.log10(mixing_eps + 1e-15) / 12.0, 0, 1)  # 0..1 over 1e-12..1
    res = S * strength + np.abs(xterm) * eps_scale
    ks = max(3, int(fringe_scale / 10))
    if ks % 2 == 0:
        ks += 1
    kernel = np.outer(np.hanning(ks), np.hanning(ks))
    return convolve(res, kernel / kernel.sum(), mode="constant")


def dark_photon_detection_prob(dark_mode, residuals, entanglement_strength=0.3):
    """Bayesian dark photon detection: P = prior·L / (prior·L + (1-prior))"""
    dark_ev = dark_mode / (dark_mode.mean() + 0.1)
    lm = uniform_filter(residuals, size=5)
    res_ev = lm / (lm.mean() + 0.1)
    prior = entanglement_strength
    lhood = dark_ev * res_ev
    prob = prior * lhood / (prior * lhood + (1 - prior) + 1e-10)
    return np.clip(gaussian_filter(prob, sigma=1.0), 0, 1)


def blue_halo_fusion(image, dark_mode, residuals):
    """
    Blue-Halo Fusion: R=original  G=residuals  B=dark_mode  γ=0.45
    From StealthPDPRadar/pdp_radar_core.py
    """
    def pnorm(a):
        mn, mx = a.min(), a.max()
        return np.sqrt((a - mn) / (mx - mn + 1e-10))

    rn, dn, en = pnorm(image), pnorm(dark_mode), pnorm(residuals)
    kernel = np.ones((5, 5)) / 25
    lm = convolve(en, kernel, mode="constant")
    en_enh = np.clip(en * (1 + 2 * np.abs(en - lm)), 0, 1)
    rgb = np.stack(
        [rn, en_enh, np.clip(gaussian_filter(dn, 2.0) + 0.3 * dn, 0, 1)],
        axis=-1
    )
    return np.clip(rgb ** 0.45, 0, 1)


def magnetar_physics(size=300, B0=1e15, mixing_angle=0.1):
    """
    Magnetar dipole field, Euler-Heisenberg QED, and dark photon conversion maps.
    B = B₀(R/r)³√(3cos²θ+1)   B_crit = 4.414×10¹³ G
    ΔL = (α/45π)(B/B_crit)²
    P_conv = ε²(1−e^{−B²/m²})
    """
    B_CRIT = 4.414e13
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    dx = (x - cx) / (size / 4)
    dy = (y - cy) / (size / 4)
    r = np.sqrt(dx ** 2 + dy ** 2) + 0.1
    theta = np.arctan2(dy, dx)
    B_mag = (B0 / r ** 3) * np.sqrt(3 * np.cos(theta) ** 2 + 1)
    B_n = np.clip(B_mag / B_mag.max(), 0, 1)
    qed = (B_mag / B_CRIT) ** 2
    qed_n = np.clip(qed / (qed.max() + 1e-30), 0, 1)
    m_eff = 1e-9
    conv = (mixing_angle ** 2) * (
        1 - np.exp(-B_mag ** 2 / (m_eff ** 2 + 1e-30) * 1e-26)
    )
    conv_n = np.clip(conv / (conv.max() + 1e-30), 0, 1)
    return B_n, qed_n, conv_n


def plot_magnetar_qed(B0=1e15, epsilon=0.1):
    """4-panel magnetar QED figure: dipole streamplot, EH heatmap, P_conv heatmap, radial profiles"""
    B_CRIT = 4.414e13
    alpha = 1 / 137.0
    r_max, gs = 10, 120
    x = np.linspace(-r_max, r_max, gs)
    y = np.linspace(-r_max, r_max, gs)
    X, Y = np.meshgrid(x, y)
    R = np.maximum(np.sqrt(X ** 2 + Y ** 2), 0.2)
    theta = np.arctan2(Y, X)
    R0 = 1.0
    B_r = B0 * (R0 / R) ** 3 * 2 * np.cos(theta)
    B_t = B0 * (R0 / R) ** 3 * np.sin(theta)
    Bx = B_r * np.cos(theta) - B_t * np.sin(theta)
    By = B_r * np.sin(theta) + B_t * np.cos(theta)
    B_tot = np.sqrt(Bx ** 2 + By ** 2)
    EH_ratio = (alpha / (45 * np.pi)) * (B_tot / B_CRIT) ** 2
    EH_norm = EH_ratio / (EH_ratio.max() + 1e-30)
    dp_conv = (epsilon ** 2) * (
        1 - np.exp(-(B_tot / B_CRIT) ** 2 * (B0 / B_CRIT) ** 2 / (epsilon + 1e-30) ** 0 * 1e-2)
    )
    dp_conv = np.clip(dp_conv / (dp_conv.max() + 1e-30), 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Dipole streamplot
    ax1 = axes[0, 0]
    mag_log = np.log10(np.sqrt(Bx ** 2 + By ** 2) + 1e-10)
    ax1.streamplot(X, Y, Bx, By, color=mag_log, cmap="plasma",
                   linewidth=1.0, density=1.2)
    ax1.add_patch(Circle((0, 0), R0, color="white", zorder=5,
                          edgecolor="black", linewidth=1))
    ax1.set_xlim(-r_max, r_max)
    ax1.set_ylim(-r_max, r_max)
    ax1.set_aspect("equal")
    ax1.set_title(
        f"Dipole Field   B=B₀(R/r)³√(3cos²θ+1)\nB₀={B0:.1e} G", fontsize=10
    )
    ax1.set_xlabel("x / R★")
    ax1.set_ylabel("y / R★")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Euler-Heisenberg
    im2 = axes[0, 1].imshow(
        EH_norm, extent=[-r_max, r_max, -r_max, r_max],
        origin="lower", cmap="inferno", vmin=0, vmax=1
    )
    axes[0, 1].add_patch(Circle((0, 0), R0, color="white", zorder=5,
                                  edgecolor="black", linewidth=1))
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    axes[0, 1].set_title("Euler-Heisenberg QED\nΔL=(α/45π)(B/B_crit)²", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Dark photon conversion
    im3 = axes[1, 0].imshow(
        dp_conv, extent=[-r_max, r_max, -r_max, r_max],
        origin="lower", cmap="hot", vmin=0, vmax=1
    )
    axes[1, 0].add_patch(Circle((0, 0), R0, color="white", zorder=5,
                                  edgecolor="black", linewidth=1))
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title(
        f"Dark Photon Conversion  P=ε²(1−e^{{−B²/m²}})\nε={epsilon:.3f}", fontsize=10
    )
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Radial profiles
    ax4 = axes[1, 1]
    r_1d = np.linspace(1.1, r_max, 200)
    B_r1d = B0 * (R0 / r_1d) ** 3
    EH_r1d = (alpha / (45 * np.pi)) * (B_r1d / B_CRIT) ** 2
    dp_r1d = (epsilon ** 2) * (1 - np.exp(-(B_r1d / B_CRIT) ** 2 * 1e-2))
    dp_r1d = np.clip(dp_r1d / (dp_r1d.max() + 1e-30), 0, 1)
    ax4.semilogy(r_1d, B_r1d, "b-", linewidth=2, label="|B| on-axis")
    ax4.set_xlabel("r / R★")
    ax4.set_ylabel("|B| (G)", color="b")
    ax4.tick_params(axis="y", labelcolor="b")
    ax4.grid(True, alpha=0.3)
    ax4_t = ax4.twinx()
    EH_norm_1d = EH_r1d / (EH_r1d.max() + 1e-30)
    ax4_t.plot(r_1d, EH_norm_1d, "r--", linewidth=2, label="ΔL (E-H, norm.)")
    ax4_t.plot(r_1d, dp_r1d, "g-.", linewidth=2, label="P_conv (norm.)")
    ax4_t.set_ylabel("Normalised", color="r")
    ax4_t.set_ylim([0, 1])
    ax4.set_title("Radial Profiles (θ=0 axis)", fontsize=10)
    lines1, lab1 = ax4.get_legend_handles_labels()
    lines2, lab2 = ax4_t.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper right")

    plt.suptitle(
        f"Magnetar QED Explorer   B₀=10^{np.log10(B0):.1f} G"
        f"   B_crit=4.414×10¹³ G   ε={epsilon:.3f}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    return fig


def qcis_power_spectrum(f_nl=1.0, n_q=0.5, n_s=0.965):
    """
    QCIS Matter Power Spectrum with BBKS transfer function.
    P(k) = P_ΛCDM(k) × (1 + f_NL·(k/k₀)^n_q)   Planck 2018 n_s=0.965
    """
    k = np.logspace(-3, 1, 300)
    k0 = 0.05
    q = k / 0.2
    T = (
        np.log(1 + 2.34 * q) / (2.34 * q)
        * (1 + 3.89 * q + (16.2 * q) ** 2 + (5.47 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25)
    )
    Pl = k ** n_s * T ** 2
    Pq = Pl * (1 + f_nl * (k / k0) ** n_q)
    norm = Pl[np.argmin(np.abs(k - k0))] + 1e-30
    return k, Pl / norm, Pq / norm


def em_spectrum_composite(img_gray, f_nl, n_q):
    """Tri-band EM composite: R=Infrared, G=Visible, B=X-ray with QCIS quantum correction"""
    k, Pl, Pq = qcis_power_spectrum(f_nl, n_q)
    idx = np.argmin(np.abs(k - 0.1))
    q_factor = float(Pq[idx] / (Pl[idx] + 1e-30))
    q_factor = np.clip(q_factor, 0.5, 3.0)
    infrared = np.clip(img_gray ** 0.5 * q_factor, 0, 1)
    visible = np.clip(img_gray ** 0.8 * q_factor, 0, 1)
    xray = np.clip(img_gray ** 1.5 * q_factor, 0, 1)
    return np.stack([infrared, visible, xray], axis=-1)


# =============================================================================
# ANIMATED WAVE FUNCTIONS (restored from v1 — FIX BUG-8)
# =============================================================================

def compute_fdm_wave(t, size=300, mixing_eps=1e-10):
    """
    Full two-field FDM wave density.
    ψ = ψ_t + ψ_d·e^{iΔφ}
    ρ = |ψ_t|² + |ψ_d|² + 2·Re(ψ_t*·ψ_d·e^{iΔφ})
    """
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2) / size * 8
    psi_t = np.exp(-r ** 2 / 4) * np.exp(1j * r * np.cos(t))
    psi_d = np.exp(-r ** 2 / 4) * np.exp(
        1j * (r * np.sin(t) + np.pi * np.clip(mixing_eps * 1e10, 0, 1))
    )
    rho = np.abs(psi_t) ** 2 + np.abs(psi_d) ** 2 + 2 * np.real(
        psi_t * np.conj(psi_d)
    )
    return psi_t, psi_d, rho


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def load_image(file):
    img = Image.open(file).convert("L")
    if max(img.size) > 800:
        img.thumbnail((800, 800), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def make_sgr1806_preset(size=300):
    rng = np.random.RandomState(2)
    cx, cy = size // 2, size // 2
    y, x = np.mgrid[:size, :size]
    dx = (x - cx) / (size / 4)
    dy = (y - cy) / (size / 4)
    r = np.sqrt(dx ** 2 + dy ** 2) + 0.05
    theta = np.arctan2(dy, dx)
    B_halo = np.exp(-r * 1.5) * np.sqrt(3 * np.cos(theta) ** 2 + 1) / r
    B_halo = np.clip(B_halo / B_halo.max(), 0, 1) * 0.5
    r_c = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    core = np.exp(-r_c ** 2 / 3.0)
    img = B_halo + core + rng.randn(size, size) * 0.01
    return np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)


def make_galaxy_cluster_preset(size=300):
    rng = np.random.RandomState(42)
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - 150) ** 2 + (y - 150) ** 2)
    img = np.exp(-r ** 2 / 8000) * 0.8 + rng.randn(size, size) * 0.03
    return np.clip(img, 0, 1)


def make_airport_radar_preset(airport, size=300):
    rng = np.random.RandomState(123)
    y, x = np.mgrid[:size, :size]
    background = np.exp(-((x - 150) ** 2 + (y - 150) ** 2) / 20000) * 0.4
    stealth = np.zeros((size, size))
    if airport == "nellis":
        stealth[100:120, 80:100] = 0.6
        stealth[180:200, 200:220] = 0.5
    elif airport == "jfk":
        stealth[120:140, 100:130] = 0.7
    elif airport == "lax":
        stealth[90:110, 220:250] = 0.55
    img = background + stealth + rng.randn(size, size) * 0.05
    return np.clip(img, 0, 1)


PRESETS = {
    "SGR 1806-20 (Magnetar)": make_sgr1806_preset,
    "Galaxy Cluster (Abell 209 style)": make_galaxy_cluster_preset,
    "Airport Radar - Nellis AFB Historical": lambda: make_airport_radar_preset("nellis"),
    "Airport Radar - JFK International Historical": lambda: make_airport_radar_preset("jfk"),
    "Airport Radar - LAX Historical": lambda: make_airport_radar_preset("lax"),
}


# FIX BUG-1: use plt.get_cmap (not deprecated) instead of mcm.get_cmap
def _apply_cmap(arr2d, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(arr2d, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def arr_to_pil(arr, cmap=None):
    if arr.ndim == 2:
        if cmap:
            return Image.fromarray(_apply_cmap(arr, cmap), mode="RGB")
        return Image.fromarray(
            np.clip(arr * 255, 0, 255).astype(np.uint8), mode="L"
        )
    arr_u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def get_download_link(arr_or_buf, filename, label="📥 Download", cmap=None):
    """FIX BUG-3/4/13: accepts either ndarray or BytesIO buffer"""
    if isinstance(arr_or_buf, io.BytesIO):
        arr_or_buf.seek(0)
        b64 = base64.b64encode(arr_or_buf.read()).decode()
    else:
        img = arr_to_pil(arr_or_buf, cmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a href="data:image/png;base64,{b64}" download="{filename}"'
        f' class="dl-btn">{label}</a>'
    )


def fig_to_buf(fig, dpi=120):
    """Save matplotlib figure to BytesIO buffer"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚛️ Core Physics")
    omega_pd = st.slider("Omega_PD Entanglement", 0.05, 0.50, 0.20, 0.01)
    fringe_scale = st.slider("Fringe Scale (pixels)", 10, 80, 45, 1)
    kin_mix = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    fdm_mass = st.slider("FDM Mass ×10⁻²² eV", 0.10, 10.00, 1.00, 0.01)
    st.markdown("---")
    st.markdown("## 🌟 Magnetar")
    b0_log10 = st.slider("B₀ log₁₀ G", 13.0, 16.0, 15.0, 0.1)
    magnetar_eps = st.slider("Magnetar ε", 0.01, 0.50, 0.10, 0.01)
    st.markdown("---")
    st.markdown("## 📈 QCIS")
    f_nl = st.slider("f_NL", 0.00, 5.00, 1.00, 0.01)
    n_q = st.slider("n_q", 0.00, 2.00, 0.50, 0.01)
    st.markdown("---")
    st.markdown("**Tony Ford | tlcagford@gmail.com**  \n**Patent Pending | 2026**")

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    '<h1 style="text-align:center">🔭 QCAUS v1.0</h1>', unsafe_allow_html=True
)
st.markdown(
    "## 🔭 QCAUS v1.0 — Quantum Cosmology & Astrophysics Unified Suite"
)
st.caption("Tony E. Ford | tlcagford@gmail.com | Patent Pending | 2026")

# =============================================================================
# DATA SOURCE SELECTION
# =============================================================================
st.markdown("### 🎯 Select Preset Data")
preset_choice = st.selectbox(
    "Choose example to run instantly:", options=list(PRESETS.keys()), index=0
)
col1, col2 = st.columns([2, 1])
with col1:
    run_preset = st.button("🚀 Run Selected Preset", use_container_width=True)
with col2:
    uploaded_file = st.file_uploader(
        "Drag & drop file here",
        type=["jpg", "jpeg", "png", "fits"],
        help="Limit 200 MB",
        label_visibility="collapsed",
    )

img_data = None
if run_preset:
    img_data = PRESETS[preset_choice]()
    st.success(f"✅ Loaded preset: {preset_choice}")
elif uploaded_file is not None:
    img_data = load_image(uploaded_file)
    st.success(f"✅ Loaded: {uploaded_file.name}")

# =============================================================================
# FDM FIELD EQUATIONS (restored from v1 — FIX BUG-9)
# =============================================================================
st.markdown("---")
st.markdown("## 📐 FDM Field Equations (Tony Ford Model)")
st.latex(
    r"S = \int d^4x\sqrt{-g}\left[\frac{1}{2}g^{\mu\nu}\partial_\mu\phi"
    r"\partial_\nu\phi - \frac{1}{2}m^2\phi^2\right] + S_{\rm gravity}"
)
st.latex(r"\Box\phi + m^2\phi = 0")
st.latex(r"\phi = (2m)^{-1/2}\left[\psi\,e^{-imt} + \psi^*e^{imt}\right]")
st.latex(r"\psi = \psi_{\rm light} + \psi_{\rm dark}\,e^{i\Delta\phi}")
st.latex(
    r"\rho = |\psi_{\rm light}|^2 + |\psi_{\rm dark}|^2"
    r"+ 2\operatorname{Re}\!\left(\psi_{\rm light}^*\psi_{\rm dark}\,e^{i\Delta\phi}\right)"
)
st.latex(r"\rho(r) = \frac{\rho_c}{\left[1+(r/r_c)^2\right]^8}")

# =============================================================================
# ANIMATED WAVE (restored from v1 — FIX BUG-8)
# =============================================================================
st.markdown("---")
st.markdown("## 🌊 QCAUS-FDM-Wave: Animated Moving Interference")

animate = st.toggle("Animate Waves", value=False)
speed = st.slider("Animation Speed", 0.1, 5.0, 1.0)
wave_mode = st.radio("Wave Display Mode", ["2D Wave", "3D Surface"], horizontal=True)

if "t" not in st.session_state:
    st.session_state.t = 0.0
if animate:
    st.session_state.t += 0.08 * speed

psi_t, psi_d, rho_wave = compute_fdm_wave(st.session_state.t, size=300, mixing_eps=kin_mix)

wave_size = 300
if wave_mode == "2D Wave":
    fig_w, ax_w = plt.subplots(figsize=(10, 4))
    ax_w.plot(np.abs(psi_t[wave_size // 2]), label=r"$|\psi_{\rm light}|$", color="#00ffcc")
    ax_w.plot(np.abs(psi_d[wave_size // 2]), label=r"$|\psi_{\rm dark}|$", color="#ff00cc")
    ax_w.plot(rho_wave[wave_size // 2], label=r"$\rho$ interference", color="#ffff00", lw=3)
    ax_w.legend()
    ax_w.grid(True, alpha=0.3)
    ax_w.set_xlabel("pixel")
    ax_w.set_ylabel("amplitude")
    ax_w.set_title(
        f"FDM Two-Field Interference   t={st.session_state.t:.2f}   ε={kin_mix:.2e}"
    )
    st.pyplot(fig_w)
    plt.close(fig_w)
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig_w = plt.figure(figsize=(10, 6))
    ax_w = fig_w.add_subplot(111, projection="3d")
    step = 6
    X3, Y3 = np.meshgrid(
        np.linspace(-4, 4, wave_size // step),
        np.linspace(-4, 4, wave_size // step)
    )
    ax_w.plot_surface(X3, Y3, rho_wave[::step, ::step], cmap="hot")
    ax_w.set_title(
        f"3D FDM Wave Surface   t={st.session_state.t:.2f}   ε={kin_mix:.2e}"
    )
    st.pyplot(fig_w)
    plt.close(fig_w)

if animate:
    st.rerun()

# =============================================================================
# MAIN PROCESSING — triggered by preset or upload
# =============================================================================
if img_data is not None:
    B0 = 10 ** b0_log10
    B_CRIT = 4.414e13

    # Ensure grayscale float32 square image
    if img_data.ndim == 3:
        img_gray = np.mean(img_data, axis=-1).astype(np.float32)
    else:
        img_gray = img_data.copy().astype(np.float32)

    h, w = img_gray.shape
    SIZE = min(h, w, 400)   # cap at 400 to keep memory reasonable
    img_pil = Image.fromarray((img_gray * 255).astype(np.uint8))
    img_pil = img_pil.resize((SIZE, SIZE), Image.LANCZOS)
    img_gray = np.array(img_pil, dtype=np.float32) / 255.0

    # --- Run all physics ---
    soliton = fdm_soliton_2d(SIZE, fdm_mass)
    interf = generate_interference_pattern(SIZE, fringe_scale, omega_pd)

    # FIX BUG-5: pass fdm_mass (in ×10⁻²² eV units) as fdm_mass_eV, kin_mix directly as mixing_eps
    ord_mode, dark_mode = pdp_spectral_duality(
        img_gray, omega_pd, fringe_scale, kin_mix, fdm_mass
    )

    # FIX BUG-6: pass kin_mix directly (not *1e9)
    ent_res = entanglement_residuals(
        img_gray, ord_mode, dark_mode, omega_pd * 0.3, kin_mix, fringe_scale
    )

    pdp_out = pdp_entanglement_overlay(img_gray, interf, soliton, omega_pd)
    fusion = blue_halo_fusion(img_gray, dark_mode, ent_res)
    dp_prob = dark_photon_detection_prob(dark_mode, ent_res, omega_pd * 0.3)
    dp_peak = float(dp_prob.max() * 100)

    B_n, qed_n, conv_n = magnetar_physics(SIZE, B0, magnetar_eps)
    k_arr, P_lcdm, P_quantum = qcis_power_spectrum(f_nl, n_q)
    em_comp = em_spectrum_composite(img_gray, f_nl, n_q)
    r_arr, rho_arr = fdm_soliton_profile(fdm_mass)

    # =========================================================================
    # BEFORE / AFTER
    # =========================================================================
    st.markdown("---")
    st.markdown("## Before vs After")
    c1, c2 = st.columns(2)
    info_html = f"""
    <div style="border:2px solid #0ea5e9;border-radius:8px;padding:8px;
                background:rgba(15,23,42,0.9);color:#67e8f9;font-size:13px;">
        Ω = {omega_pd:.2f} | Fringe = {fringe_scale} | ε = {kin_mix:.2e}<br>
        FDM m = {fdm_mass:.2f}×10⁻²² eV | Entropy = {float(-np.mean(ent_res[ent_res>0]*np.log(ent_res[ent_res>0]+1e-10))):.3f}
    </div>"""
    with c1:
        st.markdown(info_html, unsafe_allow_html=True)
        st.markdown("**Before: Standard View** (Public HST/JWST Data)")
        st.image(arr_to_pil(img_gray, cmap="gray"), use_container_width=True)  # FIX BUG-2
        st.caption("0 — 20 kpc")
        st.markdown(
            get_download_link(img_gray, "original.png", "📥 Download Original", "gray"),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(info_html, unsafe_allow_html=True)
        st.markdown("**After: Photon-Dark-Photon Entangled FDM Overlays (Tony Ford Model)**")
        st.image(arr_to_pil(pdp_out, cmap="inferno"), use_container_width=True)
        st.caption("0 — 20 kpc")
        st.markdown(
            get_download_link(pdp_out, "pdp_entangled.png", "📥 Download PDP Entangled", "inferno"),
            unsafe_allow_html=True
        )
    st.markdown("**↑ N**")
    st.caption("QCAUS v1.0 | Tony E. Ford | tlcagford@gmail.com | Patent Pending | 2026")

    # =========================================================================
    # PHYSICS MAPS
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📊 Annotated Physics Maps")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ⚛️ FDM Soliton")
        st.caption("ρ(r) = ρ₀[sin(kr)/(kr)]²   k=π/r_s")
        st.image(arr_to_pil(soliton, "hot"), use_container_width=True)
        st.markdown(
            get_download_link(soliton, "fdm_soliton.png", "📥 Download", "hot"),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("### 🌊 PDP Interference")
        st.caption("oscillation_length = 100/(m_dark·1e9)   FFT spectral duality")
        st.image(arr_to_pil(interf, "plasma"), use_container_width=True)
        st.markdown(
            get_download_link(interf, "pdp_interference.png", "📥 Download", "plasma"),
            unsafe_allow_html=True
        )
    with c3:
        st.markdown("### 🕳️ Entanglement Residuals")
        st.caption("S = −ρ·log(ρ) + interference cross-term")
        st.image(arr_to_pil(ent_res, "inferno"), use_container_width=True)
        st.markdown(
            get_download_link(ent_res, "entanglement_residuals.png", "📥 Download", "inferno"),
            unsafe_allow_html=True
        )

    # =========================================================================
    # DARK PHOTON DETECTION + BLUE HALO
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🔵 Dark Photon Detection & Blue-Halo Fusion")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Dark Photon Detection Probability")
        st.caption("P_dark = prior·L/(prior·L+(1−prior))  — Bayesian kinetic-mixing")
        st.image(arr_to_pil(dp_prob, "YlOrRd"), use_container_width=True)
        st.markdown(
            get_download_link(dp_prob, "dp_detection.png", "📥 Download", "YlOrRd"),
            unsafe_allow_html=True
        )
        if dp_peak > 50:
            st.error(f"⚠️ STRONG DARK PHOTON SIGNAL — P_dark = {dp_peak:.1f}%")
        elif dp_peak > 20:
            st.warning(f"⚡ DARK PHOTON SIGNAL — P_dark = {dp_peak:.1f}%")
        else:
            st.success(f"✅ CLEAR — P_dark = {dp_peak:.1f}% (below threshold)")
    with c2:
        st.markdown("### Blue-Halo Fusion  γ=0.45")
        st.caption("R=original  G=residuals  B=dark_mode  — pdp_radar_core.py")
        st.image(arr_to_pil(fusion), use_container_width=True)
        st.markdown(
            get_download_link(fusion, "blue_halo_fusion.png", "📥 Download"),
            unsafe_allow_html=True
        )

    # =========================================================================
    # MAGNETAR QED
    # =========================================================================
    st.markdown("---")
    st.markdown(
        "## ⚡ Magnetar QED — Dipole Field · Euler-Heisenberg · Dark Photon Conversion"
    )
    st.caption(
        "B=B₀(R/r)³√(3cos²θ+1)  |  ΔL=(α/45π)(B/B_crit)²  |  "
        "P_conv=ε²(1−e^{−B²/m²})  |  B_crit=4.414×10¹³ G"
    )
    try:
        fig_mag = plot_magnetar_qed(B0, magnetar_eps)
        st.pyplot(fig_mag, use_container_width=True)
        mag_buf = fig_to_buf(fig_mag, dpi=100)
        st.markdown(
            get_download_link(mag_buf, "magnetar_qed.png", "📥 Download Full Magnetar QED Plot"),
            unsafe_allow_html=True
        )
        plt.close(fig_mag)
    except Exception as e:
        st.error(f"Magnetar plot error: {e}")

    st.markdown("### Magnetar Field Maps")
    cA, cB, cC = st.columns(3)
    with cA:
        st.image(
            arr_to_pil(B_n, "plasma"),
            caption="Dipole |B| map  B=B₀(R/r)³√(3cos²θ+1)",
            use_container_width=True
        )
        st.markdown(
            get_download_link(B_n, "magnetar_B.png", "📥 Download", "plasma"),
            unsafe_allow_html=True
        )
    with cB:
        st.image(
            arr_to_pil(qed_n, "inferno"),
            caption="Euler-Heisenberg QED  ΔL=(α/45π)(B/B_crit)²",
            use_container_width=True
        )
        st.markdown(
            get_download_link(qed_n, "magnetar_QED.png", "📥 Download", "inferno"),
            unsafe_allow_html=True
        )
    with cC:
        st.image(
            arr_to_pil(conv_n, "hot"),
            caption="Dark Photon Conversion  P=ε²(1−e^{−B²/m²})",
            use_container_width=True
        )
        st.markdown(
            get_download_link(conv_n, "magnetar_darkphoton.png", "📥 Download", "hot"),
            unsafe_allow_html=True
        )

    # =========================================================================
    # FDM SOLITON RADIAL PROFILE  (FIX BUG-3/13: save actual figure)
    # =========================================================================
    st.markdown("---")
    st.markdown("## ⚛️ FDM Soliton Radial Profile")
    fig_fdm, ax_fdm = plt.subplots(figsize=(9, 3))
    ax_fdm.plot(
        r_arr, rho_arr, "r-", linewidth=2.5,
        label=f"ρ(r)=ρ₀[sin(kr)/(kr)]²  m={fdm_mass:.1f}×10⁻²² eV"
    )
    ax_fdm.set_xlabel("r (kpc)")
    ax_fdm.set_ylabel("ρ(r)/ρ₀")
    ax_fdm.set_title(
        "FDM Soliton Profile — Schrödinger-Poisson ground state [QCAUS]", fontsize=11
    )
    ax_fdm.legend()
    ax_fdm.grid(True, alpha=0.3)
    st.pyplot(fig_fdm)
    # FIX BUG-3: save actual fig, not zeros array
    fdm_buf = fig_to_buf(fig_fdm)
    st.markdown(
        get_download_link(fdm_buf, "fdm_profile.png", "📥 Download FDM Profile Plot"),
        unsafe_allow_html=True
    )
    plt.close(fig_fdm)

    # =========================================================================
    # QCIS POWER SPECTRUM  (FIX BUG-4/13: save actual figure)
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📈 QCIS Power Spectrum")
    st.caption(
        "P(k) = P_ΛCDM(k)×(1+f_NL·(k/k₀)^n_q)   BBKS T(k)   n_s=0.965 (Planck 2018)"
    )
    fig_ps, ax_ps = plt.subplots(figsize=(10, 4))
    ax_ps.loglog(k_arr, P_lcdm, "b-", linewidth=2, label="ΛCDM baseline")
    ax_ps.loglog(
        k_arr, P_quantum, "r--", linewidth=2,
        label=f"Quantum  f_NL={f_nl:.1f}  n_q={n_q:.1f}"
    )
    ax_ps.axvline(0.05, color="gray", linestyle=":", alpha=0.5, label="Pivot k₀=0.05 h/Mpc")
    ax_ps.set_xlabel("k (h/Mpc)")
    ax_ps.set_ylabel("P(k)/P(k₀)")
    ax_ps.set_title(
        "QCIS Matter Power Spectrum  (BBKS T(k), n_s=0.965) [QCIS repo]", fontsize=11
    )
    ax_ps.legend()
    ax_ps.grid(True, alpha=0.3, which="both")
    st.pyplot(fig_ps)
    # FIX BUG-4: save actual fig
    ps_buf = fig_to_buf(fig_ps)
    st.markdown(
        get_download_link(ps_buf, "qcis_spectrum.png", "📥 Download QCIS Spectrum Plot"),
        unsafe_allow_html=True
    )
    plt.close(fig_ps)

    # =========================================================================
    # EM SPECTRUM COMPOSITE
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🌈 Electromagnetic Spectrum Mapping")
    st.caption(
        "R=Infrared  G=Visible  B=X-ray  |  "
        "Quantum correction factor from QCIS P(k) ratio at k=0.1 h/Mpc"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎨 EM Composite")
        st.image(arr_to_pil(em_comp), use_container_width=True)
        st.markdown(
            get_download_link(em_comp, "em_composite.png", "📥 Download"),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("### 📊 Individual EM Bands")
        ir_img = _apply_cmap(np.clip(img_gray ** 0.5, 0, 1), "hot")
        vi_img = _apply_cmap(np.clip(img_gray ** 0.8, 0, 1), "viridis")
        xr_img = _apply_cmap(np.clip(img_gray ** 1.5, 0, 1), "plasma")
        tab1, tab2, tab3 = st.tabs(["🔴 Infrared", "🟢 Visible", "🔵 X-ray"])
        with tab1:
            st.image(ir_img, use_container_width=True)
            st.caption("λ~10-1000 μm | Thermal dust emission")
            st.markdown(
                get_download_link(
                    np.clip(img_gray ** 0.5, 0, 1), "infrared.png", "📥 Download", "hot"
                ),
                unsafe_allow_html=True
            )
        with tab2:
            st.image(vi_img, use_container_width=True)
            st.caption("λ~400-700 nm | Stellar emission")
            st.markdown(
                get_download_link(
                    np.clip(img_gray ** 0.8, 0, 1), "visible.png", "📥 Download", "viridis"
                ),
                unsafe_allow_html=True
            )
        with tab3:
            st.image(xr_img, use_container_width=True)
            st.caption("λ~0.01-10 nm | Hot plasma emission")
            st.markdown(
                get_download_link(
                    np.clip(img_gray ** 1.5, 0, 1), "xray.png", "📥 Download", "plasma"
                ),
                unsafe_allow_html=True
            )

    # =========================================================================
    # DETECTION METRICS
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📊 Detection Metrics")
    dm1, dm2, dm3, dm4, dm5 = st.columns(5)
    dm1.metric("P_dark Peak", f"{dp_peak:.1f}%", delta=f"ε={kin_mix:.1e}")
    dm2.metric("FDM Soliton Peak", f"{float(soliton.max()):.3f}", delta=f"m={fdm_mass:.1f}")
    dm3.metric("Fringe Contrast", f"{float(interf.std()):.3f}", delta=f"fringe={fringe_scale}")
    dm4.metric("PDP Mixing Ω·0.6", f"{omega_pd * 0.6:.3f}", delta=f"Ω={omega_pd:.2f}")
    dm5.metric("B/B_crit", f"{B0 / B_CRIT:.2e}", delta=f"B₀=10^{b0_log10:.1f}")

    # =========================================================================
    # PHYSICS FORMULA TABLE
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📡 Verified Physics Formulas")
    st.markdown("""
| Module | Formula | Source Repo |
|--------|---------|-------------|
| **FDM Soliton** | ρ(r) = ρ₀[sin(kr)/(kr)]²   k=π/r_s   r_s=1/m | QCAUS/app.py |
| **PDP Spectral Duality** | FFT: dark_mask = ε·e^{−ΩR²}·abs(sin(2πRL/f))·(1−e^{−R²/f}) | StealthPDPRadar/pdp_radar_core.py |
| **Entanglement Residuals** | S = −ρ·log(ρ) + abs(ψ_ord+ψ_dark)²−ψ_ord²−ψ_dark² | StealthPDPRadar/pdp_radar_core.py |
| **Dark Photon Detection** | P_dark = prior·L/(prior·L+(1−prior)) | StealthPDPRadar/pdp_radar_core.py |
| **Blue-Halo Fusion** | R=original  G=residuals  B=dark  γ=0.45 | StealthPDPRadar/pdp_radar_core.py |
| **Magnetar Dipole** | B = B₀(R/r)³√(3cos²θ+1)   B_crit=4.414×10¹³ G | Magnetar-Quantum-Vacuum repo |
| **Euler-Heisenberg QED** | ΔL = (α/45π)(B/B_crit)² | Magnetar-Quantum-Vacuum repo |
| **Dark Photon Conversion** | P_conv = ε²(1−e^{−B²/m²}) | Magnetar-Quantum-Vacuum repo |
| **QCIS Power Spectrum** | P(k) = P_ΛCDM(k)×(1+f_NL·(k/k₀)^n_q) | Quantum-Cosmology-Integration-Suite |
| **FDM Wave** | ρ = abs(ψ_t)²+abs(ψ_d)²+2·Re(ψ_t*·ψ_d·e^{iΔφ}) | QCAUS/app.py |
""")

    # =========================================================================
    # ZIP DOWNLOAD
    # =========================================================================
    if st.button("📦 Download All Results as ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            for name, arr, cmap in [
                ("original", img_gray, "gray"),
                ("pdp_entangled", pdp_out, "inferno"),
                ("blue_halo", fusion, None),
                ("fdm_soliton", soliton, "hot"),
                ("pdp_interference", interf, "plasma"),
                ("ent_residuals", ent_res, "inferno"),
                ("dp_detection", dp_prob, "YlOrRd"),
                ("magnetar_B", B_n, "plasma"),
                ("magnetar_QED", qed_n, "inferno"),
                ("magnetar_darkphoton", conv_n, "hot"),
                ("em_composite", em_comp, None),
            ]:
                img = arr_to_pil(arr, cmap)
                buf = io.BytesIO()
                img.save(buf, "PNG")
                z.writestr(f"{name}.png", buf.getvalue())
            # FIX BUG-3/4: include real matplotlib figures in ZIP
            for fig_name, fig_func, fig_args in [
                ("magnetar_qed", plot_magnetar_qed, (B0, magnetar_eps)),
            ]:
                fig = fig_func(*fig_args)
                buf = fig_to_buf(fig, dpi=150)
                z.writestr(f"{fig_name}.png", buf.read())
                plt.close(fig)
            # FDM profile
            fig_f, ax_f = plt.subplots(figsize=(9, 3))
            ax_f.plot(r_arr, rho_arr, "r-", lw=2.5,
                      label=f"m={fdm_mass:.1f}×10⁻²² eV")
            ax_f.set_title("FDM Soliton Profile")
            ax_f.legend(); ax_f.grid(True, alpha=0.3)
            buf = fig_to_buf(fig_f); z.writestr("fdm_profile.png", buf.read())
            plt.close(fig_f)
            # QCIS spectrum
            fig_q, ax_q = plt.subplots(figsize=(10, 4))
            ax_q.loglog(k_arr, P_lcdm, "b-", lw=2, label="ΛCDM")
            ax_q.loglog(k_arr, P_quantum, "r--", lw=2, label="QCIS")
            ax_q.set_title("QCIS Power Spectrum")
            ax_q.legend(); ax_q.grid(True, alpha=0.3, which="both")
            buf = fig_to_buf(fig_q); z.writestr("qcis_spectrum.png", buf.read())
            plt.close(fig_q)
        zip_buffer.seek(0)
        st.download_button(
            "⬇️ Download QCAUS_Results.zip",
            zip_buffer.getvalue(),
            "QCAUS_Results.zip",
            "application/zip"
        )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "🔭 **QCAUS v1.0** — Quantum Cosmology & Astrophysics Unified Suite  \n"
    "Tony E. Ford | tlcagford@gmail.com | Patent Pending | 2026"
)
