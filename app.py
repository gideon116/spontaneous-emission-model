import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

KB = 8.617333e-5  # eV K^-1
h  = 4.135667e-15  # eV s
c  = 2.99792458e8  # m s^-1
h_b = h/2*np.pi

def T_D(ex, eg, eu, th):
    return (1/(2 * eu)) * np.exp(-(np.abs((ex-eg)/eu)**th))

def G_s(ep, eg, eu, th, ex):
    return T_D(ep, 0, eu, th) * np.sqrt((ex - eg) - ep)

def G(a, b, n, eg, eu, th, ex):

    h = (b - a) / n
    integral = 0.5 * G_s(a, eg, eu, th, ex) + 0.5 * G_s(b, eg, eu, th, ex)
    for i in range(1, n):
        integral += G_s(a + i * h, eg, eu, th, ex)
    integral *= h
    
    return integral

    # xs     = np.linspace(a, b, n+1)
    # ys     = G_s(xs, eg, eu, th, ex)
    # return np.trapezoid(ys, xs)

def alpha_s(eg, eu, th, ex, alpha_0):
    return G(-eg, ex - eg, 1000, eg, eu, th, ex) * alpha_0

def ed(ex, ef, eh, kt):
    return (1 - 1/(np.exp((ex - eh)/kt) + 1)) - 1/(np.exp((ex - ef)/kt) + 1)

def a_s(eg, eu, th, ex, alpha_0, ef, eh, kt):
    return 1 - np.exp(-alpha_s(eg, eu, th, ex, alpha_0) * ed(ex, ef, eh, kt))

def I_sn(eg, ex, eu, alpha_0, kt, ef, eh, th):
    return (2 * np.pi / ((h**3) * (c**2))) * ((ex**2)/(np.exp((ex - (ef - eh))/kt) - 1)) \
        * a_s(eg, eu, th, ex, alpha_0, ef, eh, kt)

def pl_calc(hv, eg, eu, alpha_0, temp, ef, eh, th):
    kt = KB * temp
    return I_sn(eg, hv, eu, alpha_0, kt, ef, eh, th)

def nii(T, Eg=1.42, N_D=1e23, dn=1e20, m_e=0.063, m_h=0.51, doped=True):
    
    k = 1.380649e-23
    h = 6.626e-34
    m0 = 9.109e-31

    m_e_star = m_e * m0
    m_h_star = m_h * m0

    Nc = 2 * ((2 * np.pi * m_e_star * k * T / (h**2))**(3/2))
    Nv = 2 * ((2 * np.pi * m_h_star * k * T / (h**2))**(3/2))

    ni = np.sqrt(Nc * Nv * np.exp(-Eg/(KB*T)))
    
    if doped:
        n0 = N_D
        p0 = (ni**2/N_D)
    else:
        # m_e* x m_h* = ((n0*p0 * np.exp(Eg/(KB*T)) / ((2 * ((2 * np.pi * k * T / (h**2))**(3/2)))**2))**(2/3))/(m0**2)
        n0 = ni
        p0 = ni

    Efe = Eg - KB * T * np.log(Nc/(n0 + dn))
    Efh = KB * T * np.log(Nv/(p0 + dn))
    
    return Efe, Efh

# sidebar
st.sidebar.title("Spontaneous Emission Model for Nonequilibrium Conditions")


# uploading
use_default = st.sidebar.checkbox("Use default GaAs PL", False)

if use_default:
    
    default_path = "default_GaAs_PL.txt"
    raw = open("GaAs_PL.txt", "r").read()

else:
    uploaded = st.sidebar.file_uploader(
        "Or upload your own PL file",
        type=["txt", "csv"]
    )
    if uploaded is None:
        st.sidebar.info("Make sure you have two columns: one labeled 'wl' that has wavelengths in nm, \
            and another column containing the PL intensity. If you just want to try the model out, \
                        select 'Use default GaAs PL'")
        st.stop()
    try:
        raw = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.sidebar.error("Cannot decode file - must be plain ASCII/UTF-8 text.")
        st.stop()


lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
header = lines[0].replace(",", "\t").split("\t")
if len(header) != 2:
    st.sidebar.error("File must have exactly two columns: wavelength + intensity.")
    st.stop()
if header[0].lower() not in ("wl", "wavelength", "lambda"):
    st.sidebar.error("First column header must be 'wl' or 'wavelength'.")
    st.stop()

try:
    arr = np.loadtxt(io.StringIO("\n".join(lines[1:])))
except Exception as e:
    st.sidebar.error(f"Could not parse numeric data: {e}")
    st.stop()

wl = arr[:, 0]
I_raw = arr[:, 1] # counts (or counts/s)

with st.sidebar.expander("Processing options", expanded=True):

    baseline_type = st.selectbox("Baseline correction", ["Yes", "None"])
    
    # crop
    wl_min, wl_max = float(wl.min()), float(wl.max())
    crop = st.slider("Wavelength window (nm)", wl_min, wl_max, (wl_min, wl_max), 1.0)

    # counts or counts/s?
    is_cps = st.checkbox("Data is in counts / s", False)
    acq_t = st.number_input("Acquisition time, s", 0.0, 1e4, 3.0, 0.1, disabled=not is_cps)

    # spot size
    side_um = st.number_input("Spot length (µm) - if square", 1.0, 1000.0, 40.0, 1.0)
    area_m2 = (side_um * 1e-6) ** 2

    # counts to photons
    keep_default = st.checkbox("Use default counts→photons factor (5x10¹¹)", True)
    factor = st.number_input("Custom factor", 1e8, 1e14, 5e11, 1e10, format="%.2e", disabled=keep_default)

    # QE correction
    corr_QE = st.checkbox("Apply detector-QE correction", True)

def process_single_pl(wl_nm, counts, *, baseline="Yes", cps=False, acq_time=3.0, factor=5e11, area_m2=1.6e-9, 
                      apply_qe=True, crop_lo=None, crop_hi=None):
    
    if baseline == "Yes":
        counts = counts - np.min(counts)

    # crop wavelength
    mask = (wl_nm >= crop_lo) & (wl_nm <= crop_hi)
    w = wl_nm[mask]
    I = counts[mask].astype(float)

    # counts/s to counts
    if cps:
        I *= acq_time

    # counts to photons
    I *= factor * (1240.0 / w)

    # photons to photons m^2
    I /= area_m2

    # QE correction (only 650–1050 nm calibrated)
    if apply_qe:
        qe = (3.611624475091300E-13*w**6 - 1.917725566132970E-09*w**5
              + 4.221574705418870E-06*w**4 - 4.927570402158930E-03*w**3
              + 3.213591581944010E+00*w**2 - 1.109299740317640E+03*w
              + 1.582787629983810E+05) / 100.0
        I /= qe

    return w, I


wl_proc, I_meas = process_single_pl(wl, I_raw, baseline=baseline_type, cps=is_cps, acq_time=acq_t, factor=(5e11 if keep_default else factor),
                                    area_m2=area_m2, apply_qe=corr_QE, crop_lo=crop[0], crop_hi=crop[1])

hv   = 1240.0 / wl_proc

# params
st.sidebar.title("Model parameters")

Eg = st.sidebar.slider("E₉ (eV)", 1.0, 2.0, 1.42, 0.001, format="%.4f")
Eu = st.sidebar.slider("Eᵤ (eV)", 0.001, 0.05, 0.009, 0.0005, format="%.4f")
alpha0d = st.sidebar.slider("α₀ x d", 1.0, 100.0, 50.0)
T = st.sidebar.slider("Temperature (K)", 50, 400, 300, 1)
th = st.sidebar.slider("θ exponent", 0.5, 3.0, 1.0, 0.05)

# Fermi levels
with st.sidebar.expander("Processingd options", expanded=True):
    get_manual_QFLS = st.selectbox(r"**Choose how to detemine $\mathsf{E_F^e}$ and $\mathsf{E_F^h}$**", 
                            ["Calculate", "Manual"])

    if get_manual_QFLS == "Manual":
        Ef = st.number_input(r"$\mathsf{E_F^e}$ (eV)",  0.0, Eg, 1.25)
        Eh = st.number_input(r"$\mathsf{E_F^e}$ (eV)",  0.0, Eg, 0.15)
    else:
        dn = st.slider("Δn = 10^x (m⁻³)", 14.0, 29.0, 23.0, step=0.001)
        dn = 10 ** dn

        doped = st.checkbox("Doped material?", True)
        N_D = st.number_input("N_D (m⁻³)", 1e14, 1e29, 1e17, 1e17, format="%.2e")
        m_e = st.number_input("mₑ* / m₀", 0.01, 5.0, 0.063, 0.01)
        m_h = st.number_input("mₕ* / m₀", 0.01, 5.0, 0.51, 0.01)

        Ef, Eh = nii(T, Eg=Eg, N_D=N_D, dn=dn, m_e=m_e, m_h=m_h, doped=doped)
        st.latex(fr"E_F^e = {Ef:.4f}\,\mathrm{{eV}},\quad E_F^h = {Eh:.4f}\,\mathrm{{eV}}")


# calc
I_calc = pl_calc(hv, Eg, Eu, alpha0d, T, Ef, Eh, th)

# if normalization
norm_choice = st.radio("Normalize", ["No", "Yes"], horizontal=True)


if norm_choice == "Yes":
    I_meas = I_meas / I_meas.max()
    I_calc = I_calc / I_calc.max()

# plot

fig, ax = plt.subplots()
ax.plot(hv, I_meas, label="Measured", lw=2)
ax.plot(hv, I_calc, label="Model",    lw=2, alpha=0.8)
ax.set_xlabel("Photon energy (eV)")
ax.set_ylabel(r"Absolute Intensity ($\mathsf{\frac{Photons}{m^2 \ s \ eV}}$)")
ax.legend()
ax.invert_xaxis()

st.pyplot(fig)

# Now print EF and EH *below* the figure, each on its own line:
st.markdown(rf"$E_F^e = {Ef:.4f}\,\mathrm{{eV}}$")
st.markdown(rf"$E_F^h = {Eh:.4f}\,\mathrm{{eV}}$")
