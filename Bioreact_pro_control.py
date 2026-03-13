import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(page_title="BioReact Pro Control", layout="wide")

# =========================================================
# ESTILOS
# =========================================================
st.markdown("""
<style>
:root {
    --bg-soft: #0f172a;
    --card: #111827;
    --card-2: #172033;
    --border: rgba(166, 206, 99, 0.22);
    --accent: #a6ce63;
    --accent-2: #d29bff;
    --text: #f5f7fb;
    --muted: #aab5c8;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}

.host-cell-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0 16px 0;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}

.app-icon-container {
    width: 54px;
    height: 54px;
    background: linear-gradient(180deg, #1a2544 0%, #121a31 100%);
    border: 1px solid #2a3557;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
    flex-shrink: 0;
}

.app-icon-text {
    color: var(--accent);
    font-weight: 900;
    font-size: 20px;
    letter-spacing: 0.02em;
}

.brand-content {
    line-height: 1.15;
}

.app-title-main {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0;
    color: var(--text);
}

.app-subtitle-main {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--muted);
    margin: 0.15rem 0 0 0;
}

.author-highlight {
    color: var(--accent);
    font-weight: 700;
}

.section-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(23,32,51,0.98) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1rem 1rem 0.6rem 1rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.14);
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(23,32,51,0.98) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.kicker {
    display: inline-block;
    padding: 0.22rem 0.58rem;
    border: 1px solid rgba(166,206,99,0.28);
    color: var(--muted);
    border-radius: 999px;
    font-size: 0.76rem;
    margin-bottom: 0.55rem;
}

.small-muted {
    color: var(--muted);
    font-size: 0.92rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 0.9rem;
}
</style>

<div class="host-cell-header">
    <div class="app-icon-container">
        <span class="app-icon-text">BR</span>
    </div>
    <div class="brand-content">
        <h1 class="app-title-main">BioReact Pro</h1>
        <p class="app-subtitle-main">
            Host Cell Lab Suite • Tool by <span class="author-highlight">Emiliano Balderas Ramírez</span>, Bioengineer PhD student
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# UTILIDADES DE UI
# =========================================================
def _sync_from_slider(name: str):
    st.session_state[f"{name}_num"] = st.session_state[f"{name}_slider"]

def _sync_from_num(name: str, min_value: float, max_value: float):
    val = float(st.session_state[f"{name}_num"])
    val = max(min_value, min(max_value, val))
    st.session_state[f"{name}_num"] = val
    st.session_state[f"{name}_slider"] = val

def slider_with_exact_input(
    label: str,
    name: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float,
    fmt: str = "%.3f",
):
    if f"{name}_slider" not in st.session_state:
        st.session_state[f"{name}_slider"] = default
    if f"{name}_num" not in st.session_state:
        st.session_state[f"{name}_num"] = default

    st.session_state[f"{name}_slider"] = max(min_value, min(max_value, float(st.session_state[f"{name}_slider"])))
    st.session_state[f"{name}_num"] = max(min_value, min(max_value, float(st.session_state[f"{name}_num"])))

    c1, c2 = st.sidebar.columns([3.0, 1.55])

    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=f"{name}_slider",
            on_change=_sync_from_slider,
            args=(name,),
        )

    with c2:
        st.number_input(
            "Valor",
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=fmt,
            key=f"{name}_num",
            on_change=_sync_from_num,
            args=(name, min_value, max_value),
            label_visibility="collapsed",
        )

    return float(st.session_state[f"{name}_num"])

def matrix_to_latex(A, name="J"):
    rows = []
    for row in A:
        rows.append(" & ".join(f"{float(v):.6f}" for v in row))
    body = r"\\ ".join(rows)
    return f"{name}=" + r"\begin{bmatrix}" + body + r"\end{bmatrix}"

# =========================================================
# MODELOS CINÉTICOS
# =========================================================
def mu_monod(S, p):
    S = max(float(S), 0.0)
    denom = p["Ks"] + S
    return (p["mu_max"] * S / denom) if denom > 0 else 0.0

def dmu_dS_monod(S, p):
    S = max(float(S), 0.0)
    denom = (p["Ks"] + S) ** 2
    return (p["mu_max"] * p["Ks"] / denom) if denom > 0 else 0.0

def mu_haldane(S, p):
    S = max(float(S), 0.0)
    denom = p["Ks"] + S + (S**2 / p["Ki"])
    return (p["mu_max"] * S / denom) if denom > 0 else 0.0

def dmu_dS_haldane(S, p):
    S = max(float(S), 0.0)
    denom = (p["Ks"] + S + (S**2 / p["Ki"])) ** 2
    return (p["mu_max"] * (p["Ks"] - (S**2 / p["Ki"])) / denom) if denom > 0 else 0.0

def get_mu(S, p, model):
    return mu_monod(S, p) if model == "Monod" else mu_haldane(S, p)

def get_dmu_dS(S, p, model):
    return dmu_dS_monod(S, p) if model == "Monod" else dmu_dS_haldane(S, p)

# =========================================================
# BALANCES
# =========================================================
def rhs(y, D, p, model):
    X, S = y
    mu = get_mu(S, p, model)
    return np.array([
        X * (mu - D),
        D * (p["Sr"] - S) - (mu * X / p["Yxs"])
    ], dtype=float)

# =========================================================
# RELACIONES DE SETPOINT
# =========================================================
def compute_nominal_dilution(S_setpoint, p, model):
    return max(0.0, get_mu(S_setpoint, p, model))

def compute_target_biomass(S_setpoint, p):
    return p["Yxs"] * (p["Sr"] - S_setpoint)

# =========================================================
# CONTROL
# =========================================================
def control_law(error, integral_error, derivative_error, D_nominal, cp):
    if cp["type"] == "None":
        D = D_nominal
    elif cp["type"] == "P":
        D = D_nominal + cp["Kp"] * error
    elif cp["type"] == "PI":
        D = D_nominal + cp["Kp"] * error + cp["Ki"] * integral_error
    elif cp["type"] == "PID":
        D = D_nominal + cp["Kp"] * error + cp["Ki"] * integral_error + cp["Kd"] * derivative_error
    else:
        D = D_nominal

    return float(np.clip(D, cp["D_min"], cp["D_max"]))

# =========================================================
# RK4 CON CONTROL DIGITAL
# =========================================================
# Corrige la línea anterior si copiaste por fragmentos:
def rk4_one_step_control(y, D_applied, p, model, dt):
    k1 = dt * rhs(y, D_applied, p, model)
    k2 = dt * rhs(y + 0.5 * k1, D_applied, p, model)
    k3 = dt * rhs(y + 0.5 * k2, D_applied, p, model)
    k4 = dt * rhs(y + k3, D_applied, p, model)

    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    y_next = np.maximum(y_next, 0.0)

    return y_next, {
        "k1X": float(k1[0]), "k1S": float(k1[1]),
        "k2X": float(k2[0]), "k2S": float(k2[1]),
        "k3X": float(k3[0]), "k3S": float(k3[1]),
        "k4X": float(k4[0]), "k4S": float(k4[1]),
    }

def run_sim_control(y0, p, cp, model, S_target):
    dt = p["dt"]
    t_final = p["t_f"]
    n_steps = int(np.ceil(t_final / dt))

    y = np.array(y0, dtype=float)
    t = 0.0

    D_nominal = compute_nominal_dilution(S_target, p, model)

    integral_error = 0.0
    prev_error = None

    first_step_debug = None
    rows = []

    for i in range(n_steps + 1):
        X, S = float(y[0]), float(y[1])
        mu = float(get_mu(S, p, model))
        error = float(S - S_target)

        if prev_error is None:
            derivative_error = 0.0
        else:
            derivative_error = float((error - prev_error) / dt)

        D_applied = control_law(
            error=error,
            integral_error=integral_error,
            derivative_error=derivative_error,
            D_nominal=D_nominal,
            cp=cp,
        )

        rows.append({
            "t": float(t),
            "X": X,
            "S": S,
            "D": D_applied,
            "error": error,
            "mu": mu,
        })

        if i == n_steps:
            break

        y_new, debug = rk4_one_step_control(y, D_applied, p, model, dt)

        if first_step_debug is None:
            first_step_debug = {
                "X0": X,
                "S0": S,
                "dt": dt,
                "D0": D_applied,
                **debug,
                "X1": float(y_new[0]),
                "S1": float(y_new[1]),
            }

        # Actualizar memoria del controlador para el siguiente paso
        integral_error += error * dt
        prev_error = error

        y = y_new
        t += dt

    df = pd.DataFrame(rows)
    return df, D_nominal, first_step_debug

# =========================================================
# LINEALIZACIÓN LOCAL EN EL SETPOINT
# =========================================================
def normalize_eigenvectors(evecs):
    normalized = []
    for i in range(evecs.shape[1]):
        v = evecs[:, i].astype(complex)
        if abs(v[0]) > 1e-12:
            v = v / v[0]
        else:
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
        normalized.append(v)
    return normalized

def classify_eigenvalues(evals):
    re = np.real(evals)
    im = np.imag(evals)

    if np.any(re > 0) and np.any(re < 0):
        return "Inestable (silla)"
    if np.all(re < 0):
        if np.all(np.abs(im) < 1e-10):
            return "Estable (nodo)"
        return "Estable (foco/espiral)"
    if np.all(re > 0):
        if np.all(np.abs(im) < 1e-10):
            return "Inestable (nodo)"
        return "Inestable (foco/espiral)"
    return "Marginal / indeterminado"

def analyze_target_local(p, model, cp, S_target):
    X_target = compute_target_biomass(S_target, p)
    D_target = compute_nominal_dilution(S_target, p, model)

    feasible = (S_target >= 0) and (S_target <= p["Sr"]) and (X_target >= 0)
    if not feasible:
        return {
            "feasible": False,
            "message": "El setpoint elegido no es físicamente factible."
        }

    mu_p = get_dmu_dS(S_target, p, model)
    Yxs = p["Yxs"]
    Sr = p["Sr"]

    # Sin control o control P: sistema 2x2 exacto
    if cp["type"] in ["None", "P"]:
        dD_dS = 0.0 if cp["type"] == "None" else cp["Kp"]

        J = np.array([
            [0.0, X_target * (mu_p - dD_dS)],
            [-D_target / Yxs, dD_dS * (Sr - S_target) - D_target - (X_target * mu_p / Yxs)]
        ], dtype=float)

        evals, evecs = np.linalg.eig(J)

        return {
            "feasible": True,
            "exact_linearization": True,
            "dimension": 2,
            "X_target": float(X_target),
            "S_target": float(S_target),
            "D_target": float(D_target),
            "mu_prime": float(mu_p),
            "J": J,
            "detJ": float(np.linalg.det(J)),
            "evals": evals,
            "evecs": normalize_eigenvectors(evecs),
            "stable": bool(np.all(np.real(evals) < 0)),
            "classification": classify_eigenvalues(evals),
            "symbolic_latex": (
                r"J_{loc}=\begin{bmatrix}"
                r"0 & \hat X\left(\mu'(\hat S)-D'_S\right)\\"
                r"-\dfrac{D_s}{Y_{x/s}} & D'_S(S_r-\hat S)-D_s-\dfrac{\hat X\mu'(\hat S)}{Y_{x/s}}"
                r"\end{bmatrix}"
            ),
            "note": (
                "Aquí D'_S = 0 para 'None' y D'_S = K_p para control P."
            )
        }

    # Control PI: sistema aumentado 3x3 exacto
    if cp["type"] == "PI":
        Kp = cp["Kp"]
        Ki = cp["Ki"]

        J = np.array([
            [0.0, X_target * (mu_p - Kp), -X_target * Ki],
            [-D_target / Yxs,
             Kp * (Sr - S_target) - D_target - (X_target * mu_p / Yxs),
             Ki * (Sr - S_target)],
            [0.0, 1.0, 0.0]
        ], dtype=float)

        evals, evecs = np.linalg.eig(J)

        return {
            "feasible": True,
            "exact_linearization": True,
            "dimension": 3,
            "X_target": float(X_target),
            "S_target": float(S_target),
            "D_target": float(D_target),
            "mu_prime": float(mu_p),
            "J": J,
            "detJ": float(np.linalg.det(J)),
            "evals": evals,
            "evecs": normalize_eigenvectors(evecs),
            "stable": bool(np.all(np.real(evals) < 0)),
            "classification": classify_eigenvalues(evals),
            "symbolic_latex": (
                r"J_{loc}=\begin{bmatrix}"
                r"0 & \hat X\left(\mu'(\hat S)-K_p\right) & -\hat X K_i\\"
                r"-\dfrac{D_s}{Y_{x/s}} & K_p(S_r-\hat S)-D_s-\dfrac{\hat X\mu'(\hat S)}{Y_{x/s}} & K_i(S_r-\hat S)\\"
                r"0 & 1 & 0"
                r"\end{bmatrix}"
            ),
            "note": (
                "La tercera variable de estado corresponde al integrador del controlador, con İ = S - Ŝ."
            )
        }

    # PID: no dar jacobiana exacta sin filtro derivativo
    return {
        "feasible": True,
        "exact_linearization": False,
        "X_target": float(X_target),
        "S_target": float(S_target),
        "D_target": float(D_target),
        "message": (
            "Para PID ideal, la linealización continua exacta requiere modelar el término derivativo con más cuidado "
            "(por ejemplo, usando filtro derivativo). Aquí la evaluación principal se hace por simulación temporal."
        )
    }

# =========================================================
# VISUALIZACIÓN EN PLANO DE FASE
# =========================================================
def trajectory_tends_to_washout(sim_df, p):
    final_X = float(sim_df["X"].iloc[-1])
    final_S = float(sim_df["S"].iloc[-1])

    x_scale = max(1.0, float(sim_df["X"].max()))
    s_scale = max(1.0, float(p["Sr"]))

    x_tol = max(0.05 * x_scale, 0.05)
    s_tol = max(0.05 * s_scale, 0.25)

    return (final_X <= x_tol) and (abs(final_S - p["Sr"]) <= s_tol)

def add_phase_eigenvectors(fig, target_analysis):
    if not target_analysis.get("exact_linearization", False):
        return

    Xs = target_analysis["X_target"]
    Ss = target_analysis["S_target"]

    colors = ["#1565c0", "#8e24aa", "#ef6c00"]
    labels = ["v1", "v2", "v3"]

    x_scale = max(1.5, 0.16 * max(Xs, 1.0))
    s_scale = max(1.5, 0.16 * max(Ss, 1.0))

    max_vecs = min(2, len(target_analysis["evecs"]))  # en el plano X-S solo mostramos hasta 2
    for i in range(max_vecs):
        v = target_analysis["evecs"][i]
        vx = np.real(v[0])
        vs = np.real(v[1])

        x1 = Xs - x_scale * vx
        x2 = Xs + x_scale * vx
        y1 = Ss - s_scale * vs
        y2 = Ss + s_scale * vs

        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            name=f"Eigenvector {i+1}",
            line=dict(color=colors[i], width=2.4, dash="dash"),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[x2],
            y=[y2],
            mode="markers+text",
            text=[labels[i]],
            textposition="top right",
            marker=dict(size=8, color=colors[i], symbol="circle"),
            name=f"Etiqueta {labels[i]}",
            showlegend=False
        ))

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🧪 Configuración")
model_type = st.sidebar.selectbox("Modelo cinético", ["Monod", "Haldane"])
control_strategy = st.sidebar.selectbox("Estrategia de control", ["None", "P", "PI", "PID"])

st.sidebar.header("⚙️ Parámetros fisiológicos")
mu_max = slider_with_exact_input("μmax (h⁻¹)", "mu_max", 0.001, 2.000, 1.000, 0.001)
Ks = slider_with_exact_input("Ks (g/L)", "Ks", 0.001, 10.000, 1.000, 0.001)

if model_type == "Haldane":
    Ki = slider_with_exact_input("Ki (inhibición)", "Ki", 0.100, 100.000, 25.000, 0.001)
else:
    Ki = 25.0

Yxs = slider_with_exact_input("Rendimiento Yx/s", "Yxs", 0.001, 1.000, 0.500, 0.001)
Sr = slider_with_exact_input("Sr alimentación (g/L)", "Sr", 1.000, 200.000, 100.000, 0.001)

st.sidebar.header("🎯 Setpoint de sustrato")
if "S_target_slider" not in st.session_state:
    st.session_state["S_target_slider"] = min(23.960, Sr)
if "S_target_num" not in st.session_state:
    st.session_state["S_target_num"] = min(23.960, Sr)

st.session_state["S_target_slider"] = min(float(st.session_state["S_target_slider"]), float(Sr))
st.session_state["S_target_num"] = min(float(st.session_state["S_target_num"]), float(Sr))

S_target = slider_with_exact_input("S objetivo Ŝ (g/L)", "S_target", 0.001, float(Sr), min(23.960, float(Sr)), 0.001)
X_target = compute_target_biomass(S_target, {"Yxs": Yxs, "Sr": Sr})

st.sidebar.caption(
    f"X̂ es derivado, no independiente: X̂ = Yx/s · (Sr - Ŝ) = {X_target:.4f} g/L"
)

st.sidebar.header("🎮 Parámetros de control")
if control_strategy == "None":
    Kp, Ki_gain, Kd_gain = 0.0, 0.0, 0.0
    st.sidebar.info("Sin control: D(t) = Dₛ.")
elif control_strategy == "P":
    Kp = st.sidebar.number_input("Kp", value=-0.200, step=0.001, format="%.3f")
    Ki_gain, Kd_gain = 0.0, 0.0
elif control_strategy == "PI":
    Kp = st.sidebar.number_input("Kp", value=-0.200, step=0.001, format="%.3f")
    Ki_gain = st.sidebar.number_input("Ki", value=-0.010, step=0.001, format="%.3f")
    Kd_gain = 0.0
else:
    Kp = st.sidebar.number_input("Kp", value=-0.200, step=0.001, format="%.3f")
    Ki_gain = st.sidebar.number_input("Ki", value=-0.010, step=0.001, format="%.3f")
    Kd_gain = st.sidebar.number_input("Kd", value=-0.001, step=0.001, format="%.3f")

st.sidebar.caption(
    "Con e = S - Ŝ, ganancias negativas suelen tener sentido: si S > Ŝ, entonces D debe disminuir."
)

D_min = st.sidebar.number_input("D mínimo", min_value=0.0, value=0.0, step=0.001, format="%.3f")
D_max = st.sidebar.number_input("D máximo", min_value=0.01, value=2.0, step=0.001, format="%.3f")

st.sidebar.header("📍 Simulación")
x0 = st.sidebar.number_input("X inicial (g/L)", min_value=0.0, value=20.0, step=0.001, format="%.3f")
s0 = st.sidebar.number_input("S inicial (g/L)", min_value=0.0, value=45.0, step=0.001, format="%.3f")
dt = st.sidebar.number_input("Paso RK4 Δt (h)", min_value=0.001, value=0.010, step=0.001, format="%.3f")
t_f = st.sidebar.number_input("Tiempo final (h)", min_value=0.1, value=80.0, step=0.1, format="%.3f")

params = {
    "mu_max": mu_max,
    "Ks": Ks,
    "Ki": Ki,
    "Yxs": Yxs,
    "Sr": Sr,
    "dt": dt,
    "t_f": t_f,
    "model": model_type,
}

cp = {
    "type": control_strategy,
    "Kp": Kp,
    "Ki": Ki_gain,
    "Kd": Kd_gain,
    "D_min": D_min,
    "D_max": D_max,
}

# =========================================================
# EJECUCIÓN
# =========================================================
sim_df, D_nominal, first_step = run_sim_control([x0, s0], params, cp, model_type, S_target)
target_analysis = analyze_target_local(params, model_type, cp, S_target)

final_X = float(sim_df["X"].iloc[-1])
final_S = float(sim_df["S"].iloc[-1])
final_D = float(sim_df["D"].iloc[-1])
final_error = float(sim_df["error"].iloc[-1])

show_washout_on_phase = trajectory_tends_to_washout(sim_df, params)

# =========================================================
# RESUMEN RÁPIDO
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("S target Ŝ (g/L)", f"{S_target:.4f}")
m2.metric("X target X̂ (g/L)", f"{X_target:.4f}")
m3.metric("D nominal Dₛ (h⁻¹)", f"{D_nominal:.4f}")
m4.metric("D final (h⁻¹)", f"{final_D:.4f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("X final (g/L)", f"{final_X:.4f}")
m6.metric("S final (g/L)", f"{final_S:.4f}")
m7.metric("Error final e = S-Ŝ", f"{final_error:.4f}")
m8.metric("Pasos RK4", f"{len(sim_df) - 1}")

# =========================================================
# GRÁFICAS PRINCIPALES
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="kicker">Dynamics + control</span>', unsafe_allow_html=True)
    st.subheader("Dinámica temporal")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=sim_df["t"], y=sim_df["X"],
            name="Biomasa X",
            line=dict(color="#a6ce63", width=3)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sim_df["t"], y=sim_df["S"],
            name="Sustrato S",
            line=dict(color="#2a3557", width=3)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sim_df["t"], y=np.full(len(sim_df), S_target),
            name="Setpoint Ŝ",
            line=dict(color="#c62828", width=2, dash="dash")
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sim_df["t"], y=sim_df["D"],
            name="Dilución D(t)",
            line=dict(color="#ff9800", width=2)
        ),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=20, t=30, b=30),
        xaxis_title="Tiempo (h)",
    )
    fig.update_yaxes(title_text="X y S (g/L)", secondary_y=False)
    fig.update_yaxes(title_text="D (h⁻¹)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="kicker">Phase plane</span>', unsafe_allow_html=True)
    st.subheader("Plano de fase: X vs S")

    fig_p = go.Figure()

    fig_p.add_trace(go.Scatter(
        x=sim_df["X"], y=sim_df["S"],
        name="Trayectoria RK4",
        mode="lines",
        line=dict(color="#666666", dash="dot", width=3)
    ))

    fig_p.add_trace(go.Scatter(
        x=[sim_df["X"].iloc[0]],
        y=[sim_df["S"].iloc[0]],
        mode="markers",
        name="Inicio",
        marker=dict(size=11, color="#000000", symbol="circle")
    ))

    fig_p.add_trace(go.Scatter(
        x=[sim_df["X"].iloc[-1]],
        y=[sim_df["S"].iloc[-1]],
        mode="markers",
        name="Final",
        marker=dict(size=11, color="#a6ce63", symbol="circle")
    ))

    fig_p.add_trace(go.Scatter(
        x=[X_target],
        y=[S_target],
        mode="markers",
        name="Setpoint",
        marker=dict(size=13, color="red", symbol="diamond")
    ))

    if target_analysis.get("exact_linearization", False):
        add_phase_eigenvectors(fig_p, target_analysis)

    if show_washout_on_phase:
        fig_p.add_trace(go.Scatter(
            x=[0.0],
            y=[Sr],
            mode="markers",
            name="Washout",
            marker=dict(size=12, color="#c62828", symbol="x")
        ))

    x_max_plot = max(float(sim_df["X"].max()), X_target, 1.0) * 1.15
    y_max_plot = max(float(sim_df["S"].max()), S_target, Sr if show_washout_on_phase else 0.0, 1.0) * 1.15

    fig_p.update_layout(
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=20, t=30, b=30),
        xaxis_title="Biomasa X (g/L)",
        yaxis_title="Sustrato S (g/L)",
        xaxis=dict(range=[0, x_max_plot]),
        yaxis=dict(range=[0, y_max_plot]),
    )

    st.plotly_chart(fig_p, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ANÁLISIS LOCAL EN EL SETPOINT
# =========================================================
st.divider()
st.subheader("🔍 Análisis local en el setpoint")

a1, a2, a3 = st.columns([1.0, 1.2, 1.8])

with a1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("X̂", f"{X_target:.4f} g/L")
    st.metric("Ŝ", f"{S_target:.4f} g/L")
    st.metric("Dₛ", f"{D_nominal:.4f} h⁻¹")
    st.markdown('</div>', unsafe_allow_html=True)

with a2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if target_analysis.get("exact_linearization", False):
        evals = target_analysis["evals"]
        stable = target_analysis["stable"]
        st.write("**Eigenvalores**")
        st.code("\n".join([f"λ{i+1} = {ev:.6f}" for i, ev in enumerate(evals)]))
        st.write(f"**Determinante:** `{target_analysis['detJ']:.6f}`")
        st.info(f"Veredicto local: {'ESTABLE ✅' if stable else 'INESTABLE ❌'}")
        st.caption(f"Clasificación: {target_analysis['classification']}")
    else:
        st.write("**Análisis lineal**")
        st.warning(target_analysis["message"])
    st.markdown('</div>', unsafe_allow_html=True)

with a3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if target_analysis.get("exact_linearization", False):
        st.write("**Matriz Jacobiana local**")
        J = target_analysis["J"]
        J_df = pd.DataFrame(
            J,
            index=["dX/dt", "dS/dt"] if target_analysis["dimension"] == 2 else ["dX/dt", "dS/dt", "dI/dt"],
            columns=["∂/∂X", "∂/∂S"] if target_analysis["dimension"] == 2 else ["∂/∂X", "∂/∂S", "∂/∂I"],
        )
        st.dataframe(J_df.style.format("{:.6f}"), use_container_width=True)

        st.write("**Eigenvectores normalizados**")
        vec_rows = ["X", "S"] if target_analysis["dimension"] == 2 else ["X", "S", "I"]
        evec_table = {"Componente": vec_rows}
        for i, vec in enumerate(target_analysis["evecs"], start=1):
            evec_table[f"v{i}"] = [np.real(vec[j]) for j in range(len(vec_rows))]
        ev_df = pd.DataFrame(evec_table)
        style_map = {col: "{:.6f}" for col in ev_df.columns if col != "Componente"}
        st.dataframe(ev_df.style.format(style_map), use_container_width=True)
    else:
        st.write("**Nota**")
        st.write("Para PID, la estabilidad exacta requiere modelar explícitamente el derivativo.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# GUÍA DOCENTE
# =========================================================
st.divider()
with st.expander("📖 Guía de modelado y control (paso a paso)"):
    st.markdown("### 1. Definición del sistema: balances y funciones de estado")
    st.markdown("Definimos las dos ecuaciones de estado como:")
    st.latex(r"f_1(X,S,D)=\frac{dX}{dt}=X(\mu-D)")
    st.latex(r"f_2(X,S,D)=\frac{dS}{dt}=D(S_r-S)-\frac{\mu X}{Y_{x/s}}")

    st.markdown("### 2. Definición de la velocidad específica de crecimiento")
    if model_type == "Monod":
        st.latex(r"\mu(S)=\frac{\mu_{max}S}{K_s+S}")
        st.latex(r"\mu'(S)=\frac{\mu_{max}K_s}{(K_s+S)^2}")
    else:
        st.latex(r"\mu(S)=\frac{\mu_{max}S}{K_s+S+\frac{S^2}{K_i}}")
        st.latex(r"\mu'(S)=\frac{\mu_{max}\left(K_s-\frac{S^2}{K_i}\right)}{\left(K_s+S+\frac{S^2}{K_i}\right)^2}")

    st.markdown("### 3. Balances con μ sustituida")
    if model_type == "Monod":
        st.latex(r"\frac{dX}{dt}=X\left(\frac{\mu_{max}S}{K_s+S}-D\right)")
        st.latex(r"\frac{dS}{dt}=D(S_r-S)-\frac{X}{Y_{x/s}}\left(\frac{\mu_{max}S}{K_s+S}\right)")
    else:
        st.latex(r"\frac{dX}{dt}=X\left(\frac{\mu_{max}S}{K_s+S+\frac{S^2}{K_i}}-D\right)")
        st.latex(r"\frac{dS}{dt}=D(S_r-S)-\frac{X}{Y_{x/s}}\left(\frac{\mu_{max}S}{K_s+S+\frac{S^2}{K_i}}\right)")

    st.markdown("### 4. Setpoint de sustrato y biomasa asociada")
    st.markdown(
        "En esta app, el objetivo de control es **Ŝ**. "
        "La biomasa objetivo **X̂** no se fija de forma independiente; se deduce del balance de sustrato en estado estacionario."
    )
    st.latex(r"D_s=\mu(\hat S)")
    st.latex(r"\hat X=Y_{x/s}(S_r-\hat S)")
    st.markdown(
        f"- Ŝ = `{S_target:.6f}` g/L\n"
        f"- X̂ = `{X_target:.6f}` g/L\n"
        f"- Dₛ = `{D_nominal:.6f}` h⁻¹"
    )

    st.markdown("### 5. Error y ley de control")
    st.latex(r"e=S-\hat S")
    if control_strategy == "None":
        st.latex(r"D(t)=D_s")
    elif control_strategy == "P":
        st.latex(r"D(t)=D_s+K_p\,e")
    elif control_strategy == "PI":
        st.latex(r"D(t)=D_s+K_p\,e+K_i\int e\,dt")
    else:
        st.latex(r"D(t)=D_s+K_p\,e+K_i\int e\,dt+K_d\frac{de}{dt}")

    st.markdown(
        "Con esta convención de error, **ganancias negativas** suelen ser razonables: "
        "si \(S>\hat S\), entonces \(e>0\) y un \(K_p<0\) hace que \(D\) disminuya."
    )

    st.markdown("### 6. Jacobiana local, forma simbólica, valores numéricos y determinante")
    if target_analysis.get("exact_linearization", False):
        st.latex(target_analysis["symbolic_latex"])
        st.caption(target_analysis["note"])

        J = target_analysis["J"]
        st.latex(matrix_to_latex(J, "J_{num}"))
        st.latex(r"\det(J)=" + f"{target_analysis['detJ']:.6f}")
    else:
        st.warning(target_analysis["message"])

    st.markdown("### 7. Primer paso RK4")
    if first_step is not None:
        st.markdown(
            f"Se aplicó inicialmente una dilución `D0 = {first_step['D0']:.6f}` con `Δt = {first_step['dt']:.6f}`."
        )
        rk_df = pd.DataFrame({
            "Pendiente": ["k1", "k2", "k3", "k4"],
            "X": [first_step["k1X"], first_step["k2X"], first_step["k3X"], first_step["k4X"]],
            "S": [first_step["k1S"], first_step["k2S"], first_step["k3S"], first_step["k4S"]],
        })
        st.dataframe(rk_df.style.format({"X": "{:.6f}", "S": "{:.6f}"}), use_container_width=True)
        st.markdown(
            f"**Resultado del primer paso:** X₁ = `{first_step['X1']:.6f}`, "
            f"S₁ = `{first_step['S1']:.6f}`"
        )

# =========================================================
# TABLA TEMPORAL
# =========================================================
st.divider()
with st.expander("📈 Ver tabla temporal RK4 + control"):
    st.dataframe(
        sim_df.style.format({
            "t": "{:.4f}",
            "X": "{:.6f}",
            "S": "{:.6f}",
            "D": "{:.6f}",
            "error": "{:.6f}",
            "mu": "{:.6f}",
        }),
        use_container_width=True,
        height=340,
    )