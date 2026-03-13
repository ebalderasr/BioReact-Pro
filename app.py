import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import fsolve

# =========================================================
# CONFIGURACIÓN Y ESTILOS (Host Cell Lab Suite)
# =========================================================
st.set_page_config(page_title="BioReact Engine", layout="wide")

st.markdown("""
<style>
:root { --accent: #a6ce63; --text: #f5f7fb; --muted: #aab5c8; --border: rgba(166, 206, 99, 0.2); }
.host-cell-header { display: flex; align-items: center; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--border); margin-bottom: 18px; }
.app-icon-container { width: 52px; height: 52px; background: #1a2544; border-radius: 12px; display: flex; align-items: center; justify-content: center; }
.app-icon-text { color: var(--accent); font-weight: 900; font-size: 20px; }
.author-highlight { color: var(--accent); font-weight: 700; }
.metric-card { background: rgba(17,24,39,0.9); border: 1px solid var(--border); border-radius: 12px; padding: 12px; margin-bottom: 10px; }
</style>
<div class="host-cell-header">
    <div class="app-icon-container"><span class="app-icon-text">BR</span></div>
    <div class="brand-content">
        <h1 style='margin:0; font-size:1.7rem;'>BioReact</h1>
        <p style='margin:0; font-size:0.88rem; color:#aab5c8;'>
            Host Cell Lab Suite • <span class="author-highlight">Emiliano Balderas Ramírez</span>, Bioengineer PhD student
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# UTILIDADES DE UI (3 Decimales Sincronizados)
# =========================================================
def _sync_from_slider(name): st.session_state[f"{name}_n"] = st.session_state[f"{name}_s"]
def _sync_from_num(name, min_v, max_v):
    val = max(min_v, min(max_v, float(st.session_state[f"{name}_n"])))
    st.session_state[f"{name}_n"] = val
    st.session_state[f"{name}_s"] = val

def synced_input(label, name, min_v, max_v, default, step=0.001):
    if f"{name}_n" not in st.session_state: st.session_state[f"{name}_n"] = default
    c1, c2 = st.sidebar.columns([3, 1.55])
    c1.slider(label, min_v, max_v, step=step, key=f"{name}_s", value=st.session_state[f"{name}_n"], on_change=_sync_from_slider, args=(name,))
    c2.number_input("", min_v, max_v, step=step, format="%.3f", key=f"{name}_n", on_change=_sync_from_num, args=(name, min_v, max_v), label_visibility="collapsed")
    return st.session_state[f"{name}_n"]

# =========================================================
# MOTOR MATEMÁTICO (Balances X, S)
# =========================================================
def get_mu(S, p, model):
    S = max(float(S), 0.0)
    if model == "Monod":
        return (p['mu_max'] * S) / (p['Ks'] + S)
    else: # Haldane
        return (p['mu_max'] * S) / (p['Ks'] + S + (S**2 / p['Ki']))

def get_dmu_dS(S, p, model):
    S = max(float(S), 1e-9)
    if model == "Monod":
        return (p['mu_max'] * p['Ks']) / (p['Ks'] + S)**2
    else: # Haldane
        num = p['mu_max'] * (p['Ks'] - (S**2 / p['Ki']))
        den = (p['Ks'] + S + (S**2 / p['Ki']))**2
        return num / den

def f_system(y, p, model):
    X, S = y
    mu = get_mu(S, p, model)
    dX = X * (mu - p['D'])
    dS = p['D']*(p['Sr'] - S) - (mu * X / p['Yxs'])
    return np.array([dX, dS])

def run_rk4(y0, p, model):
    dt, steps = p['dt'], int(p['t_f'] / p['dt'])
    y = np.array(y0, dtype=float)
    data = []
    for i in range(steps):
        t = i * dt
        k1 = dt * f_system(y, p, model)
        k2 = dt * f_system(y + 0.5*k1, p, model)
        k3 = dt * f_system(y + 0.5*k2, p, model)
        k4 = dt * f_system(y + k3, p, model)
        y = np.maximum(y + (k1 + 2*k2 + 2*k3 + k4)/6, 0)
        data.append([t, y[0], y[1]])
    return pd.DataFrame(data, columns=['t', 'X', 'S'])

# =========================================================
# ESTADOS ESTACIONARIOS Y EIGENVECTORES
# =========================================================
def find_steady_states(p, model):
    states = [{"name": "Washout", "X": 0.0, "S": p['Sr']}]
    
    if model == "Monod":
        if p['mu_max'] > p['D']:
            s_star = (p['D'] * p['Ks']) / (p['mu_max'] - p['D'])
            x_star = p['Yxs'] * (p['Sr'] - s_star)
            if 0 < s_star < p['Sr']:
                states.append({"name": "Steady State", "X": x_star, "S": s_star})
    else: # Haldane (Solución cuadrática de mu = D)
        a = p['D'] / p['Ki']
        b = p['D'] - p['mu_max']
        c = p['D'] * p['Ks']
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            roots = [(-b - np.sqrt(discriminant)) / (2*a), (-b + np.sqrt(discriminant)) / (2*a)]
            for i, s_sol in enumerate(roots):
                x_sol = p['Yxs'] * (p['Sr'] - s_sol)
                if 0 < s_sol < p['Sr'] and x_sol > 0:
                    states.append({"name": f"Punto Crítico {i+1}", "X": x_sol, "S": s_sol})
    return states

def analyze_stability(ss, p, model):
    mu = get_mu(ss['S'], p, model)
    dmu = get_dmu_dS(ss['S'], p, model)
    # Jacobiana 2x2
    J = np.array([
        [mu - p['D'], ss['X'] * dmu],
        [-mu / p['Yxs'], -p['D'] - (ss['X'] * dmu / p['Yxs'])]
    ])
    evals, evecs = np.linalg.eig(J)
    return J, evals, evecs

# =========================================================
# INTERFAZ (SIDEBAR)
# =========================================================
st.sidebar.header("🧪 Cinética")
model_type = st.sidebar.selectbox("Modelo", ["Monod", "Haldane"])

st.sidebar.header("⚙️ Parámetros")
mu_max = synced_input("μmax (h⁻¹)", "mu_max", 0.1, 2.0, 1.0)
Ks = synced_input("Ks (g/L)", "Ks", 0.1, 10.0, 1.0)
Ki = synced_input("Ki (Solo Haldane)", "Ki", 1.0, 100.0, 25.0)
Yxs = synced_input("Yx/s", "Yxs", 0.1, 1.0, 0.5)
Sr = synced_input("Sr (g/L)", "Sr", 10.0, 200.0, 100.0)
D = synced_input("D (h⁻¹)", "D", 0.01, 1.5, 0.5)

st.sidebar.header("📍 Simulación")
x0 = st.sidebar.number_input("X0", value=0.200, format="%.3f")
s0 = st.sidebar.number_input("S0", value=15.000, format="%.3f")
t_f = st.sidebar.number_input("Tiempo (h)", value=80)
params = {'mu_max': mu_max, 'Ks': Ks, 'Ki': Ki, 'Yxs': Yxs, 'Sr': Sr, 'D': D, 't_f': t_f, 'dt': 0.01}

# CÁLCULOS
df = run_rk4([x0, s0], params, model_type)
ss_list = find_steady_states(params, model_type)

# GRÁFICAS
c1, c2 = st.columns(2)
with c1:
    st.subheader("Dinámica Temporal")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df['t'], y=df['X'], name="X (Biomasa)", line=dict(color='#a6ce63', width=3)))
    fig_t.add_trace(go.Scatter(x=df['t'], y=df['S'], name="S (Sustrato)", line=dict(color='#2a3557')))
    fig_t.update_layout(template="plotly_white", height=450, xaxis_title="Tiempo (h)", yaxis_title="g/L")
    st.plotly_chart(fig_t, use_container_width=True)

with c2:
    st.subheader("Plano de Fase e Invariantes")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df['X'], y=df['S'], name="Trayectoria", line=dict(color='#666', dash='dot')))
    
    for ss in ss_list:
        J, evals, evecs = analyze_stability(ss, params, model_type)
        stable = all(np.real(evals) < 0)
        color = '#34c759' if stable else '#ff3b30'
        
        # Punto Crítico
        fig_p.add_trace(go.Scatter(x=[ss['X']], y=[ss['S']], mode='markers', name=ss['name'],
                                  marker=dict(size=12, color=color, symbol='diamond')))
        
        # Trazo de Eigenvectores (Normalizados para visualización)
        for i in range(2):
            v = np.real(evecs[:, i])
            scale = 10 # Escala de visualización
            fig_p.add_trace(go.Scatter(x=[ss['X'] - v[0]*scale, ss['X'] + v[0]*scale],
                                      y=[ss['S'] - v[1]*scale, ss['S'] + v[1]*scale],
                                      mode='lines', line=dict(color='blue', width=1, dash='dash'),
                                      showlegend=False))

    fig_p.update_layout(template="plotly_white", height=450, xaxis_title="X", yaxis_title="S")
    st.plotly_chart(fig_p, use_container_width=True)

# TABLA DE ESTABILIDAD
st.divider()
st.subheader("🔍 Análisis de Estabilidad")
tabs = st.tabs([ss['name'] for ss in ss_list])
for i, tab in enumerate(tabs):
    with tab:
        ss = ss_list[i]
        J, evals, evecs = analyze_stability(ss, params, model_type)
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Coordenadas:** X={ss['X']:.4f}, S={ss['S']:.4f}")
            st.write("**Eigenvalores:**")
            st.code(f"λ1: {evals[0]:.4f}\nλ2: {evals[1]:.4f}")
        with col_b:
            st.write("**Matriz Jacobiana:**")
            st.table(pd.DataFrame(J, columns=['∂/∂X', '∂/∂S'], index=['f1', 'f2']))

# GUÍA DOCENTE
with st.expander("📖 Fundamentos (Monod vs Haldane)"):
    st.markdown("### Balances de Masa")
    st.latex(r"\frac{dX}{dt} = X(\mu - D), \quad \frac{dS}{dt} = D(S_r - S) - \frac{\mu X}{Y_{x/s}}")
    if model_type == "Monod":
        st.latex(r"\mu(S) = \frac{\mu_{max} S}{K_s + S}")
    else:
        st.latex(r"\mu(S) = \frac{\mu_{max} S}{K_s + S + S^2/K_i}")
        st.write("En Haldane, al resolver $\mu = D$ se obtiene una ecuación cuadrática. Esto permite la **multiplicidad de estados**, donde un punto puede ser un nodo estable y el otro un punto de silla (inestable).")