import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit.components.v1 as components
import os

# --- Configurações da Página ---
st.set_page_config(
    page_title="Titulação Ácido-Base — 11.º Ano",
    layout="wide",
    page_icon="🧪"
)

# Importação do novo componente HTML local (Bureta com Animação)
_parent_dir = os.path.dirname(os.path.abspath(__file__))
_comp_dir = os.path.join(_parent_dir, "bureta_component")
bureta_ui = components.declare_component("bureta_ui", path=_comp_dir)

# --- Constantes ---
KW = 1.0e-14

INDICADORES = {
    "Fenolftaleína": {
        "ph_min": 8.2,
        "ph_max": 10.0,
        "cor_acida": "Incolor",
        "cor_basica": "Rosa carmim",
        "rgb_acido": (0.55, 0.70, 1.0, 0.2),
        "rgb_transicao": (1.0, 0.75, 0.85, 0.6),
        "rgb_basico": (0.90, 0.21, 0.47, 0.85),
    },
    "Vermelho de Metilo": {
        "ph_min": 4.4,
        "ph_max": 6.2,
        "cor_acida": "Vermelho",
        "cor_basica": "Amarelo",
        "rgb_acido": (0.95, 0.1, 0.1, 0.85),
        "rgb_transicao": (1.0, 0.55, 0.0, 0.85),
        "rgb_basico": (0.95, 0.95, 0.0, 0.85),
    },
    "Azul de Bromotimol": {
        "ph_min": 6.0,
        "ph_max": 7.6,
        "cor_acida": "Amarelo",
        "cor_basica": "Azul",
        "rgb_acido": (0.95, 0.95, 0.0, 0.85),
        "rgb_transicao": (0.0, 0.75, 0.0, 0.85),
        "rgb_basico": (0.0, 0.25, 0.85, 0.85),
    },
}

ACIDOS = {
    "HCl (Ácido Clorídrico) — forte":        {"tipo": "forte", "ka": None,    "nome_curto": "HCl"},
    "HNO₃ (Ácido Nítrico) — forte":          {"tipo": "forte", "ka": None,    "nome_curto": "HNO₃"},
    "CH₃COOH (Ácido Acético) — fraco":       {"tipo": "fraco", "ka": 1.76e-5, "nome_curto": "CH₃COOH"},
    "HCOOH (Ácido Fórmico) — fraco":         {"tipo": "fraco", "ka": 1.77e-4, "nome_curto": "HCOOH"},
    "C₆H₅COOH (Ácido Benzóico) — fraco":     {"tipo": "fraco", "ka": 6.28e-5, "nome_curto": "C₆H₅COOH"},
}

BASES = {
    "NaOH (Hidróxido de Sódio) — forte":     {"tipo": "forte", "kb": None,    "nome_curto": "NaOH"},
    "KOH (Hidróxido de Potássio) — forte":   {"tipo": "forte", "kb": None,    "nome_curto": "KOH"},
    "NH₃ (Amoníaco) — fraca":                {"tipo": "fraco", "kb": 1.77e-5, "nome_curto": "NH₃"},
    "CH₃NH₂ (Metilamina) — fraca":           {"tipo": "fraco", "kb": 4.47e-4, "nome_curto": "CH₃NH₂"},
    "C₆H₅NH₂ (Anilina) — fraca":             {"tipo": "fraco", "kb": 3.98e-10,"nome_curto": "C₆H₅NH₂"},
}

def calcular_ph_pe(tipo_a, ka, tipo_b, kb, c_sal):
    """
    Calcula o pH teórico no ponto de equivalência (hidrólise do sal).
    c_sal: concentração do sal formado no P.Eq. (≈ c/2 para concentrações iguais).
    """
    if tipo_a == "forte" and tipo_b == "forte":
        return 7.0
    elif tipo_a == "fraco" and tipo_b == "forte":
        # Sal de ácido fraco + base forte → solução básica (hidrólise do anião)
        # A⁻ + H₂O ⇌ HA + OH⁻ ; Kh = Kw / Ka
        if ka is None or ka <= 0 or c_sal <= 0:
            return 7.0
        kh = KW / ka
        ph = 14 + 0.5 * np.log10(kh * c_sal)
        return float(np.clip(ph, 0, 14))
    elif tipo_a == "forte" and tipo_b == "fraco":
        # Sal de ácido forte + base fraca → solução ácida (hidrólise do catião)
        # BH⁺ + H₂O ⇌ B + H₃O⁺ ; Ka_conj = Kw / Kb
        if kb is None or kb <= 0 or c_sal <= 0:
            return 7.0
        ka_conj = KW / kb
        ph = -0.5 * np.log10(ka_conj * c_sal)
        return float(np.clip(ph, 0, 14))
    else:
        # Ácido fraco + base fraca: pH ≈ 7 + 0.5·pKa − 0.5·pKb  (aproximação)
        if ka is None or kb is None:
            return 7.0
        ph = 7.0 + 0.5 * np.log10(ka / (KW / kb))
        return float(np.clip(ph, 0, 14))


# ── Cálculo de pH (bissecção — sem scipy) ──────────────────────────────────
def _balanco(H, ca_t, ka, tipo_a, cb_t, kb, tipo_b):
    OH = KW / max(H, 1e-30)
    if tipo_a == "forte":
        A_minus = ca_t
    else:
        A_minus = (ka * ca_t) / (ka + H) if ca_t > 0 else 0.0
    if tipo_b == "forte":
        B_plus = cb_t
    else:
        B_plus = (kb * cb_t) / (kb + OH) if cb_t > 0 else 0.0
    return H + B_plus - OH - A_minus


def calcular_ph(va, ca, ka, tipo_a, vb, cb, kb, tipo_b):
    vt = va + vb
    if vt == 0:
        return 7.0
    ca_t = (ca * va) / vt
    cb_t = (cb * vb) / vt

    lo, hi = 1e-14, 10.0
    f_lo = _balanco(lo, ca_t, ka, tipo_a, cb_t, kb, tipo_b)
    f_hi = _balanco(hi, ca_t, ka, tipo_a, cb_t, kb, tipo_b)

    if f_lo * f_hi > 0:
        return 7.0

    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = _balanco(mid, ca_t, ka, tipo_a, cb_t, kb, tipo_b)
        if abs(f_mid) < 1e-18 or (hi - lo) < 1e-18:
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    H_final = (lo + hi) / 2
    return float(np.clip(-np.log10(max(H_final, 1e-15)), 0, 14))


def obter_cor(ph, ind_nome):
    ind = INDICADORES[ind_nome]
    if ph < ind["ph_min"]:
        return ind["cor_acida"], ind["rgb_acido"]
    elif ph > ind["ph_max"]:
        return ind["cor_basica"], ind["rgb_basico"]
    else:
        return "Zona de viragem", ind["rgb_transicao"]


# ── Session state ──────────────────────────────────────────────────────────
if "modo_app" not in st.session_state:
    st.session_state.modo_app = "Modo Automático"
if "fase_pratica" not in st.session_state:
    st.session_state.fase_pratica = 1
if "passo_proc" not in st.session_state:
    st.session_state.passo_proc = 1
if "dados" not in st.session_state:
    st.session_state.dados = []
if "dados_praticos" not in st.session_state:
    st.session_state.dados_praticos = pd.DataFrame([{"V (mL)": None, "pH": None}])
if "anim_passo" not in st.session_state:
    st.session_state.anim_passo = 0  # 0=idle, 1=playing step1, 2=playing step2
if "vol_add" not in st.session_state:
    st.session_state.vol_add = 0.0
if "config" not in st.session_state:
    st.session_state.config = {}

def resetar():
    st.session_state.dados = []
    st.session_state.dados_praticos = pd.DataFrame([{"V (mL)": None, "pH": None}])
    st.session_state.vol_add = 0.0
    st.session_state.fase_pratica = 1
    st.session_state.passo_proc = 1
    st.session_state.anim_passo = 0
    st.session_state.auto_started = False

def verificar_config(nova):
    if st.session_state.config != nova:
        resetar()
        st.session_state.config = dict(nova)

# ── CSS customizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0e1117; }

.metric-box {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-label { font-size: 13px; color: #8899aa; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value { font-size: 48px; font-weight: 700; color: #e0f0ff; margin-top: 4px; line-height: 1; }
.metric-unit  { font-size: 14px; color: #7799bb; margin-top: 2px; }

/* Display digital para pH */
.ph-meter {
    background: #2a2d3e;
    border: 3px solid #1e2233;
    border-radius: 8px;
    padding: 10px 20px;
    display: inline-block;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    margin: 15px 0;
}
.ph-meter-label { font-size: 11px; color: #778899; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.ph-meter-display {
    font-family: 'Courier New', monospace;
    font-size: 38px;
    font-weight: bold;
    color: #4df07a;
    text-shadow: 0 0 5px rgba(77, 240, 122, 0.5);
    background: #11141c;
    padding: 5px 15px;
    border-radius: 4px;
    min-width: 120px;
    text-align: right;
}

.copo-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 16px 0;
}
.copo {
    width: 120px;
    height: 160px;
    border: 3px solid #aaa;
    border-top: none;
    border-radius: 0 0 30px 30px;
    position: relative;
    overflow: hidden;
}
.liquido {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 80%;
    border-radius: 0 0 27px 27px;
    transition: background-color 0.6s ease;
}
.copo-label { margin-top: 8px; font-size: 12px; color: #8899aa; }

.btn-row { display: flex; gap: 8px; margin-top: 4px; }

.info-card {
    background: #1a1d2e;
    border-left: 4px solid #3a7bd5;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #cdd6e0;
}

section[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #1e2535;
}

/* Styling the data editor specifically for manual entry */
[data-testid="stDataFrame"] {
    background: #161925;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Modo de Operação")
    modo_app = st.radio("Escolha o modo:", ["Modo Automático", "Modo Prático-Experimental"], label_visibility="collapsed")
    if modo_app != st.session_state.modo_app:
        st.session_state.modo_app = modo_app
        resetar()
        st.rerun()

    st.markdown("---")
    st.markdown("## ⚗️ Configurar Titulação")

    modo = st.radio("**Titulado no copo:**", ["Ácido → titulante é a Base", "Base → titulante é o Ácido"])
    acido_base = "Ácido" if "Ácido" in modo.split("→")[0] else "Base"

    if acido_base == "Ácido":
        titulado_nome = st.selectbox("🧪 Ácido (Titulado)", list(ACIDOS.keys()))
        titulante_nome = st.selectbox("🧫 Base (Titulante — na bureta)", list(BASES.keys()))
    else:
        titulado_nome = st.selectbox("🧪 Base (Titulado)", list(BASES.keys()))
        titulante_nome = st.selectbox("🧫 Ácido (Titulante — na bureta)", list(ACIDOS.keys()))

    c_titulado = st.number_input("Concentração do Titulado (mol/dm³)", 0.001, 2.0, 0.100, format="%.3f")
    v_titulado = st.number_input("Volume do Titulado (mL)", 5.0, 100.0, 20.0, step=1.0)
    c_titulante = st.number_input("Concentração do Titulante (mol/dm³)", 0.001, 2.0, 0.100, format="%.3f")

    indicador = st.selectbox("🌡️ Indicador Ácido-Base", list(INDICADORES.keys()))

    nova_config = dict(
        modo=modo, titulado=titulado_nome, titulante=titulante_nome,
        c_titulado=c_titulado, v_titulado=v_titulado,
        c_titulante=c_titulante, indicador=indicador
    )
    verificar_config(nova_config)

    v_eq_teorico = (c_titulado * v_titulado) / c_titulante
    # Concentration of salt at EP = moles / total volume at EP
    v_total_pe = v_titulado + v_eq_teorico  # mL
    c_sal_pe = (c_titulado * v_titulado) / v_total_pe  # mol/dm³ (mL cancels)
    # Determine ka/kb for EP pH
    _ka_pe = ACIDOS[titulado_nome if acido_base == "Ácido" else titulante_nome]["ka"]
    _kb_pe = BASES[titulante_nome if acido_base == "Ácido" else titulado_nome]["kb"]
    _tipo_a_pe = ACIDOS[titulado_nome if acido_base == "Ácido" else titulante_nome]["tipo"]
    _tipo_b_pe = BASES[titulante_nome if acido_base == "Ácido" else titulado_nome]["tipo"]
    ph_pe = calcular_ph_pe(_tipo_a_pe, _ka_pe, _tipo_b_pe, _kb_pe, c_sal_pe)

    st.markdown("---")
    st.markdown(f"**Volume de equivalência teórico:** `{v_eq_teorico:.2f} mL`")
    st.markdown(f"**pH no ponto de equivalência:** `{ph_pe:.2f}`")
    st.caption("Calculado por hidrólise do sal formado na neutralização (25 °C)")

    if st.button("🔄 Reiniciar Titulação", width="stretch"):
        resetar()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULOS GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════
if acido_base == "Ácido":
    va = v_titulado
    ca = c_titulado
    tipo_a = ACIDOS[titulado_nome]["tipo"]
    ka = ACIDOS[titulado_nome]["ka"] or 1e-20
    vb = st.session_state.vol_add
    cb = c_titulante
    tipo_b = BASES[titulante_nome]["tipo"]
    kb = BASES[titulante_nome]["kb"] or 1e-20
else:
    vb = v_titulado
    cb = c_titulado
    tipo_b = BASES[titulado_nome]["tipo"]
    kb = BASES[titulado_nome]["kb"] or 1e-20
    va = st.session_state.vol_add
    ca = c_titulante
    tipo_a = ACIDOS[titulante_nome]["tipo"]
    ka = ACIDOS[titulante_nome]["ka"] or 1e-20

ph_atual = calcular_ph(va, ca, ka, tipo_a, vb, cb, kb, tipo_b)
cor_nome, cor_rgba = obter_cor(ph_atual, indicador)
cor_css = f"rgba({int(cor_rgba[0]*255)}, {int(cor_rgba[1]*255)}, {int(cor_rgba[2]*255)}, {cor_rgba[3]})"
cor_texto = f"rgb({int(cor_rgba[0]*255)}, {int(cor_rgba[1]*255)}, {int(cor_rgba[2]*255)})"

if indicador == "Fenolftaleína":
    near_ep = 2.7 <= ph_atual <= INDICADORES[indicador]["ph_min"]
else:
    near_ep = False

# ═══════════════════════════════════════════════════════════════════════════
# RENDERIZAÇÃO DO MODO ESCOLHIDO
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.modo_app == "Modo Automático":
    st.markdown("## 🧪 Simulador de Titulação Ácido-Base")
    st.markdown("**Modo Automático** | Atividade Laboratorial 3 — 11.º Ano")
    st.markdown("---")

    # Pre-calculate full curve
    v_max = v_eq_teorico * 2.5
    n_pts = 120
    vols_auto = np.linspace(0, v_max, n_pts)
    dados_auto = []
    for v in vols_auto:
        if acido_base == "Ácido":
            p = calcular_ph(v_titulado, c_titulado, ka, tipo_a, v, c_titulante, kb, tipo_b)
        else:
            p = calcular_ph(v, c_titulante, ka, tipo_a, v_titulado, c_titulado, kb, tipo_b)
        cn, _ = obter_cor(p, indicador)
        dados_auto.append({"V (mL)": round(float(v), 2), "pH": round(p, 2), "Cor da Solução": cn})

    df_auto = pd.DataFrame(dados_auto)

    col_anim, col_grafico = st.columns([1, 1.8], gap="large")

    if "auto_started" not in st.session_state:
        st.session_state.auto_started = False
    if "auto_step" not in st.session_state:
        st.session_state.auto_step = 1

    # Shared state for current point
    n_mostrar = st.session_state.auto_step if st.session_state.auto_started else 1
    row = dados_auto[n_mostrar - 1]
    ph_show = row["pH"]
    _, rgba_show = obter_cor(ph_show, indicador)
    css_show = f"rgba({int(rgba_show[0]*255)},{int(rgba_show[1]*255)},{int(rgba_show[2]*255)},{rgba_show[3]})"
    txt_show = f"rgb({int(rgba_show[0]*255)},{int(rgba_show[1]*255)},{int(rgba_show[2]*255)})"

    # Burette liquid level: full at step 1, empty at last step
    bureta_fill_pct = 1.0 - (n_mostrar - 1) / (n_pts - 1)  # 1 → 0
    bureta_liq_height = int(bureta_fill_pct * 185)           # max 185px inside tube
    bureta_liq_y = 18 + (185 - bureta_liq_height)           # top of liquid rect

    # Beaker liquid: ~40% of beaker height always, colour changes
    drop_anim = "drop-anim 0.8s ease-in infinite" if st.session_state.auto_started else "none"

    with col_anim:
        st.markdown("### ⚗️ Bancada Virtual")

        if not st.session_state.auto_started:
            st.info("Pronto para iniciar a titulação automática.")
            if st.button("▶️ Iniciar Titulação", type="primary"):
                st.session_state.auto_started = True
                st.session_state.auto_step = 1
                st.rerun()
        else:
            n_mostrar = st.slider(
                "Volume adicionado (arrasta para simular o tempo):",
                1, n_pts, st.session_state.auto_step,
                label_visibility="visible"
            )
            st.session_state.auto_step = n_mostrar

            # Recalculate visuals after slider update
            row = dados_auto[n_mostrar - 1]
            ph_show = row["pH"]
            _, rgba_show = obter_cor(ph_show, indicador)
            css_show = f"rgba({int(rgba_show[0]*255)},{int(rgba_show[1]*255)},{int(rgba_show[2]*255)},{rgba_show[3]})"
            txt_show = f"rgb({int(rgba_show[0]*255)},{int(rgba_show[1]*255)},{int(rgba_show[2]*255)})"
            bureta_fill_pct = 1.0 - (n_mostrar - 1) / (n_pts - 1)
            bureta_liq_height = int(bureta_fill_pct * 185)
            bureta_liq_y = 18 + (185 - bureta_liq_height)

        # pH display
        st.markdown(f"""
        <div style='background:#1a1d2e;border:1px solid #2a3a5c;border-radius:12px;padding:14px 20px;text-align:center;margin-top:8px;'>
            <div style='display:flex;justify-content:space-around;align-items:center;'>
                <div>
                    <div style='font-size:11px;color:#8899aa;text-transform:uppercase;letter-spacing:1px;'>Volume</div>
                    <div style='font-size:26px;font-weight:700;color:#e0f0ff;'>{df_auto['V (mL)'].iloc[n_mostrar-1]:.2f} <span style='font-size:13px;color:#7799bb;'>mL</span></div>
                </div>
                <div>
                    <div style='font-size:11px;color:#8899aa;text-transform:uppercase;letter-spacing:1px;'>pH</div>
                    <div style='font-family:Courier New,monospace;font-size:32px;font-weight:bold;color:#4df07a;text-shadow:0 0 8px rgba(77,240,122,0.4);'>{ph_show:.2f}</div>
                </div>
                <div style='width:44px;height:44px;border-radius:50%;background:{css_show};border:2px solid #aabcd0;box-shadow:0 0 10px {css_show};'></div>
            </div>
            <div style='margin-top:6px;font-size:12px;color:#8899aa;'>Cor: <strong style='color:{txt_show};'>{row['Cor da Solução']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Equivalence point alert
        v_atual = df_auto["V (mL)"].iloc[n_mostrar - 1]
        if st.session_state.auto_started and v_atual >= v_eq_teorico:
            # Type of EP: determine colour coding
            is_neutral = _tipo_a_pe == "forte" and _tipo_b_pe == "forte"
            ep_color = "#4df07a" if not is_neutral else "#4da3ff"
            ep_bg = "linear-gradient(135deg,#1a2e1a,#0e2410)" if not is_neutral else "linear-gradient(135deg,#0e1a2e,#081222)"
            ep_border = ep_color
            ep_label = (
                "Solução neutra (sal de ácido forte + base forte)" if is_neutral
                else "Solução básica — hidrólise do anião" if _tipo_a_pe == "fraco"
                else "Solução ácida — hidrólise do catião"
            )
            st.markdown(f"""
            <div style='
                margin-top:12px;
                background: {ep_bg};
                border: 2px solid {ep_border};
                border-radius:12px;
                padding:14px 18px;
                text-align:center;
                box-shadow: 0 0 18px rgba(77,240,122,0.35);
                animation: pulse-ep 2s ease-in-out infinite;
            '>
                <div style='font-size:18px;'>🎯 <strong style='color:{ep_color};'>Ponto de Equivalência Atingido!</strong></div>
                <div style='margin-top:8px;font-size:13px;color:#aaccaa;'>Volume teórico do P.Eq.:</div>
                <div style='font-size:28px;font-weight:800;color:#ffffff;margin-top:2px;'>
                    {v_eq_teorico:.2f} <span style='font-size:15px;color:#7799bb;'>mL</span>
                </div>
                <div style='margin-top:8px;font-size:13px;color:#aaccaa;'>pH no P.Eq. (hidrólise do sal, 25 °C):</div>
                <div style='font-family:Courier New,monospace;font-size:26px;font-weight:bold;color:{ep_color};margin-top:2px;'>
                    {ph_pe:.2f}
                </div>
                <div style='margin-top:6px;font-size:11px;color:#8899aa;'>{ep_label}</div>
            </div>
            <style>
            @keyframes pulse-ep {{
                0%, 100% {{ box-shadow: 0 0 18px rgba(77,240,122,0.35); }}
                50%       {{ box-shadow: 0 0 30px rgba(77,240,122,0.7); }}
            }}
            </style>
            """, unsafe_allow_html=True)

        # Animated SVG: burette + beaker
        components.html(f"""
        <!DOCTYPE html><html><body style="margin:0;padding:10px 0 0 0;background:transparent;overflow:hidden;">
        <div style="display:flex;justify-content:center;align-items:flex-start;width:100%;">
        <svg width="220" height="380" viewBox="0 0 220 380" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="bureta-clip">
              <rect x="77" y="18" width="26" height="185"/>
            </clipPath>
            <clipPath id="beaker-clip">
              <rect x="30" y="245" width="140" height="115"/>
            </clipPath>
            <style>
              @keyframes drop-anim {{
                0%   {{ transform: translateY(0px);  opacity: 1; }}
                80%  {{ transform: translateY(52px); opacity: 1; }}
                81%  {{ transform: translateY(52px); opacity: 0; }}
                100% {{ transform: translateY(52px); opacity: 0; }}
              }}
            </style>
          </defs>

          <!-- ── Suporte universal ── -->
          <rect x="18" y="10" width="10" height="345" fill="#888" rx="3"/>
          <rect x="10" y="348" width="80" height="12" fill="#555" rx="4"/>

          <!-- ── Pinça ── -->
          <rect x="28" y="30" width="55" height="8" fill="#444" rx="2"/>
          <path d="M 83 26 L 103 24 L 103 32 L 83 36 Z" fill="#555"/>
          <path d="M 83 38 L 103 42 L 103 34 L 83 35 Z" fill="#555"/>

          <!-- ── Bureta (tubo de vidro) ── -->
          <!-- glass walls -->
          <rect x="76" y="16" width="3"  height="200" fill="none" stroke="#aabcd0" stroke-width="2"/>
          <rect x="101" y="16" width="3" height="200" fill="none" stroke="#aabcd0" stroke-width="2"/>
          <!-- top cap -->
          <rect x="74" y="14" width="32" height="5" fill="#aabcd0" rx="2"/>
          <!-- liquid inside burette -->
          <rect x="79" y="{bureta_liq_y}" width="22" height="{bureta_liq_height}"
                fill="rgba(120,180,255,0.55)" clip-path="url(#bureta-clip)"/>
          <!-- scale marks -->
          <line x1="76" y1="28"  x2="81" y2="28"  stroke="#99aabb" stroke-width="1"/>
          <line x1="76" y1="57.6"  x2="81" y2="57.6"  stroke="#99aabb" stroke-width="1"/>
          <line x1="76" y1="87.2" x2="81" y2="87.2" stroke="#99aabb" stroke-width="1"/>
          <line x1="76" y1="116.8" x2="81" y2="116.8" stroke="#99aabb" stroke-width="1"/>
          <line x1="76" y1="146.4" x2="81" y2="146.4" stroke="#99aabb" stroke-width="1"/>
          <line x1="76" y1="176" x2="81" y2="176" stroke="#99aabb" stroke-width="1"/>
          <text x="69" y="31"  fill="#8899aa" font-size="8" text-anchor="end">0</text>
          <text x="69" y="60.6"  fill="#8899aa" font-size="8" text-anchor="end">10</text>
          <text x="69" y="90.2" fill="#8899aa" font-size="8" text-anchor="end">20</text>
          <text x="69" y="119.8" fill="#8899aa" font-size="8" text-anchor="end">30</text>
          <text x="69" y="149.4" fill="#8899aa" font-size="8" text-anchor="end">40</text>
          <text x="69" y="179" fill="#8899aa" font-size="8" text-anchor="end">50</text>

          <!-- ── Torneira ── -->
          <rect x="74" y="216" width="32" height="8" fill="#555" rx="3"/>
          <rect x="108" y="218" width="12" height="4" fill="#777" rx="1"/>

          <!-- ── Nozzle + gota animada ── -->
          <!-- nozzle tube -->
          <rect x="87" y="224" width="6" height="20" fill="#888" rx="2"/>
          <!-- animated drop -->
          <g style="animation: {drop_anim}; transform-origin: 90px 244px;">
            <ellipse cx="90" cy="244" rx="4" ry="5" fill="{css_show}" opacity="0.9"/>
          </g>

          <!-- ── Copo de precipitação ── -->
          <!-- liquid in beaker -->
          <rect x="34" y="305" width="132" height="50" fill="{css_show}" rx="3" clip-path="url(#beaker-clip)"/>
          <!-- glass walls -->
          <path d="M 28 245 L 28 348 Q 28 358 40 358 L 160 358 Q 172 358 172 348 L 172 245"
                fill="none" stroke="#aabcd0" stroke-width="3"/>
          <!-- rim -->
          <line x1="26" y1="245" x2="174" y2="245" stroke="#aabcd0" stroke-width="3"/>
          <!-- stir bar -->
          <rect x="75" y="350" width="50" height="7" rx="3" fill="#ccccee" stroke="#9999bb" stroke-width="1"/>
          <rect x="92" y="350" width="16" height="7" rx="1" fill="#cc4488"/>

          <!-- ── Agitador magnético ── -->
          <rect x="10" y="362" width="200" height="14" fill="#e0e0e0" rx="3"/>
          <rect x="10" y="360" width="200" height="5" fill="#cccccc" rx="2"/>
          <circle cx="45"  cy="369" r="4" fill="#888"/>
          <circle cx="175" cy="369" r="4" fill="#888"/>
          <rect x="95" y="365" width="30" height="6" rx="2" fill="#555"/>
        </svg>
        </div>
        </body></html>
        """, height=390)

    with col_grafico:
        st.markdown("### 📈 Curva de Titulação")
        df_vis = df_auto.iloc[:n_mostrar]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#13151f")
        ax.plot(df_vis["V (mL)"], df_vis["pH"], color="#4da3ff", linewidth=2.5,
                marker="o", markersize=4, markerfacecolor="#ffdd57", markeredgecolor="#4da3ff", zorder=3)
        # Highlight current point
        ax.plot(df_vis["V (mL)"].iloc[-1], df_vis["pH"].iloc[-1],
                marker="o", markersize=10, color="#ff6b6b", zorder=5)
        ax.axvline(x=v_eq_teorico, color="#ff6b6b", linestyle="--", linewidth=1.5, label=f"P.Eq. ≈ {v_eq_teorico:.2f} mL")
        ax.axhline(y=7.0, color="#69db7c", linestyle=":", linewidth=1.2, alpha=0.6, label="pH = 7")
        ind = INDICADORES[indicador]
        ax.axhspan(ind["ph_min"], ind["ph_max"], alpha=0.15, color="orange", label=f"Zona de viragem ({indicador})")
        ax.set_xlabel("Volume de Titulante (mL)", color="#99aabb", fontsize=11)
        ax.set_ylabel("pH", color="#99aabb", fontsize=11)
        ax.set_ylim(0, 14); ax.set_xlim(0, v_max)
        ax.tick_params(colors="#99aabb")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a3a5c")
        ax.legend(facecolor="#1a1d2e", edgecolor="#2a3a5c", labelcolor="#cdd6e0", fontsize=9)
        ax.grid(True, color="#1e2535", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 Tabela de Registo")
        st.dataframe(df_vis.iloc[::5], width="stretch", hide_index=True)






elif st.session_state.modo_app == "Modo Prático-Experimental":
    st.markdown("## 🥼 Modo Prático-Experimental")
    st.markdown("**Siga o procedimento do Trabalho Laboratorial 3.** Terá de executar cada passo para chegar à fase de titulação.")
    st.markdown("---")

    fase = st.session_state.fase_pratica

    if fase == 1:
        st.subheader("Fase 1: Seleção de Material")
        st.write("Selecione **apenas** o material necessário para realizar a titulação potenciométrica descrita no protocolo.")
        
        col1, col2 = st.columns(2)
        with col1:
            m3 = st.checkbox("Pipeta volumétrica de 20 mL")
            m5 = st.checkbox("Balão volumétrico de 250 mL")
            m7 = st.checkbox("Agitador magnético e barra magnética")
            m2 = st.checkbox("Copo de precipitação de 100 mL")
            m9 = st.checkbox("Bico de Bunsen")
            m11 = st.checkbox("Papel absorvente")
        with col2:
            m1 = st.checkbox("Bureta de 25 mL")
            m6 = st.checkbox("Suporte universal e pinça")
            m10 = st.checkbox("Proveta de 50 mL")
            m4 = st.checkbox("Pompete")
            m8 = st.checkbox("Medidor de pH")
            m12 = st.checkbox("Funil de decantação")
            m13 = st.checkbox("Cadinho de porcelana")
            m14 = st.checkbox("Vidro de relógio")

        if st.button("Validar Seleção de Material", type="primary"):
            corretos = [m1, m2, m3, m4, m6, m7, m8, m11]
            incorretos = [m5, m9, m10, m12, m13, m14]
            if all(corretos) and not any(incorretos):
                st.success("✅ Material selecionado corretamente! Pode avançar.")
                st.session_state.fase_pratica = 2
                st.rerun()
            else:
                st.error("❌ A seleção está incorreta. Verifique o material necessário no protocolo e tente novamente.")

    elif fase == 2:
        st.subheader("Fase 2: Seleção de Reagentes")
        st.write("Selecione **apenas** os reagentes e consumíveis adequados à titulação configurada.")
        
        col1, col2 = st.columns(2)
        with col1:
            r6 = st.checkbox("Cloreto de Sódio (sólido)")
            r1 = st.checkbox(titulado_nome)
            r8 = st.checkbox("Sulfato de Cobre (aq)")
            r3 = st.checkbox(indicador)
        with col2:
            r4 = st.checkbox("Água destilada (esguicho)")
            r2 = st.checkbox(titulante_nome)
            r7 = st.checkbox("Solução de Iodo")

        col_b, _ = st.columns([1, 3])
        with col_b:
            if st.button("Retroceder"):
                st.session_state.fase_pratica = 1
                st.rerun()

        if st.button("Validar Seleção de Reagentes", type="primary"):
            corretos = [r1, r2, r3]
            incorretos = [r4, r6, r7, r8]
            if all(corretos) and not any(incorretos):
                st.success("✅ Reagentes selecionados corretamente!")
                st.session_state.fase_pratica = 3
                st.rerun()
            else:
                st.error("❌ A seleção está incorreta. Verifique o protocolo.")

    elif fase == 3:
        st.subheader("Fase 3: Procedimento e Registo Manual")
        
        # Sequenciador de passos
        passo = st.session_state.passo_proc
        
        st.markdown(f"**Passo atual: {passo}/3**")
        
        if passo == 1:
            st.info("🔬 **Passo 1/2**: Pipetar 20,00 mL de titulado para o copo de precipitação e adicionar 3 gotas de indicador.")

            if st.session_state.anim_passo != 1:
                if st.button("▶️ Executar: Pipetar e Adicionar Indicador", type="primary"):
                    st.session_state.anim_passo = 1
                    st.rerun()
            else:
                components.html(f"""
                <!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;overflow:hidden;">
                <div style="display:flex;justify-content:center;align-items:center;width:100%;height:340px;">
                    <div style="position:relative;width:300px;height:320px;">
                        <style>
                            body{{background:transparent;}}
                            @keyframes pipeta-move{{0%,5%{{transform:translateY(-260px);opacity:0}}15%,45%{{transform:translateY(0px);opacity:1}}50%,100%{{transform:translateY(-260px);opacity:0}}}}
                            @keyframes pipeta-liq{{0%,15%{{height:120px;y:70px;}}38%,100%{{height:0px;y:190px;}}}}
                            @keyframes pipeta-stream{{0%,18%{{opacity:0}}22%,40%{{opacity:1}}43%,100%{{opacity:0}}}}
                            @keyframes copo-liq-rise{{0%,22%{{height:0px;y:128px;}}45%,100%{{height:50px;y:78px;}}}}
                            @keyframes dropper-move{{0%,50%{{transform:translate(50px,-250px);opacity:0}}62%,85%{{transform:translate(50px,0);opacity:1}}90%,100%{{transform:translate(50px,-250px);opacity:0}}}}
                            @keyframes drop-fall{{0%,62%{{opacity:0;transform:translateY(0)}}65%,66%{{opacity:1;transform:translateY(0)}}68%{{opacity:0;transform:translateY(55px)}}70%,71%{{opacity:1;transform:translateY(0)}}73%{{opacity:0;transform:translateY(55px)}}75%,76%{{opacity:1;transform:translateY(0)}}78%{{opacity:0;transform:translateY(55px)}}100%{{opacity:0}}}}
                            .pipeta{{position:absolute;left:125px;top:10px;animation:pipeta-move 10s ease-in-out 1;}}
                            .dropper{{position:absolute;left:75px;top:10px;animation:dropper-move 10s ease-in-out 1;}}
                            .beaker{{position:absolute;left:70px;bottom:0px;}}
                        </style>
                        <svg class="beaker" width="160" height="140" viewBox="0 0 160 140">
                            <rect x="10" y="128" width="140" height="0" fill="{cor_css}" rx="4" style="animation:copo-liq-rise 10s 1 forwards;"/>
                            <path d="M 5 5 L 5 115 Q 5 130 20 130 L 140 130 Q 155 130 155 115 L 155 5" fill="none" stroke="#aabcd0" stroke-width="4"/>
                            <rect x="55" y="122" width="50" height="7" rx="3" fill="#e0e0ee" stroke="#9999bb"/>
                            <rect x="72" y="122" width="16" height="7" rx="1" fill="#cc4488"/>
                        </svg>
                        <svg class="pipeta" width="55" height="260" viewBox="0 0 55 260">
                            <rect x="24" y="235" width="7" height="55" fill="{cor_css}" style="animation:pipeta-stream 10s 1;opacity:0;"/>
                            <circle cx="27" cy="22" r="20" fill="#e74c3c"/>
                            <rect x="22" y="40" width="10" height="18" fill="#c0392b"/>
                            <rect x="20" y="55" width="14" height="100" fill="rgba(180,210,255,0.15)" stroke="#aabcd0" stroke-width="2"/>
                            <rect x="22" y="70" width="10" height="120" fill="{cor_css}" style="animation:pipeta-liq 10s 1;opacity:0.7;"/>
                            <ellipse cx="27" cy="175" rx="20" ry="28" fill="rgba(180,210,255,0.15)" stroke="#aabcd0" stroke-width="2"/>
                            <polygon points="20,203 34,203 30,240 24,240" fill="rgba(180,210,255,0.15)" stroke="#aabcd0" stroke-width="2"/>
                        </svg>
                        <svg class="dropper" width="35" height="130" viewBox="0 0 35 130">
                            <circle cx="17" cy="115" r="5" fill="#e8336d" style="animation:drop-fall 10s 1;opacity:0;"/>
                            <path d="M 5 30 Q 5 5 17 5 Q 29 5 29 30 L 27 50 L 7 50 Z" fill="#2c3e50"/>
                            <rect x="12" y="50" width="11" height="50" fill="rgba(255,255,255,0.15)" stroke="#aabcd0" stroke-width="2"/>
                            <rect x="14" y="52" width="7" height="30" fill="#e8336d" opacity="0.7"/>
                            <polygon points="12,100 23,100 21,120 14,120" fill="rgba(255,255,255,0.15)" stroke="#aabcd0" stroke-width="2"/>
                        </svg>
                    </div>
                </div>
                </body></html>
                """, height=340)

                st.info("⏳ Aguarde o fim da animação (~10 segundos) e depois confirme para continuar.")
                if st.button("✅ Confirmar: Pipetar e indicador adicionados — Avançar", type="primary"):
                    st.session_state.anim_passo = 0
                    st.session_state.passo_proc = 2
                    st.rerun()



        elif passo == 2:
            st.info("🔬 **Passo 2/2**: Preparar a bureta com o titulante e realizar a montagem para a titulação potenciométrica.")

            if st.session_state.anim_passo != 2:
                if st.button("▶️ Executar: Realizar Montagem", type="primary"):
                    st.session_state.anim_passo = 2
                    st.rerun()
            else:
                components.html(f"""
                <!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;overflow:hidden;">
                <div style="display:flex;justify-content:center;align-items:center;width:100%;height:390px;">
                    <div style="position:relative;width:320px;height:360px;">
                        <style>
                            body{{background:transparent;}}
                            @keyframes build-1{{0%,10%{{opacity:0;transform:translateY(25px)}}18%,88%{{opacity:1;transform:translateY(0)}}95%,100%{{opacity:0}}}}
                            @keyframes build-2{{0%,22%{{opacity:0;transform:translateY(25px)}}30%,88%{{opacity:1;transform:translateY(0)}}95%,100%{{opacity:0}}}}
                            @keyframes build-3{{0%,34%{{opacity:0;transform:translateY(25px)}}42%,88%{{opacity:1;transform:translateY(0)}}95%,100%{{opacity:0}}}}
                            @keyframes build-4{{0%,46%{{opacity:0;transform:translateY(25px)}}54%,88%{{opacity:1;transform:translateY(0)}}95%,100%{{opacity:0}}}}
                            @keyframes build-5{{0%,68%{{opacity:0;transform:translateY(25px)}}76%,88%{{opacity:1;transform:translateY(0)}}95%,100%{{opacity:0}}}}
                            @keyframes funil-anim{{0%,53%{{transform:translateY(-30px);opacity:0}}58%,72%{{transform:translateY(0);opacity:1}}78%,100%{{transform:translateY(-30px);opacity:0}}}}
                            @keyframes liq-enche{{0%,57%{{height:0px;y:215px;}}72%,100%{{height:180px;y:35px;}}}}
                            .suporte{{position:absolute;left:40px;top:10px;animation:build-1 12s 1;opacity:0;}}
                            .agitador{{position:absolute;left:70px;top:305px;animation:build-2 12s 1;opacity:0;}}
                            .pinca{{position:absolute;left:55px;top:90px;animation:build-3 12s 1;opacity:0;}}
                            .bureta-g{{position:absolute;left:120px;top:5px;animation:build-4 12s 1;opacity:0;}}
                            .copo-g{{position:absolute;left:90px;top:210px;animation:build-5 12s 1;opacity:0;}}
                            .funil-g{{position:absolute;left:115px;top:-30px;animation:funil-anim 12s 1;opacity:0;}}
                        </style>
                        <svg class="suporte" width="140" height="310">
                            <rect x="10" y="295" width="120" height="14" fill="#444" rx="4"/>
                            <rect x="22" y="0" width="13" height="295" fill="#777"/>
                        </svg>
                        <svg class="agitador" width="180" height="38">
                            <path d="M 5 5 L 175 5 L 185 18 L 0 18 Z" fill="#e8e8e8" stroke="#888" stroke-width="1.5"/>
                            <rect x="0" y="18" width="185" height="14" fill="#cfcfcf" stroke="#888" stroke-width="1.5"/>
                            <circle cx="35" cy="25" r="5" fill="#555"/><circle cx="155" cy="25" r="5" fill="#555"/>
                            <rect x="80" y="21" width="30" height="8" fill="#333" rx="2"/>
                        </svg>
                        <svg class="pinca" width="80" height="30">
                            <rect x="0" y="10" width="40" height="8" fill="#333"/>
                            <path d="M 40 3 L 65 0 L 65 10 L 40 15 Z" fill="#555"/>
                            <path d="M 40 28 L 65 32 L 65 22 L 40 17 Z" fill="#555"/>
                        </svg>
                        <svg class="bureta-g" width="45" height="270">
                            <rect x="12" y="215" width="21" height="0" fill="rgba(140,180,255,0.4)" style="animation:liq-enche 12s 1 forwards;"/>
                            <path d="M 10 10 L 10 215 L 17 226 L 17 238 L 28 238 L 28 226 L 35 215 L 35 10" fill="none" stroke="#aabcd0" stroke-width="3"/>
                            <rect x="5" y="229" width="35" height="7" fill="#555" rx="2"/>
                            <path d="M 20 238 L 20 255 L 25 255 L 25 238 Z" fill="#777"/>
                        </svg>
                        <svg class="funil-g" width="55" height="45">
                            <polygon points="5,0 50,0 34,28 21,28" fill="rgba(255,255,255,0.35)" stroke="#aabcd0" stroke-width="2"/>
                            <rect x="21" y="28" width="13" height="17" fill="rgba(255,255,255,0.35)" stroke="#aabcd0" stroke-width="2"/>
                        </svg>
                        <svg class="copo-g" width="130" height="110">
                            <rect x="12" y="65" width="106" height="35" fill="{cor_css}" rx="4"/>
                            <path d="M 8 5 L 8 95 Q 8 108 22 108 L 108 108 Q 122 108 122 95 L 122 5" fill="none" stroke="#aabcd0" stroke-width="3"/>
                            <rect x="42" y="98" width="46" height="7" rx="3" fill="#e0e0ee" stroke="#9999bb"/>
                            <rect x="57" y="98" width="14" height="7" rx="1" fill="#cc4488"/>
                        </svg>
                    </div>
                </div>
                </body></html>
                """, height=390)

                st.info("⏳ Aguarde o fim da animação (~12 segundos) e depois confirme para continuar.")
                if st.button("✅ Confirmar: Montagem realizada — Avançar para Titulação", type="primary"):
                    st.session_state.anim_passo = 0
                    st.session_state.passo_proc = 3
                    st.rerun()




        elif passo == 3:
            col_bancada, col_grafico = st.columns([1, 1.5], gap="large")

            with col_bancada:
                # Erro do medidor de pH (oscilação até 5% baseada no volume para ser consistente)
                np.random.seed(int(st.session_state.vol_add * 1000))
                erro_perc = np.random.uniform(-0.05, 0.05)
                ph_pratico = max(0.0, min(14.0, ph_atual * (1 + erro_perc)))
                np.random.seed() # reset

                # Medidor de pH digital
                st.markdown(f"""
                <div class='ph-meter' style='text-align: center;'>
                    <div class='ph-meter-label'>pH Meter</div>
                    <div class='ph-meter-display'>{ph_pratico:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"<div style='text-align: center; color: #8899aa; font-size: 14px; margin-bottom: 10px;'>Cor da solução: <strong style='color: {cor_texto}'>{cor_nome}</strong> ({indicador})</div>", unsafe_allow_html=True)
                
                novo_vol = bureta_ui(vol_add=st.session_state.vol_add, cor_hex=cor_css, near_ep=near_ep, key="bureta_pratico")
                if novo_vol is not None and novo_vol != st.session_state.vol_add:
                    st.session_state.vol_add = novo_vol
                    st.rerun()

            with col_grafico:
                st.markdown("### 📋 Tabela de Registo (Manual)")
                st.write("Lê o volume na bureta e o pH no medidor digital e introduz os valores na tabela. O gráfico atualizará com os teus registos.")
                
                # Editor de dados manual
                edited_df = st.data_editor(
                    st.session_state.dados_praticos,
                    num_rows="dynamic",
                    width="stretch",
                    column_config={
                        "V (mL)": st.column_config.NumberColumn("V (mL)", format="%.2f", min_value=0.0, max_value=50.0),
                        "pH": st.column_config.NumberColumn("pH", format="%.2f", min_value=0.0, max_value=14.0)
                    }
                )
                
                # Guarda os dados na session para manter consistência
                st.session_state.dados_praticos = edited_df

                # Gráfico baseado nos dados inseridos manualmente
                st.markdown("### 📈 Curva de Titulação Experimental")
                
                # Limpa linhas vazias para fazer o gráfico
                df_valid = edited_df.dropna(subset=["V (mL)", "pH"])
                
                fig3, ax3 = plt.subplots(figsize=(7, 4.5))
                fig3.patch.set_facecolor("#0e1117")
                ax3.set_facecolor("#13151f")

                if not df_valid.empty and len(df_valid) >= 2:
                    vols_man = df_valid["V (mL)"].tolist()
                    phs_man = df_valid["pH"].tolist()
                    
                    # Sort to plot correctly if user inserts out of order
                    sorted_pairs = sorted(zip(vols_man, phs_man))
                    vols_man = [x for x, y in sorted_pairs]
                    phs_man = [y for x, y in sorted_pairs]

                    ax3.plot(vols_man, phs_man, color="#ffdd57", linewidth=2.5, marker="o",
                            markersize=6, markerfacecolor="#4da3ff", markeredgecolor="#ffdd57", zorder=3)

                ax3.set_xlabel("Volume de Titulante (mL)", color="#99aabb", fontsize=11)
                ax3.set_ylabel("pH", color="#99aabb", fontsize=11)
                ax3.set_ylim(0, 14)
                ax3.set_xlim(0, max(50, df_valid["V (mL)"].max() * 1.1 if not df_valid.empty else 50))
                ax3.tick_params(colors="#99aabb")
                for spine in ax3.spines.values(): spine.set_edgecolor("#2a3a5c")
                ax3.grid(True, color="#1e2535", linewidth=0.8)

                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
