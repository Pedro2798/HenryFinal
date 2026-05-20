"""LegalMove - Streamlit wizard for contract comparison.

Single-page front-end: upload an ORIGINAL contract image and its AMENDMENT,
hit "Analyze", get the validated `ContractChangeOutput` JSON. This file is
purely a presentation layer on top of ``src.main.run_pipeline`` - the pipeline
itself (vision, two agents, Pydantic validation, Langfuse tracing) is reused
unchanged so the CLI and the web app share the same code.

Deploy
------
Push to GitHub, then on Streamlit Community Cloud connect this repo and use
``streamlit_app.py`` as the entry point. Add the OpenAI + Langfuse credentials
(and optionally ``app_password``) under *Settings -> Secrets* in the Streamlit
Cloud dashboard.
"""

from __future__ import annotations

# --- make `src` importable when Streamlit Cloud launches us from the root ---
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Local dev reads .env; on Streamlit Cloud secrets come from st.secrets and are
# synced into os.environ below so the pipeline (which reads env vars) works
# unchanged in both environments.
load_dotenv()


def _get_secret(key: str, default=None):
    """Safe st.secrets access (does not raise if secrets are not configured)."""

    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


_SECRET_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_TRACING_ENVIRONMENT",
)
for _k in _SECRET_KEYS:
    _v = _get_secret(_k)
    if _v and not os.getenv(_k):
        os.environ[_k] = str(_v)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LegalMove - Comparador de contratos",
    page_icon=":scroll:",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Optional password gate (enabled when the `app_password` secret is set)
# ---------------------------------------------------------------------------
def _password_gate() -> bool:
    expected = _get_secret("app_password")
    if not expected:
        return True  # No password configured -> open access (local dev)
    if st.session_state.get("auth_ok"):
        return True

    st.title("LegalMove - Acceso restringido")
    st.caption("Esta demo usa una API key de OpenAI con costo por uso.")
    pw = st.text_input("Contraseña", type="password", key="pw_input")
    if st.button("Entrar", type="primary"):
        if pw == str(expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False


if not _password_gate():
    st.stop()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("LegalMove - Comparador de contratos")
st.caption(
    "Subí el contrato **original** y su **enmienda** (imágenes). El sistema "
    "lee con GPT-4o Visión, dos agentes IA analizan los cambios y devuelven "
    "un reporte JSON validado por Pydantic, trazado en Langfuse."
)

with st.sidebar:
    st.markdown("### Cómo funciona")
    st.markdown(
        "1. **GPT-4o Visión** transcribe ambas imágenes.\n"
        "2. **Agente 1 (Contextualización)** mapea las secciones.\n"
        "3. **Agente 2 (Extracción)** identifica "
        "adiciones / eliminaciones / modificaciones.\n"
        "4. **Pydantic** valida el JSON final.\n"
        "5. **Langfuse** registra la traza jerárquica."
    )
    st.divider()
    model = st.selectbox(
        "Modelo",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="Default: gpt-4o (lo que pide la consigna).",
    )
    st.divider()
    st.caption(
        "Repo: [github.com/Pedro2798/HenryFinal]"
        "(https://github.com/Pedro2798/HenryFinal)"
    )


# Optional demo pairs (only if the test contracts are in the deploy)
_TEST_DIR = Path(_ROOT) / "data" / "test_contracts"
_PAIRS = {
    "Contrato de servicios (cambio simple)": (
        _TEST_DIR / "01_service_agreement_original.png",
        _TEST_DIR / "02_service_agreement_amendment.png",
    ),
    "NDA (cambios complejos: add + del + mod)": (
        _TEST_DIR / "03_nda_original.png",
        _TEST_DIR / "04_nda_amendment.png",
    ),
    "Servicios CON manchas y desalineación (robustez)": (
        _TEST_DIR / "05_service_agreement_original_dirty.png",
        _TEST_DIR / "06_service_agreement_amendment_dirty.png",
    ),
}
_PAIRS_AVAILABLE = {
    label: paths for label, paths in _PAIRS.items()
    if all(p.exists() for p in paths)
}

mode_options = ["Subir mis imágenes"]
if _PAIRS_AVAILABLE:
    mode_options.append("Usar un par de prueba")
mode = st.radio("Origen de los contratos", options=mode_options, horizontal=True)

original_bytes = None
amendment_bytes = None
orig_name = "original.png"
amd_name = "amendment.png"

if mode == "Subir mis imágenes":
    col1, col2 = st.columns(2)
    with col1:
        of = st.file_uploader(
            "Contrato ORIGINAL",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key="orig",
        )
        if of is not None:
            original_bytes = of.getvalue()
            orig_name = of.name
            st.image(of, caption="Original", use_container_width=True)
    with col2:
        af = st.file_uploader(
            "ENMIENDA / Adenda",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            key="amd",
        )
        if af is not None:
            amendment_bytes = af.getvalue()
            amd_name = af.name
            st.image(af, caption="Enmienda", use_container_width=True)
else:
    pair_label = st.selectbox("Par de prueba", list(_PAIRS_AVAILABLE.keys()))
    orig_path, amd_path = _PAIRS_AVAILABLE[pair_label]
    original_bytes = orig_path.read_bytes()
    amendment_bytes = amd_path.read_bytes()
    orig_name = orig_path.name
    amd_name = amd_path.name
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_bytes, caption=orig_name, use_container_width=True)
    with col2:
        st.image(amendment_bytes, caption=amd_name, use_container_width=True)


st.divider()

run_disabled = not (original_bytes and amendment_bytes)
if st.button(
    "Analizar contratos",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
):
    missing = [
        n for n in (
            "OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"
        )
        if not os.getenv(n)
    ]
    if missing:
        st.error(
            "Faltan credenciales: " + ", ".join(missing) +
            ". Configurá los secrets en Streamlit Cloud "
            "(Settings -> Secrets) o en .env localmente."
        )
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        orig_path = Path(td) / orig_name
        amd_path = Path(td) / amd_name
        orig_path.write_bytes(original_bytes)
        amd_path.write_bytes(amendment_bytes)

        with st.status("Procesando…", expanded=True) as status:
            try:
                # Reuse the exact CLI pipeline + its helpers.
                from src.main import (
                    _apply_langfuse_env_defaults,
                    _default_session_id,
                    _default_user_id,
                    _init_langfuse,
                    run_pipeline,
                )

                _apply_langfuse_env_defaults()

                st.write("Inicializando Langfuse (mask de credenciales activo)…")
                lf = _init_langfuse()

                st.write(f"Parseando ambas imágenes con `{model}` (Visión)…")
                st.write("Agente 1: mapa estructural • Agente 2: extracción…")

                session_id = _default_session_id(str(orig_path), str(amd_path))
                user_id = _default_user_id()

                result = run_pipeline(
                    str(orig_path), str(amd_path),
                    model=model,
                    langfuse=lf,
                    session_id=session_id,
                    user_id=user_id,
                )
                lf.flush()

                status.update(label="Análisis completo", state="complete")

            except Exception as exc:
                status.update(label="Falló el análisis", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()

        # ----- Results -----
        st.success("Reporte generado y validado por Pydantic.")

        st.subheader("Reporte estructurado (ContractChangeOutput)")
        st.json(result.model_dump())

        st.subheader("Resumen narrativo")
        st.markdown(result.summary_of_the_change)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Secciones modificadas**")
            for s in result.sections_changed:
                st.code(s, language=None)
        with col_b:
            st.markdown("**Tópicos afectados**")
            for t in result.topics_touched:
                st.code(t, language=None)

        st.divider()
        st.markdown(
            f"**Langfuse session:** `{session_id}` — "
            "buscala en *Sessions* dentro de tu dashboard de Langfuse para "
            "ver el árbol completo de spans + tokens + latencia."
        )

        st.download_button(
            "Descargar JSON",
            data=result.model_dump_json(indent=2),
            file_name="contract_change_report.json",
            mime="application/json",
        )

st.divider()
st.caption(
    "LegalMove — multi-agent contract comparison "
    "· GPT-4o Vision + LangChain + Pydantic + Langfuse"
)
