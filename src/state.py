from __future__ import annotations
import copy
from datetime import datetime
import pandas as pd
import streamlit as st

# ---------- Inicialización ----------
def ensure_defaults() -> None:
    """Inicializa todas las llaves de session_state con valores por defecto"""
    # Filtros compartidos centrales
    st.session_state.setdefault("shared.filters", {})
    st.session_state.setdefault("shared.current_page", "hist")
    
    # Detección de primera carga de páginas
    st.session_state.setdefault("sim.page_loaded", False)
    st.session_state.setdefault("hist.page_loaded", False)
    
    # Histórico
    st.session_state.setdefault("hist.df", None)
    # st.session_state.setdefault("hist.filters", {})
    st.session_state.setdefault("hist.df_filtered", None)
    st.session_state.setdefault("hist.last_loaded_at", None)
    st.session_state.setdefault("hist.skus_excluidos", None)
    st.session_state.setdefault("hist.ebitda_mensual", None)
    st.session_state.setdefault("hist.granel", None)
    st.session_state.setdefault("hist.ebitda_total", None)
    st.session_state.setdefault("hist.ebitda_simple_total", None)
    st.session_state.setdefault("hist.granel_ponderado", None)
    st.session_state.setdefault("hist.granel_optimo", None)
    st.session_state.setdefault("hist.df_optimo", None)

    # Simulación
    st.session_state.setdefault("sim.df", None)
    st.session_state.setdefault("sim.filters", {})
    st.session_state.setdefault("sim.df_filtered", None)
    st.session_state.setdefault("sim.overrides_row", {})
    st.session_state.setdefault("sim.override_pct_cost", 0.0)
    st.session_state.setdefault("sim.override_upload", None)
    st.session_state.setdefault("sim.undo_stack", [])
    st.session_state.setdefault("sim.redo_stack", [])
    st.session_state.setdefault("sim.dirty", False)
    st.session_state.setdefault("sim.last_saved_path", None)
    st.session_state.setdefault("sim.last_saved_at", None)
    st.session_state.setdefault("sim.granel", None)
    st.session_state.setdefault("sim.show_subtotals_at_top", False)
    st.session_state.setdefault("sim.plan_2026", None)
    
    # Simulación de Granel
    st.session_state.setdefault("sim.granel_df", None)
    st.session_state.setdefault("sim.granel_overrides_row", {})

    
    # Datos de fruta
    st.session_state.setdefault("fruta.receta_df", None)
    st.session_state.setdefault("fruta.info_df", None)
    st.session_state.setdefault("fruta.plan_2026", None)
    st.session_state.setdefault("sim.fruit_overrides", {})

    # UI / Otros
    st.session_state.setdefault("ui.active_tab", "Histórico")
    st.session_state.setdefault("ui.debug", False)
    st.session_state.setdefault("ui.top_n", 10)
    st.session_state.setdefault("ui.selected_rows", [])
    st.session_state.setdefault("ui.messages", [])
    st.session_state.setdefault("export.last_saved_path", None)

# ---------- Diagnóstico ----------
def session_state_table() -> None:
    """Muestra una tabla de diagnóstico de todo el session_state"""
    rows = []
    for k, v in st.session_state.items():
        vtype = type(v).__name__
        vshape = getattr(v, "shape", "")
        if vshape:
            vshape = str(vshape)
        preview = str(v)
        if isinstance(v, pd.DataFrame):
            preview = f"DataFrame[{v.shape[0]}x{v.shape[1]}]"
        elif isinstance(v, dict):
            preview = "{...}" if len(v) > 0 else "{}"
        rows.append({"key": k, "type": vtype, "shape": vshape, "preview": preview[:120]})
    
    df = pd.DataFrame(rows).sort_values("key")
    st.dataframe(df, width="stretch", height=360)

# ---------- Undo / Redo simulación ----------
def _sim_take_snapshot() -> dict:
    """Toma una instantánea del estado actual de simulación"""
    return {
        "df": st.session_state["sim.df"].copy(deep=True) if isinstance(st.session_state.get("sim.df"), pd.DataFrame) else None,
        "filters": copy.deepcopy(st.session_state.get("sim.filters", {})),
        "overrides_row": copy.deepcopy(st.session_state.get("sim.overrides_row", {})),
        "override_pct_cost": st.session_state.get("sim.override_pct_cost", 0.0),
        "override_upload": st.session_state.get("sim.override_upload", None),
        "fruit_price_overrides": copy.deepcopy(st.session_state.get("mmpp.fruit_price_overrides", {})),
        "fruit_overrides": copy.deepcopy(st.session_state.get("sim.fruit_overrides", {})),
        "granel_df": st.session_state["sim.granel_df"].copy(deep=True) if isinstance(st.session_state.get("sim.granel_df"), pd.DataFrame) else None,
        "granel_overrides_row": copy.deepcopy(st.session_state.get("sim.granel_overrides_row", {})),
    }

def sim_snapshot_push() -> None:
    """Guarda el estado actual en la pila de undo y limpia la pila de redo"""
    st.session_state["sim.undo_stack"].append(_sim_take_snapshot())
    st.session_state["sim.redo_stack"].clear()

def sim_undo() -> None:
    """Revierte al estado anterior de simulación"""
    if not st.session_state.get("sim.undo_stack"):
        return
    
    curr = _sim_take_snapshot()
    prev = st.session_state["sim.undo_stack"].pop()
    st.session_state["sim.redo_stack"].append(curr)
    
    # Restaurar estado anterior
    st.session_state["sim.df"] = prev["df"]
    st.session_state["sim.filters"] = prev["filters"]
    st.session_state["sim.overrides_row"] = prev["overrides_row"]
    st.session_state["sim.override_pct_cost"] = prev["override_pct_cost"]
    st.session_state["sim.override_upload"] = prev["override_upload"]
    st.session_state["mmpp.fruit_price_overrides"] = prev["fruit_price_overrides"]
    st.session_state["sim.fruit_overrides"] = prev["fruit_overrides"]
    st.session_state["sim.granel_df"] = prev["granel_df"]
    st.session_state["sim.granel_overrides_row"] = prev["granel_overrides_row"]
    st.session_state["sim.dirty"] = True

def sim_redo() -> None:
    """Reaplica el estado siguiente de simulación"""
    if not st.session_state.get("sim.redo_stack"):
        return
    
    next_ = st.session_state["sim.redo_stack"].pop()
    st.session_state["sim.undo_stack"].append(_sim_take_snapshot())
    
    # Aplicar estado siguiente
    st.session_state["sim.df"] = next_["df"]
    st.session_state["sim.filters"] = next_["filters"]
    st.session_state["sim.overrides_row"] = next_["overrides_row"]
    st.session_state["sim.override_pct_cost"] = next_["override_pct_cost"]
    st.session_state["sim.override_upload"] = next_["override_upload"]
    st.session_state["mmpp.fruit_price_overrides"] = next_["fruit_price_overrides"]
    st.session_state["sim.fruit_overrides"] = next_["fruit_overrides"]
    st.session_state["sim.granel_df"] = next_["granel_df"]
    st.session_state["sim.granel_overrides_row"] = next_["granel_overrides_row"]
    st.session_state["sim.dirty"] = True

# ---------- Overrides de fruta (MMPP) ----------
def apply_fruit_override(fruit: str, new_price_usdkg: float) -> None:
    """Aplica un override de precio para una fruta específica"""
    if not isinstance(fruit, str):
        return
    
    st.session_state["mmpp.fruit_price_overrides"][fruit] = float(new_price_usdkg)
    st.session_state["mmpp.dirty"] = True
    st.session_state["sim.dirty"] = True

# ---------- Helpers adicionales ----------
def get_sim_undo_count() -> int:
    """Retorna el número de operaciones disponibles para undo"""
    return len(st.session_state.get("sim.undo_stack", []))

def get_sim_redo_count() -> int:
    """Retorna el número de operaciones disponibles para redo"""
    return len(st.session_state.get("sim.redo_stack", []))

def is_sim_dirty() -> bool:
    """Retorna si la simulación tiene cambios sin guardar"""
    return st.session_state.get("sim.dirty", False)

def is_mmpp_dirty() -> bool:
    """Retorna si los datos de MMPP tienen cambios sin guardar"""
    return st.session_state.get("mmpp.dirty", False)

def clear_sim_history() -> None:
    """Limpia el historial de undo/redo"""
    st.session_state["sim.undo_stack"] = []
    st.session_state["sim.redo_stack"] = []
    st.session_state["sim.dirty"] = False

# ---------- Migración de llaves legacy ----------
def migrate_legacy_session_keys() -> None:
    """Migra llaves antiguas del session_state a los nuevos namespaces"""
    
    # Migrar archivos subidos
    if "uploaded_file" in st.session_state:
        st.session_state["hist.uploaded_file"] = st.session_state["uploaded_file"]
        del st.session_state["uploaded_file"]
    
    if "file_bytes" in st.session_state:
        st.session_state["hist.file_bytes"] = st.session_state["file_bytes"]
        del st.session_state["file_bytes"]
    
    # Migrar datos principales
    if "detalle" in st.session_state:
        st.session_state["hist.df"] = st.session_state["detalle"]
        del st.session_state["detalle"]
    
    if "df_current" in st.session_state:
        st.session_state["sim.df"] = st.session_state["df_current"]
        del st.session_state["df_current"]
    
    # Migrar ajustes universales
    if "universal_adjustments" in st.session_state:
        # Si es un dict con ajustes por fila, migrar a overrides_row
        if isinstance(st.session_state["universal_adjustments"], dict):
            st.session_state["sim.overrides_row"] = st.session_state["universal_adjustments"]
        del st.session_state["universal_adjustments"]
    
    # Migrar historial de edición
    if "edit_history" in st.session_state:
        st.session_state["sim.edit_history"] = st.session_state["edit_history"]
        del st.session_state["edit_history"]
    
    # Migrar flags de UI
    ui_flags = []
    for key in list(st.session_state.keys()):
        if key.startswith("show_"):
            ui_flags.append(key)
            # Agregar mensaje sobre el flag
            if "ui.messages" not in st.session_state:
                st.session_state["ui.messages"] = []
            st.session_state["ui.messages"].append(f"Flag UI migrado: {key}")
            del st.session_state[key]
    
    # Migrar otras variables comunes
    if "upload_applied" in st.session_state:
        if st.session_state["upload_applied"]:
            st.session_state["sim.override_upload"] = True
        del st.session_state["upload_applied"]
    
    # Marcar como dirty si hay datos migrados
    if st.session_state.get("hist.df") is not None:
        st.session_state["hist.last_loaded_at"] = datetime.now()
    
    if st.session_state.get("sim.df") is not None:
        st.session_state["sim.dirty"] = True

def ensure_session_state() -> None:
    """Inicializa y migra el session_state completo"""
    ensure_defaults()
    migrate_legacy_session_keys()

# ===================== Sistema de Filtros Compartidos =====================
def sync_filters_to_shared(page: str, filters: dict) -> None:
    """Sincroniza filtros locales al estado compartido"""
    st.session_state["shared.filters"] = filters.copy()
    st.session_state["shared.current_page"] = page

def get_shared_filters() -> dict:
    """Obtiene los filtros compartidos"""
    return st.session_state.get("shared.filters", {})

def sync_filters_from_shared(page: str) -> dict:
    """Sincroniza filtros desde el estado compartido a la página actual"""
    shared_filters = get_shared_filters()
    st.session_state["shared.current_page"] = page
    return shared_filters

def clear_shared_filters() -> None:
    """Limpia todos los filtros compartidos"""
    st.session_state["shared.filters"] = {}
