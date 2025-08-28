"""
Simulador de EBITDA por SKU (USD/kg)
Página del simulador con filtros, overrides y análisis de rentabilidad.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import math
from pathlib import Path
import io
import streamlit.components.v1 as components

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Importar con manejo de errores más robusto
try:
    # Intentar import desde src
    from src.data_io import build_detalle, REQ_SHEETS, columns_config, recalculate_totals
    from src.state import (
        ensure_session_state, session_state_table, sim_snapshot_push, 
        sim_undo, sim_redo, apply_fruit_override,
        get_sim_undo_count, get_sim_redo_count, is_sim_dirty
    )
    from src.simulator_fruit import (
        validate_fruit_inputs, get_adjusted_fruit_params, compute_mmpp_fruta_per_sku, 
        apply_fruit_overrides_to_sim, get_fruit_summary_table, validate_bulk_upload_df, 
        process_bulk_upload
    )
    from src.simulator import (
        apply_filters, get_filter_options, apply_global_overrides, 
        apply_upload_overrides, compute_ebitda, calculate_kpis,
        get_top_bottom_skus, create_ebitda_chart, create_margin_distribution_chart,
        export_escenario, validate_upload_file
    )
except ImportError as e:
    st.warning(f"⚠️ Error importando desde src/: {e}")

# ===================== Función para Validar y Corregir Signos =====================
def validate_and_correct_signs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y corrige los signos de los costos y precios en un DataFrame.
    
    Args:
        df: DataFrame a validar y corregir
        
    Returns:
        DataFrame con signos corregidos
    """
    df_corrected = df.copy()
    
    # Identificar columnas de costos
    cost_columns = [col for col in df_corrected.columns 
                    if ("USD/kg" in col or col in ["MO Directa", "MO Indirecta", "Materiales Cajas y Bolsas", 
                                                   "Materiales Indirectos", "Calidad", "Mantencion", 
                                                   "Servicios Generales", "Utilities", "Fletes", "Comex", "Guarda PT"])
                    and "Precio" not in col 
                    and "Total" not in col
                    and "EBITDA" not in col]
    
    # Corregir signos de costos (siempre negativos)
    for col in cost_columns:
        if col in df_corrected.columns:
            # Convertir a negativos si no lo están
            df_corrected[col] = -abs(df_corrected[col])
    
    # Corregir signo del precio de venta (siempre positivo)
    if "PrecioVenta (USD/kg)" in df_corrected.columns:
        df_corrected["PrecioVenta (USD/kg)"] = abs(df_corrected["PrecioVenta (USD/kg)"])
    
    return df_corrected

def recalculate_table(edited_df: pd.DataFrame, filtered_skus: list) -> pd.DataFrame:
    """
    Recalcula los totales en la tabla y actualiza la sesión.
    
    Args:
        edited_df: DataFrame editado con cambios
        filtered_skus: Lista de SKUs filtrados actualmente
        
    Returns:
        DataFrame con totales recalculados
    """
    try:
        # Validar y corregir signos antes de recalcular
        edited_df_corrected = validate_and_correct_signs(edited_df)
        
        # Recalcular totales
        edited_df_recalculated = recalculate_totals(edited_df_corrected)
        
        # Actualizar datos en sesión para la tabla editable
        st.session_state.df_current = edited_df_recalculated.copy()
        
        return edited_df_recalculated
        
    except Exception as e:
        st.error(f"❌ Error al recalcular totales: {e}")
        st.warning("⚠️ Los cambios no se guardaron en la sesión")
        return edited_df

# ===================== Sistema de Historial de Cambios =====================
def save_edit_history(sku: str, column: str, old_value: float, new_value: float) -> None:
    """
    Guarda el historial de cambios para poder revertirlos.
    
    Args:
        sku: SKU que fue editado
        column: Columna que fue editada
        old_value: Valor anterior (desde hist.df - NO editable)
        new_value: Nuevo valor (en sim.df)
    """
    if "sim.edit_history" not in st.session_state:
        st.session_state["sim.edit_history"] = {}
    
    change_key = f"{sku}_{column}"
    st.session_state["sim.edit_history"][change_key] = {
        "sku": sku,
        "column": column,
        "old_value": old_value,  # Valor original desde hist.df
        "new_value": new_value,  # Valor nuevo en sim.df
        "timestamp": pd.Timestamp.now()
    }

def revert_edit(sku: str, column: str) -> bool:
    """
    Revierte un cambio específico a su valor original.
    
    Args:
        sku: SKU a revertir
        column: Columna a revertir
        
    Returns:
        True si se pudo revertir, False en caso contrario
    """
    if "sim.edit_history" not in st.session_state:
        return False
    
    change_key = f"{sku}_{column}"
    if change_key not in st.session_state["sim.edit_history"]:
        return False
    
    change_info = st.session_state["sim.edit_history"][change_key]
    old_value = change_info["old_value"]
    
    # IMPORTANTE: NO editar hist.df - solo trabajar con sim.df
    if "sim.df" in st.session_state:
        mask = st.session_state["sim.df"]["SKU"] == sku
        if mask.any():
            idx = st.session_state["sim.df"][mask].index[0]
            # Revertir solo en sim.df, no en hist.df
            st.session_state["sim.df"].loc[idx, column] = old_value
            
            # Recalcular totales solo en sim.df
            st.session_state["sim.df"] = recalculate_totals(st.session_state["sim.df"])
            
            # Marcar como dirty para indicar que hay cambios
            st.session_state["sim.dirty"] = True
            
            # Eliminar del historial
            del st.session_state["sim.edit_history"][change_key]
            return True
    
    return False

# ===================== Función de Validación de Cálculos =====================
def validate_calculations(df: pd.DataFrame) -> dict:
    """
    Valida que los cálculos sean correctos y lógicos.
    
    Args:
        df: DataFrame con los cálculos realizados
        
    Returns:
        Diccionario con información de validación
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "calculations": {}
    }
    
    try:
        # Verificar que PrecioVenta sea positivo
        if "PrecioVenta (USD/kg)" in df.columns:
            precio = df["PrecioVenta (USD/kg)"]
            if precio <= 0:
                validation["errors"].append(f"PrecioVenta debe ser positivo, actual: {precio}")
                validation["is_valid"] = False
        
        # Verificar que Costos Totales sea positivo
        if "Costos Totales (USD/kg)" in df.columns:
            costos = df["Costos Totales (USD/kg)"]
            if costos < 0:
                validation["errors"].append(f"Costos Totales no puede ser negativo, actual: {costos}")
                validation["is_valid"] = False
        
        # Verificar que Gastos Totales sea positivo
        if "Gastos Totales (USD/kg)" in df.columns:
            gastos = df["Gastos Totales (USD/kg)"]
            if gastos < 0:
                validation["warnings"].append(f"Gastos Totales es negativo: {gastos}")
        
        # Verificar que EBITDA sea lógico
        if "EBITDA (USD/kg)" in df.columns and "PrecioVenta (USD/kg)" in df.columns and "Costos Totales (USD/kg)" in df.columns:
            ebitda = df["EBITDA (USD/kg)"]
            precio = df["PrecioVenta (USD/kg)"]
            costos = df["Costos Totales (USD/kg)"]
            
            # EBITDA debe ser Precio - Costos
            expected_ebitda = precio - costos
            if abs(ebitda - expected_ebitda) > 0.01:
                validation["errors"].append(f"EBITDA calculado incorrectamente: {ebitda} vs esperado: {expected_ebitda}")
                validation["is_valid"] = False
            
            # EBITDA no puede ser mayor que Precio
            if ebitda > precio:
                validation["errors"].append(f"EBITDA ({ebitda}) no puede ser mayor que PrecioVenta ({precio})")
                validation["is_valid"] = False
        
        # Verificar que EBITDA Pct sea lógico
        if "EBITDA Pct" in df.columns and "PrecioVenta (USD/kg)" in df.columns and "EBITDA (USD/kg)" in df.columns:
            ebitda_pct = df["EBITDA Pct"]
            precio = df["PrecioVenta (USD/kg)"]
            ebitda = df["EBITDA (USD/kg)"]
            
            if precio > 0:
                expected_pct = (ebitda / precio) * 100
                if abs(ebitda_pct - expected_pct) > 0.1:
                    validation["errors"].append(f"EBITDA Pct calculado incorrectamente: {ebitda_pct}% vs esperado: {expected_pct}%")
                    validation["is_valid"] = False
                
                # EBITDA Pct no puede ser mayor a 100%
                if ebitda_pct > 100:
                    validation["warnings"].append(f"EBITDA Pct muy alto: {ebitda_pct}%")
        
        # Guardar cálculos para referencia
        validation["calculations"] = {
            "precio": df.get("PrecioVenta (USD/kg)", [0]) if "PrecioVenta (USD/kg)" in df.columns else 0,
            "costos_totales": df.get("Costos Totales (USD/kg)", [0]) if "Costos Totales (USD/kg)" in df.columns else 0,
            "gastos_totales": df.get("Gastos Totales (USD/kg)", [0]) if "Gastos Totales (USD/kg)" in df.columns else 0,
            "ebitda": df.get("EBITDA (USD/kg)", [0]) if "EBITDA (USD/kg)" in df.columns else 0,
            "ebitda_pct": df.get("EBITDA Pct", [0]) if "EBITDA Pct" in df.columns else 0
        }
        
    except Exception as e:
        validation["errors"].append(f"Error durante validación: {str(e)}")
        validation["is_valid"] = False
    
    return validation
# ===================== Función para Aplicar Ajustes Universales =====================
def apply_universal_adjustments(df: pd.DataFrame, adjustments: dict) -> pd.DataFrame:
    if not adjustments:
        return df

    df_adjusted = df.copy()

    for cost_column, adj in adjustments.items():
        if cost_column not in df_adjusted.columns:
            st.write(f"⚠️ Columna {cost_column} no encontrada")
            continue

        # Mascara de SKUs a los que sí aplicará el ajuste (si viene lista, respétala)
        applied_skus = adj.get("applied_skus")
        if applied_skus:
            mask = df_adjusted["SKU"].astype(str).isin([str(s) for s in applied_skus])
        else:
            mask = slice(None)  # todos

        if adj["type"] == "percentage":
            df_adjusted.loc[mask, cost_column] = df_adjusted.loc[mask, cost_column] * (1 + adj["value"] / 100)
        else:  # "dollars" = nuevo valor absoluto
            df_adjusted.loc[mask, cost_column] = adj["value"]

    # Recalcular totales con la misma definición de build_detalle (ver B)
    df_adjusted = recalculate_totals(df_adjusted)
    return df_adjusted

# ===================== Configuración de la página =====================
st.set_page_config(
    page_title="Simulador de EBITDA por SKU (USD/kg)",
    page_icon="📊",
    layout="wide"
)

st.title("Simulador de EBITDA por SKU (USD/kg)")
st.markdown("Simula escenarios de variación en costos y analiza impacto en rentabilidad por SKU.")

# ===================== Carga de datos =====================
def load_base_data():
    """Carga los datos base desde archivo local o sesión."""
    
    # CAMBIO: Priorizar 'hist.df' para el simulador
    if "hist.df" in st.session_state and st.session_state["hist.df"] is not None:
        return st.session_state["hist.df"]
    
    # Si no hay datos en sesión, mostrar mensaje para cargar desde Home
    st.warning("⚠️ No hay datos cargados en la sesión")
    st.info("💡 Ve a la página Home y carga tu archivo Excel primero")
    
    # Mostrar botón para recargar
    if st.button("Ir a Datos Históricos"):
        st.switch_page("Histórico de Datos.py")
    st.stop()
    return None

# Cargar datos base
df_base = load_base_data()
df_base["SKU"] = df_base["SKU"].astype(int)

# Inicializar sim.df una sola vez
if df_base is not None and st.session_state["sim.df"] is None:
    st.session_state["sim.df"] = df_base.copy()

# Filtrar SKUs sin costos totales (igual a 0) para análisis de EBITDA más preciso
# Guardar los excluidos en variable 'skus_excluidos' para mantenerlos disponibles
if df_base is not None and "Costos Totales (USD/kg)" in df_base.columns:
    original_count = len(df_base)
    # if st.session_state["sim.df_filtered"] is not None:
    #     df_base = st.session_state["sim.df_filtered"]
    # Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
    subproductos = df_base[df_base["Costos Totales (USD/kg)"] == 0].copy()
    sin_ventas = df_base[df_base["Comex"] == 0].copy()
    skus_excluidos = pd.concat([subproductos, sin_ventas])
    skus_excluidos = skus_excluidos.drop_duplicates(subset=["SKU-Cliente"], keep="first").set_index("SKU-Cliente")
    df_base = df_base[(df_base["Costos Totales (USD/kg)"] != 0) & (df_base["Comex"] != 0)].copy()

    
    filtered_count = len(df_base)
    skus_excluidos_count = len(skus_excluidos)
    
    if original_count > filtered_count:        
        # IMPORTANTE: Recalcular totales en los datos cargados para asegurar que EBITDA Pct esté correcto
        if "EBITDA Pct" in df_base.columns:
            df_base = recalculate_totals(df_base)
        # Mostrar información sobre subproductos excluidos
        with st.expander(f"📋 **SKUs excluidos** ({skus_excluidos_count} SKUs)", expanded=False):
            st.write("**¿Por qué se excluyen estos SKUs?**")
            st.write("Son SKUs sin ventas, o con costos totales = 0, que no pueden generar EBITDA real y distorsionan el análisis financiero.")
            
            # Estadísticas de subproductos
            col1, col2, col3 = st.columns(3)
            with col1:
                if "Marca" in skus_excluidos.columns:
                    marca_counts = skus_excluidos["Marca"].value_counts()
                    st.write("**Por Marca:**")
                    for marca, count in marca_counts.head(3).items():
                        st.write(f"- {marca}: {count}")
            
            with col2:
                if "Cliente" in skus_excluidos.columns:
                    cliente_counts = skus_excluidos["Cliente"].value_counts()
                    st.write("**Por Cliente:**")
                    for cliente, count in cliente_counts.head(3).items():
                        st.write(f"- {cliente}: {count}")
            
            with col3:
                if "Especie" in skus_excluidos.columns:
                    especie_counts = skus_excluidos["Especie"].value_counts()
                    st.write("**Por Especie:**")
                    for especie, count in especie_counts.head(3).items():
                        st.write(f"- {especie}: {count}")
            
            # Tabla completa de subproductos
            st.write("**Lista completa de subproductos excluidos:**")
            st.dataframe(
                skus_excluidos[["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Costos Totales (USD/kg)"]],
                width='stretch',
                hide_index=True
            )
            
            # Botón de exportación
            csv_skus_excluidos = skus_excluidos.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Lista Completa de SKUs excluidos (CSV)",
                data=csv_skus_excluidos,
                file_name="subproductos_excluidos_completo.csv",
                mime="text/csv",
                width='stretch',
                key="download_skus_excluidos_sim_1"
            )

# ===================== Sidebar - Filtros Dinámicos =====================
st.sidebar.header("🔍 Filtros Dinámicos")

# Sistema de filtros dinámico (igual que en datos históricos)
FIELD_ALIASES = {
    "Marca": ["Marca", "Brand"],
    "Cliente": ["Cliente", "Customer"],
    "Especie": ["Especie", "Species"],
    "Condicion": ["Condicion", "Condición", "Condition"],
    "SKU": ["SKU"]
}

# Resolver alias -> columna real presente en df_base
def resolve_columns(df, aliases_map):
    resolved = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for logical, options in aliases_map.items():
        found = None
        for opt in options:
            c = cols_lower.get(opt.lower())
            if c is not None:
                found = c
                break
        if found:
            resolved[logical] = found
    return resolved

RESOLVED = resolve_columns(df_base, FIELD_ALIASES)

# Lista final de filtros (solo los que existen en la data)
FILTER_FIELDS = [k for k in ["Marca","Cliente","Especie","Condicion","SKU"] if k in RESOLVED]

def _norm_series(s: pd.Series):
    return s.fillna("(Vacío)").astype(str).str.strip()

def _apply_filters(df: pd.DataFrame, selections: dict, skip_key=None):
    out = df.copy()
    for logical, sel in selections.items():
        if logical == skip_key or not sel:
            continue
        real_col = RESOLVED[logical]
        # Mapea el placeholder "(Vacío)" a vacío real
        valid = [x if x != "(Vacío)" else "" for x in sel]
        out = out[out[real_col].fillna("").astype(str).str.strip().isin(valid)]
    return out

def _current_selections():
    selections = {}
    for logical in FILTER_FIELDS:
        selections[logical] = st.session_state.get(f"ms_sim_{logical}", [])
    return selections

# Guardar filtros en sim.filters
st.session_state["sim.filters"] = _current_selections()

# Crear filtros en filas (uno abajo del otro)
if FILTER_FIELDS:
    # Multiselects con opciones dependientes del resto, en filas separadas
    SELECTIONS = _current_selections()
    for logical in FILTER_FIELDS:
        real_col = RESOLVED[logical]
        df_except = _apply_filters(df_base, SELECTIONS, skip_key=logical)
        opts = sorted(_norm_series(df_except[real_col]).unique().tolist())
        current = [x for x in SELECTIONS.get(logical, []) if x in opts]
        st.sidebar.multiselect(logical, options=opts, default=current, key=f"ms_sim_{logical}")
else:
    st.sidebar.info("No hay campos disponibles para filtrar")

# Releer selecciones ya actualizadas por los widgets y aplicar
SELECTIONS = _current_selections()

# IMPORTANTE: Aplicar ajustes universales ANTES de filtrar
# Usar sim.df si existe y tiene ajustes, sino usar df_base
if "sim.df" in st.session_state and st.session_state["sim.df"] is not None and st.session_state.get("sim.overrides_row"):
    # Aplicar filtros a sim.df que ya incluye ajustes universales
    df_filtered = _apply_filters(st.session_state["sim.df"], SELECTIONS).copy()
else:
    # Aplicar filtros a df_base (sin ajustes universales)
    df_filtered = _apply_filters(df_base, SELECTIONS).copy()

# Orden por SKU-Cliente si existe y sin índice
sku_cliente_col = "SKU-Cliente"
if sku_cliente_col in df_filtered.columns:
    df_filtered = df_filtered.sort_values([sku_cliente_col]).reset_index(drop=True)
else:
    df_filtered = df_filtered.reset_index(drop=True)

# Guardar resultado filtrado en sim.df_filtered
st.session_state["sim.df_filtered"] = df_filtered.copy()

# ===================== Sidebar - Overrides Globales =====================
st.sidebar.header("Overrides Globales")

# Checkbox para habilitar overrides globales
enable_global = st.sidebar.checkbox("Aplicar % global a costos", value=False)

# Botones de Undo/Redo en el sidebar
col1, col2 = st.sidebar.columns(2)
with col1:
    undo_disabled = get_sim_undo_count() == 0
    if st.button("↩️ Undo", disabled=undo_disabled, help=f"Deshacer ({get_sim_undo_count()} disponible)"):
        sim_undo()
        st.rerun()

with col2:
    redo_disabled = get_sim_redo_count() == 0
    if st.button("↪️ Redo", disabled=redo_disabled, help=f"Rehacer ({get_sim_redo_count()} disponible)"):
        sim_redo()
        st.rerun()

# Mostrar estado de dirty
if is_sim_dirty():
    st.sidebar.warning("⚠️ Cambios sin guardar")
else:
    st.sidebar.success("✅ Sin cambios pendientes")

# Input para porcentaje de cambio
pct_change = 0.0  # Inicializar variable
if enable_global:
    pct_change = st.sidebar.number_input(
        "% cambio costos",
        min_value=-100.0,
        max_value=1000.0,
        value=0.0,
        step=0.5,
        format="%.1f",
        help="Porcentaje de cambio en costos (-100 a +1000)"
    )
    
    # Aplicar overrides globales sobre los datos filtrados
    df_global = apply_global_overrides(df_filtered, pct_change, enable_global)
    
    # Mostrar información del override aplicado
    if abs(pct_change) > 0.01:
        st.sidebar.success(f"✅ Override global aplicado: {pct_change:+.1f}% a costos")
        
        # Tomar snapshot antes de aplicar cambios masivos
        sim_snapshot_push()
else:
    df_global = df_filtered.copy()

# Mostrar estado de overrides
if enable_global and abs(pct_change) > 0.01:
    st.sidebar.success(f"Override global: {pct_change:+.1f}%")
elif st.session_state.get("upload_applied", False):
    st.sidebar.success("Overrides de archivo aplicados")
else:
    st.sidebar.info("Sin overrides aplicados")

# ===================== Pestañas del Simulador =====================
tab_sku, tab_precio_frutas, tab_receta = st.tabs(["📊 Retail (SKU)", "🍓 Precio Fruta", "📖 Receta"])

with tab_sku:
    # ===================== Bloque 1 - Carga de Planilla =====================
    with st.expander("📁 **Carga de Planilla (SKU-CostoNuevo)**", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Subir archivo con SKU y CostoNuevo",
                type=["xlsx", "csv"],
                help="El archivo debe contener las columnas: SKU, CostoNuevo"
            )
        with col2:
            if uploaded_file is not None:
                # Validar archivo
                is_valid, message, df_upload = validate_upload_file(uploaded_file)
                
                if is_valid:
                    st.success(f"✅ {message}")
                    
                    # Mostrar preview del archivo
                    with st.expander("📋 Preview del archivo"):
                        st.dataframe(df_upload.head(10), width='stretch')
                    
                    # Botón para aplicar overrides
                    if st.button("🚀 Aplicar Overrides", type="primary"):
                        # Tomar snapshot antes de aplicar cambios masivos
                        sim_snapshot_push()
                        
                        # Aplicar overrides desde archivo sobre los datos filtrados
                        df_with_upload, updated_count = apply_upload_overrides(df_global, df_upload)
                        
                        # Guardar en sesión
                        st.session_state.df_current = df_with_upload
                        st.session_state.upload_applied = True
                        
                        st.success(f"✅ Se aplicaron overrides a {updated_count} SKUs")
                        st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.info("📤 Selecciona un archivo para aplicar overrides")

    # ===================== Estado de la sesión =====================
    # Si hay datos en sesión, aplicarlos sobre los filtros actuales
    if st.session_state.get("sim.override_upload") and "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
        # Aplicar los overrides de sesión sobre los datos filtrados
        df_current = st.session_state["sim.df"].copy()
        # Asegurar que solo se muestren los SKUs filtrados
        filtered_skus = st.session_state["sim.df_filtered"]["SKU"].tolist()
        df_current = df_current[df_current["SKU"].isin(filtered_skus)].copy()
    else:
        df_current = df_global.copy()

    # Filtrar SKUs sin costos totales en df_current para análisis de EBITDA más preciso
    if "Costos Totales (USD/kg)" in df_current.columns:
        original_count = len(df_current)
        df_current = df_current[df_current["Costos Totales (USD/kg)"] != 0].copy()
        filtered_count = len(df_current)
        if original_count > filtered_count and st.session_state.get("show_cost_filter_info", True):
            with st.container():
                col1, col2 = st.columns([20, 1])
                with col1:
                    st.info(f"🔍 **Filtrado de datos simulados**: Se excluyeron {original_count - filtered_count} SKUs sin costos totales para un análisis de EBITDA más preciso")
                with col2:
                    if st.button("✕", key="close_cost_filter_info", help="Cerrar aviso"):
                        st.session_state.show_cost_filter_info = False
                        st.rerun()

    # Verificar si hay ajustes universales aplicados y actualizar df_current
    if st.session_state.get("sim.overrides_row"):
        # Aplicar ajustes universales a los datos filtrados
        df_current_with_adjustments = df_current.copy()
        
        for cost_column, adjustment_info in st.session_state["sim.overrides_row"].items():
            if cost_column in df_current_with_adjustments.columns:
                if adjustment_info["type"] == "percentage":
                    df_current_with_adjustments[cost_column] = df_current_with_adjustments[cost_column] * (1 + adjustment_info["value"] / 100)
                else:  # dollars
                    df_current_with_adjustments[cost_column] = adjustment_info["value"]

        
        # Recalcular totales después de aplicar ajustes
        df_current_with_adjustments = recalculate_totals(df_current_with_adjustments)
        
        # Filtrar SKUs sin costos totales después de aplicar ajustes universales
        if "Costos Totales (USD/kg)" in df_current_with_adjustments.columns:
            df_current_with_adjustments = df_current_with_adjustments[df_current_with_adjustments["Costos Totales (USD/kg)"] != 0].copy()
        
        df_current = df_current_with_adjustments.copy()

    # ===================== Bloque 2 - Tabla Editable con Todos los Costos =====================
    st.header("Detalle de Costos Simulados")


    # ===================== Ajustes Universales =====================
    st.subheader("⚙️ Ajustes Universales por Costo")

    # Obtener datos del detalle si están disponibles en la sesión
    if "hist.df" in st.session_state and st.session_state["hist.df"] is not None:
        detalle_data = st.session_state["hist.df"].copy()
        # Filtrar por SKUs actuales
        filtered_skus = df_filtered["SKU"].tolist()
        detalle_filtrado = detalle_data[detalle_data["SKU"].isin(filtered_skus)].copy()
        
        # Identificar columnas de costos (excluyendo dimensiones y totales)
        dimension_cols = ["SKU","SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]  # Removido SKU-Cliente
        total_cols = ["Costos Totales (USD/kg)", "Gastos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]
        intermediate_cols = ["PrecioVenta (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
                            "MO Total", "Materiales Total", "MMPP Total (USD/kg)"]
        
        # Columnas de costos individuales
        cost_columns = [col for col in detalle_filtrado.columns 
                        if col not in dimension_cols + total_cols + intermediate_cols]
        adj_columns = cost_columns.copy()
        adj_columns.append("PrecioVenta (USD/kg)")
        
        if cost_columns:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                selected_cost = st.selectbox(
                    "Seleccionar costo a ajustar:",
                    options=adj_columns,
                    help="Selecciona el costo específico que quieres ajustar universalmente"
                )
            
            with col2:
                adjustment_type = st.selectbox(
                    "Tipo de ajuste:",
                    options=["Porcentaje (%)", "Dólares por kg (USD/kg)"],
                    help="Ajuste por porcentaje o nuevo valor en dólares por kg"
                )
            
            with col3:
                if adjustment_type == "Porcentaje (%)":
                    adjustment_value = st.number_input(
                        "Valor del ajuste:",
                        min_value=-100.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.5,
                        format="%.1f",
                        help="Porcentaje de cambio (-100 a +1000)"
                    )
                else:
                    if selected_cost == "PrecioVenta (USD/kg)":
                        adjustment_value = st.number_input(
                            "Nuevo valor:",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.001,
                            format="%.3f",
                            help="Nuevo valor en dólares por kg"
                        )
                    else:
                        adjustment_value = st.number_input(
                            "Nuevo valor:",
                            min_value=-10.0,
                            max_value=0.0,
                            value=0.0,
                            step=0.001,
                            format="%.3f",
                            help="Nuevo valor en dólares por kg"
                        )
            
            with col4:
                if st.button("Aplicar Ajuste", type="primary"):
                    # Tomar snapshot antes de aplicar cambios masivos
                    sim_snapshot_push()
                    
                    # GUARDAR EL AJUSTE UNIVERSAL EN LA SESIÓN (NO modificar hist.df)
                    adjustment_key = f"{selected_cost}"
                    
                    # IMPORTANTE: Guardar valores originales desde hist.df (NO editable)
                    original_values = {}
                    for sku in filtered_skus:
                        if sku in st.session_state["hist.df"]["SKU"].values:
                            idx_original = st.session_state["hist.df"][st.session_state["hist.df"]["SKU"] == sku].index[0]
                            original_values[sku] = st.session_state["hist.df"].loc[idx_original, selected_cost]
                    
                    # Inicializar sim.overrides_row si no existe
                    if "sim.overrides_row" not in st.session_state:
                        st.session_state["sim.overrides_row"] = {}
                    
                    st.session_state["sim.overrides_row"][adjustment_key] = {
                        "type": "percentage" if adjustment_type == "Porcentaje (%)" else "dollars",
                        "value": adjustment_value,
                        "applied_skus": filtered_skus.copy(),  # Guardar SKUs afectados
                        "original_values": original_values,  # Guardar valores originales
                        "timestamp": pd.Timestamp.now()
                    }
                    
                    # ACTUALIZAR sim.df para que se refleje en la tabla editable y KPIs
                    # IMPORTANTE: Aplicar ajustes universales a df_base completo (no solo filtrado)
                    df_current_updated = apply_universal_adjustments(df_base, st.session_state["sim.overrides_row"])
                    
                    # IMPORTANTE: Excluir SKUs sin costos totales (igual que en df_base)
                    if "Costos Totales (USD/kg)" in df_current_updated.columns:
                        before_filter = len(df_current_updated)
                        df_current_updated = df_current_updated[df_current_updated["Costos Totales (USD/kg)"] != 0].copy()
                        after_filter = len(df_current_updated)
                    
                    # Recalcular totales en sim.df
                    df_current_updated = recalculate_totals(df_current_updated)
                                        
                    # Guardar en sim.df
                    st.session_state["sim.df"] = df_current_updated.copy()
                    
                    # Marcar como dirty
                    st.session_state["sim.dirty"] = True
                    st.rerun()
        
        # Mostrar ajustes universales activos
        if st.session_state.get("sim.overrides_row"):
            st.subheader("Ajustes Universales Activos")
            
                    # Información sobre restauración (con botón de cierre)
            if st.session_state.get("ui.messages") and any("restoration_info" in msg for msg in st.session_state["ui.messages"]):
                with st.container():
                    col1, col2 = st.columns([20, 1])
                    with col1:
                        st.info("💡 **Restauración automática**: Al eliminar un ajuste, se restauran automáticamente los valores originales del detalle histórico.")
                    with col2:
                        if st.button("✕", key="close_restoration_info", help="Cerrar aviso"):
                            # Marcar mensaje como leído
                            st.session_state["ui.messages"] = [msg for msg in st.session_state["ui.messages"] if "restoration_info" not in msg]
                            st.rerun()
            
            for cost_column, adjustment_info in st.session_state["sim.overrides_row"].items():
                adjustment_type_str = "Porcentaje" if adjustment_info["type"] == "percentage" else "Dólares"
                value_str = f"{adjustment_info['value']:+.1f}%" if adjustment_info["type"] == "percentage" else f"{adjustment_info['value']:+.3f} USD/kg"
                skus_count = len(adjustment_info["applied_skus"])
                timestamp = adjustment_info["timestamp"].strftime("%H:%M:%S")
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{cost_column}**: {value_str}")
                with col2:
                    st.write(f"**SKUs**: {skus_count}")
                with col3:
                    st.write(f"**Tipo**: {adjustment_type_str}")
                with col4:
                    if st.button("🗑️", key=f"remove_{cost_column}", help=f"Eliminar ajuste de {cost_column}"):
                        # IMPORTANTE: NO modificar hist.df - solo actualizar sim.df
                        if "sim.df" in st.session_state:
                            # Aplicar los ajustes universales restantes a df_base completo
                            remaining_adjustments = {k: v for k, v in st.session_state["sim.overrides_row"].items() if k != cost_column}
                            if remaining_adjustments:
                                # Aplicar ajustes restantes a df_base
                                df_current_updated = apply_universal_adjustments(df_base, remaining_adjustments)
                                
                                # IMPORTANTE: Excluir SKUs sin costos totales
                                if "Costos Totales (USD/kg)" in df_current_updated.columns:
                                    df_current_updated = df_current_updated[df_current_updated["Costos Totales (USD/kg)"] != 0].copy()
                                
                                # Recalcular totales
                                df_current_updated = recalculate_totals(df_current_updated)
                                st.session_state["sim.df"] = df_current_updated.copy()
                            else:
                                # Sin ajustes, usar df_base original (que ya excluye SKUs sin costos)
                                st.session_state["sim.df"] = df_base.copy()
                            
                            # Marcar como dirty
                            st.session_state["sim.dirty"] = True
                            
                            # Eliminar el ajuste
                            del st.session_state["sim.overrides_row"][cost_column]                            
                            st.rerun()
            
                            # Botón para limpiar todos los ajustes
            if st.button("Limpiar todos los ajustes", type="secondary"):
                # Tomar snapshot antes de aplicar cambios masivos
                sim_snapshot_push()
                
                # Restaurar sim.df a df_base original (que ya excluye SKUs sin costos)
                st.session_state["sim.df"] = df_base.copy()
                st.session_state["sim.overrides_row"] = {}
                st.session_state["sim.dirty"] = True
                
                st.success("✅ Todos los ajustes universales eliminados")
                st.rerun()
        
    
        # Verificar que sim.df esté disponible
        if "sim.df" not in st.session_state or st.session_state["sim.df"] is None:
            st.error("❌ **No hay datos de simulación disponibles**")
            st.info("💡 **Para usar la tabla editable, primero debes:**")
            st.info("1. 📁 Cargar datos en la página Home")
            st.info("2. 🔄 Regresar al simulador")
            st.stop()
        
        # Preparar datos para la tabla editable usando sim.df (que incluye ajustes universales)
        # Obtener datos de simulación si están disponibles en la sesión
        if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
            # Usar sim.df que ya incluye los ajustes universales aplicados
            sim_data = st.session_state["sim.df"].copy()
            # Filtrar por SKUs actuales
            filtered_skus = st.session_state["sim.df_filtered"]["SKU"].tolist()
            
            # Identificar columnas de costos (excluyendo dimensiones y totales)
            # Nota: SKU-Cliente se incluye en dimension_cols para el procesamiento pero se oculta en la tabla
            dimension_cols = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]
            total_cols = ["Gastos Totales (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]
            
            # Columnas de costos individuales
            cost_columns = [col for col in sim_data.columns 
                            if col not in dimension_cols + total_cols]
            
            # Mover columnas dimensionales al inicio
            display_order = dimension_cols + cost_columns + total_cols
            available_display_cols = [col for col in display_order if col in sim_data.columns]
            
            # Crear DataFrame para edición
            df_edit = sim_data[available_display_cols].copy()
            df_edit = df_edit[df_edit["SKU"].isin(filtered_skus)]
                    
            # Configurar columnas editables (solo costos individuales, no totales)
            editable_columns = columns_config(editable=True)
            # Aplicar estilos antes de mostrar la tabla editable (igual que en datos históricos)
            df_edit_styled = df_edit.copy()
            
            # ESTABLECER EL ÍNDICE ANTES de aplicar estilos
            df_edit_styled = df_edit_styled.set_index("SKU-Cliente")
            
            # IMPORTANTE: Guardar una copia del DataFrame original (con índice) ANTES de convertir a Styler
            df_edit_original = df_edit_styled.copy()
            
            # Aplicar formato numérico ANTES de convertir a Styler
            fmt_cols = {}
            for c in df_edit_styled.columns:
                if c not in ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]:
                    if "Pct" in c or "Porcentaje" in c:
                        fmt_cols[c] = "{:.1%}"  # Formato de porcentaje
                    elif np.issubdtype(df_edit_styled[c].dtype, np.number):
                        fmt_cols[c] = "{:.3f}"   # Formato numérico
            
            # Aplicar formato numérico al DataFrame
            if fmt_cols:
                df_edit_styled = df_edit_styled.style.format(fmt_cols)
            
            # Aplicar negritas a las columnas de totales
            total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)", "Costos Totales (USD/kg)"]
            existing_total_columns = [col for col in total_columns if col in df_edit.columns]
            
            if existing_total_columns:
                df_edit_styled = df_edit_styled.set_properties(
                    subset=existing_total_columns,
                    **{"font-weight": "bold", "background-color": "#f8f9fa"}
                )
            
            # Aplicar estilos a columnas EBITDA
            ebitda_columns = ["EBITDA (USD/kg)", "EBITDA Pct"]
            existing_ebitda_columns = [col for col in ebitda_columns if col in df_edit.columns]
            
            if existing_ebitda_columns:
                df_edit_styled = df_edit_styled.set_properties(
                    subset=existing_ebitda_columns,
                    **{"font-weight": "bold", "background-color": "#fff7ed"}
                )
            
            # El DataFrame ya tiene el índice establecido, solo aplicar estilos
            df_edit_final = df_edit_styled
            
            edited_df = st.data_editor(
                df_edit_final,
                column_config=editable_columns,
                width='stretch',
                height="auto",
                key="data_editor_detalle",
                hide_index=True
            )
            
            # Detectar cambios y recalcular totales AUTOMÁTICAMENTE
            if not edited_df.equals(df_edit_original):
                st.info("🔍 Cambios detectados en la tabla editable")
                
                # Restaurar índice para procesamiento
                edited_df_reset = edited_df.reset_index()
                
                # Guardar historial de cambios ANTES de procesar
                changes_detected = 0
                
                # Comparar contra hist.df (valores originales) para detectar cambios
                if "hist.df" in st.session_state:
                    hist_df = st.session_state["hist.df"]
                    
                    # Convertir filtered_skus a strings para que coincida con los DataFrames
                    filtered_skus_str = [str(sku) for sku in filtered_skus]
                    
                    # Buscar cambios por SKU comparando contra valores originales
                    for sku in filtered_skus_str:
                        # Buscar el SKU en hist.df (valores originales)
                        mask_hist = hist_df["SKU"] == sku
                        mask_edited = edited_df_reset["SKU"] == sku
                        
                        if mask_hist.any() and mask_edited.any():
                            # Obtener las filas correspondientes
                            original_row = hist_df[mask_hist].iloc[0]
                            edited_row = edited_df_reset[mask_edited].iloc[0]
                            
                            # Comparar columnas numéricas (excluyendo dimensiones)
                            for col in edited_df_reset.columns:
                                if col not in ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]:
                                    try:
                                        # Verificar que la columna existe en ambos DataFrames
                                        if col in original_row and col in edited_row:
                                            original_value = original_row[col]
                                            edited_value = edited_row[col]
                                            
                                            # Si hay cambio, guardar en historial
                                            if abs(original_value - edited_value) > 1e-6:  # Tolerancia para floats
                                                save_edit_history(sku, col, original_value, edited_value)
                                                changes_detected += 1
                                        else:
                                            st.warning(f"⚠️ Columna {col} no encontrada en uno de los DataFrames")
                                    except (IndexError, KeyError, TypeError) as e:
                                        st.warning(f"⚠️ Error comparando {sku} - {col}: {e}")
                                        continue
                        else:
                            st.warning(f"⚠️ SKU {sku} no encontrado en uno de los DataFrames")
                else:
                    st.warning("⚠️ No hay datos históricos disponibles para comparar cambios")
                
                if changes_detected > 0:
                    st.success(f"✅ {changes_detected} cambios detectados y guardados en historial")
                    
                    # Validar y corregir signos antes de procesar
                    edited_df_reset = validate_and_correct_signs(edited_df_reset)
                    
                    # IMPORTANTE: Recalcular totales directamente en edited_df_reset
                    edited_df_recalculated = recalculate_totals(edited_df_reset)
                    
                    # Actualizar solo sim.df (NO modificar hist.df)
                    if "sim.df" in st.session_state:
                        st.session_state["sim.df"] = edited_df_recalculated.copy()
                        st.session_state["sim.dirty"] = True
                    
                    st.success("✅ EBITDA recalculado automáticamente")
                    
                    # Forzar actualización de la vista automáticamente
                    st.rerun()
                else:
                    st.warning("⚠️ No se detectaron cambios específicos")
            else:
                st.info("ℹ️ No hay cambios en la tabla editable")
            
            # Mostrar historial de cambios y opciones de reversión
            if "sim.edit_history" in st.session_state and st.session_state["sim.edit_history"]:
                st.subheader("📝 Historial de Cambios Individuales")
                
                # Agrupar cambios por SKU para mejor visualización
                changes_by_sku = {}
                for change_key, change_info in st.session_state["sim.edit_history"].items():
                    sku = change_info["sku"]
                    if sku not in changes_by_sku:
                        changes_by_sku[sku] = []
                    changes_by_sku[sku].append(change_info)
                
                # Mostrar cambios agrupados por SKU
                for sku, changes in changes_by_sku.items():
                    with st.expander(f"🔧 SKU: {sku} ({len(changes)} cambios)", expanded=False):
                        for change in changes:
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            
                            with col1:
                                st.write(f"**{change['column']}**: {change['old_value']:.3f} → {change['new_value']:.3f}")
                            
                            with col2:
                                st.write(f"**{change['timestamp'].strftime('%H:%M:%S')}**")
                            
                            with col3:
                                st.write(f"**{change['new_value'] - change['old_value']:+.3f}**")
                            
                            with col4:
                                if st.button("↩️", key=f"revert_{change_key}_{sku}_{change['column']}", 
                                           help=f"Revertir {change['column']} a {change['old_value']:.3f}"):
                                    if revert_edit(sku, change['column']):
                                        st.success(f"✅ {change['column']} revertido a {change['old_value']:.3f}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ No se pudo revertir {change['column']}")
                        
                        # Botón para revertir todos los cambios de este SKU
                        if st.button("🔄 Revertir Todos los Cambios", key=f"revert_all_{sku}", type="secondary"):
                            reverted_count = 0
                            for change in changes:
                                if revert_edit(sku, change['column']):
                                    reverted_count += 1
                            
                            if reverted_count > 0:
                                st.success(f"✅ {reverted_count} cambios revertidos para {sku}")
                                st.rerun()
                            else:
                                st.error(f"❌ No se pudieron revertir los cambios para {sku}")
            
            # Botón para revertir todos los cambios
            if st.button("🗑️ Revertir Todos los Cambios", type="secondary", 
                    help="Revierte todos los cambios individuales a sus valores originales"):
                # Tomar snapshot antes de aplicar cambios masivos
                sim_snapshot_push()
                
                reverted_total = 0
                for change_key, change_info in list(st.session_state["sim.edit_history"].items()):
                    if revert_edit(change_info['sku'], change_info['column']):
                        reverted_total += 1
                
                if reverted_total > 0:
                    st.success(f"✅ {reverted_total} cambios revertidos en total")
                    st.rerun()
                else:
                    st.error("❌ No se pudieron revertir los cambios")
        else:
            st.error("❌ **No hay datos disponibles para el simulador**")
            st.info("💡 **Para usar el simulador, primero debes:**")
            st.info("1. 📁 Ir a la página **Inicio**")
            st.info("2. 📤 Cargar tu archivo Excel con los datos base")
            st.info("3. 🔄 Regresar al simulador")

            # Botón para ir a Inicio
            if st.button("Ir a Inicio", type="primary", width='stretch'):
                st.switch_page("Inicio.py")
            
            st.stop()

    # ===================== KPIs =====================
    # Información sobre subproductos excluidos en la vista principal
    if 'subproductos' in locals() and len(subproductos) > 0:
        if st.session_state.get("ui.messages") and any("subproductos_main" in msg for msg in st.session_state["ui.messages"]):
            
            # Información detallada sobre subproductos
            with st.expander(f"📋 **Detalles de Subproductos Excluidos** ({len(subproductos)} SKUs)", expanded=False):
                st.write("**¿Por qué se excluyen estos SKUs?**")
                st.write("Los SKUs con costos totales = 0 no pueden generar EBITDA real y distorsionan el análisis financiero.")
                
                # Estadísticas de subproductos
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "Marca" in subproductos.columns:
                        marca_counts = subproductos["Marca"].value_counts()
                        st.write("**Por Marca:**")
                        for marca, count in marca_counts.head(3).items():
                            st.write(f"- {marca}: {count}")
                
                with col2:
                    if "Cliente" in subproductos.columns:
                        cliente_counts = subproductos["Cliente"].value_counts()
                        st.write("**Por Cliente:**")
                        for cliente, count in cliente_counts.head(3).items():
                            st.write(f"- {cliente}: {count}")
                
                with col3:
                    if "Especie" in subproductos.columns:
                        especie_counts = subproductos["Especie"].value_counts()
                        st.write("**Por Especie:**")
                        for especie, count in especie_counts.head(3).items():
                            st.write(f"- {especie}: {count}")
                
                # Tabla completa de subproductos
                st.write("**Lista completa de subproductos excluidos:**")
                st.dataframe(
                    subproductos[["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Costos Totales (USD/kg)"]],
                    width='stretch'
                )
                
                # Botón de exportación
                csv_subproductos = subproductos.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Lista Completa de Subproductos (CSV)",
                    data=csv_subproductos,
                    file_name="subproductos_excluidos_completo.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_subproductos_sim_2"
                )

    st.header("📊 KPIs")

    # Calcular KPIs
    try:
        kpis = calculate_kpis(df_current)
        
        # Mostrar KPIs en métricas
        # col1, col2, col3, col4 = st.columns(4)
        col2, col3 = st.columns(2)

        # with col1:
        #     st.metric(
        #         "EBITDA Promedio (USD/kg)",
        #         f"${kpis['EBITDA Promedio (USD/kg)']:.3f}",
        #         help="EBITDA promedio por kilogramo"
        #     )
        
        with col2:
            st.metric(
                "Total SKUs",
                kpis['Total SKUs'],
                help="Número total de SKUs en la simulación (excluyendo subproductos sin costos)"
            )
            
            # Información sobre subproductos excluidos en los KPIs
            if 'subproductos' in locals() and len(subproductos) > 0:
                st.caption(f"⚠️ {len(subproductos)} subproductos excluidos (costos = 0)")
        
        with col3:
            st.metric(
                "SKUs Rentables",
                kpis['SKUs Rentables'],
                f"{kpis['SKUs Rentables']}/{kpis['Total SKUs']}",
                help="Número de SKUs con EBITDA positivo"
            )
        
        # with col4:
        #     st.metric(
        #         "Margen Promedio (%)",
        #         f"{kpis['EBITDA Promedio (%)']:.1f}%",
        #         help="Margen promedio como porcentaje del precio"
        #     )
            
    except Exception as e:
        st.error(f"❌ Error calculando KPIs: {e}")
        st.info("💡 Verifica que las columnas de EBITDA estén presentes en los datos")

    # ===================== Top y Bottom SKUs =====================
    st.header(" Top 5 y Bottom 5 SKUs por EBITDA")

    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 5 SKUs")
            top_skus, _ = get_top_bottom_skus(df_current, 5)
            if not top_skus.empty:
                # Formatear las columnas correctamente
                display_columns = ["SKU", "Cliente", "Marca"]
                
                # Buscar columnas de EBITDA disponibles
                ebitda_column = "EBITDA (USD/kg)" if "EBITDA (USD/kg)" in top_skus.columns else "EBITDAUSD_kg"
                ebitda_pct_column = "EBITDA Pct" if "EBITDA Pct" in top_skus.columns else "MargenPct"
                
                if ebitda_column in top_skus.columns:
                    display_columns.append(ebitda_column)
                if ebitda_pct_column in top_skus.columns:
                    display_columns.append(ebitda_pct_column)
                
                # Filtrar columnas que existen
                available_display_cols = [col for col in display_columns if col in top_skus.columns]
                
                st.dataframe(
                    top_skus[available_display_cols].style.format({
                        ebitda_column: "{:.3f}" if ebitda_column in top_skus.columns else None,
                        ebitda_pct_column: "{:.1f}%" if ebitda_pct_column in top_skus.columns else None
                    }),
                    width='stretch'
                )
            else:
                st.info("No hay datos para mostrar")
        
        with col2:
            st.subheader("Bottom 5 SKUs")
            _, bottom_skus = get_top_bottom_skus(df_current, 5)
            if not bottom_skus.empty:
                # Formatear las columnas correctamente
                display_columns = ["SKU", "Cliente", "Marca"]
                
                # Buscar columnas de EBITDA disponibles
                ebitda_column = "EBITDA (USD/kg)" if "EBITDA (USD/kg)" in bottom_skus.columns else "EBITDAUSD_kg"
                ebitda_pct_column = "EBITDA Pct" if "EBITDA Pct" in bottom_skus.columns else "MargenPct"
                
                if ebitda_column in bottom_skus.columns:
                    display_columns.append(ebitda_column)
                if ebitda_pct_column in bottom_skus.columns:
                    display_columns.append(ebitda_pct_column)
                
                # Filtrar columnas que existen
                available_display_cols = [col for col in display_columns if col in bottom_skus.columns]
                
                st.dataframe(
                    bottom_skus[available_display_cols].style.format({
                        ebitda_column: "{:.3f}" if ebitda_column in bottom_skus.columns else None,
                        ebitda_pct_column: "{:.1f}%" if ebitda_pct_column in bottom_skus.columns else None
                    }),
                    width='stretch'
                )
            else:
                st.info("No hay datos para mostrar")
                
    except Exception as e:
        st.error(f"❌ Error obteniendo top/bottom SKUs: {e}")
        st.info("💡 Verifica que las columnas de EBITDA estén presentes en los datos")

    # ===================== Gráficos =====================
    st.header("📈 Gráficos")

    # Configuración del gráfico
    col1, col2 = st.columns([1, 3])

    with col1:
        top_n = st.number_input(
            "Número de SKUs a mostrar",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Número de SKUs con mayor EBITDA para mostrar en el gráfico"
        )

    with col2:
        st.write("")

    # Gráfico de EBITDA por SKU
    try:
        ebitda_chart = create_ebitda_chart(df_current, top_n)
        if ebitda_chart:
            st.altair_chart(ebitda_chart)
        else:
            st.warning("⚠️ No se pudo crear el gráfico de EBITDA")
    except Exception as e:
        st.error(f"❌ Error creando gráfico de EBITDA: {e}")

    # Gráfico de distribución de márgenes
    st.subheader("📊 Distribución de Márgenes")
    try:
        margin_chart = create_margin_distribution_chart(df_current)
        if margin_chart:
            st.altair_chart(margin_chart)
        else:
            st.warning("⚠️ No se pudo crear el gráfico de distribución")
    except Exception as e:
        st.error(f"❌ Error creando gráfico de distribución: {e}")

    # ===================== Export =====================
    st.header("💾 Exportar Escenario")

    col1, col2 = st.columns([2, 1])

    with col1:
        filename_prefix = st.text_input(
            "Prefijo del archivo:",
            value="escenario_ebitda",
            help="Nombre base para el archivo de exportación"
        )

    with col2:
        if st.button("📥 Exportar a CSV", type="primary"):
            try:
                # Exportar escenario
                export_path = export_escenario(df_current, filename_prefix)
                
                # Leer archivo para descarga
                with open(export_path, 'r', encoding='utf-8') as f:
                    csv_content = f.read()
                
                # Botón de descarga
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv_content,
                    file_name=export_path.name,
                    mime="text/csv",
                    key="download_escenario_csv"
                )
                
                st.success(f"✅ Escenario exportado exitosamente a: {export_path}")
                
            except Exception as e:
                st.error(f"❌ Error exportando escenario: {e}")

    # ===================== Información adicional =====================
    st.markdown("---")
    st.markdown("""
    ### 📚 Información del Simulador

    Este simulador te permite:

    1. **Filtrar datos** por Cliente, Marca, Especie y Condición
    2. **Aplicar overrides globales** con cambios porcentuales en costos
    3. **Cargar planillas** con nuevos costos por SKU
    4. **Editar manualmente** precios y costos por fila
    5. **Analizar EBITDA** y márgenes en tiempo real
    6. **Visualizar resultados** con gráficos interactivos
    7. **Exportar escenarios** para análisis posterior

    ### 🔧 Cómo usar

    1. **Carga datos** en la página Home primero
    2. **Navega al Simulador** para análisis detallado
    3. **Aplica filtros** en el sidebar para enfocar tu análisis
    4. **Configura overrides globales** si deseas cambios porcentuales
    5. **Sube planillas** con nuevos costos para SKUs específicos
    6. **Edita valores** directamente en la tabla para ajustes finos
    7. **Analiza KPIs** y gráficos para tomar decisiones
    8. **Exporta el escenario** para compartir o analizar

    ### 📊 Interpretación de resultados

    - **EBITDA positivo**: El SKU es rentable
    - **EBITDA negativo**: El SKU genera pérdidas
    - **Margen alto**: Mayor rentabilidad relativa
    - **Margen bajo**: Menor rentabilidad relativa
    """)

with tab_precio_frutas:
    st.header("🍓 Simulador de Precios de Frutas")
    
    # Verificar que los datos de frutas estén disponibles
    receta_df = st.session_state.get("fruta.receta_df")
    info_df = st.session_state.get("fruta.info_df")
    
    if receta_df is None or info_df is None:
        st.error("❌ **Faltan datos de frutas**")
        st.info("💡 **Para usar el simulador de frutas, primero debes:**")
        st.info("1. 📁 Ir a la página **Inicio**")
        st.info("2. 📤 Cargar tu archivo Excel con las hojas RECETA_SKU e INFO_FRUTA")
        st.info("3. 🔄 Regresar al simulador")
        
        # Botón para ir a Inicio
        if st.button("Ir a Inicio", type="primary", width='stretch'):
            st.switch_page("Inicio.py")
        
        st.stop()
    
    # Inicializar fruit_overrides si no existe
    st.session_state.setdefault("sim.fruit_overrides", {})
    
    # # ===================== Información General de Frutas =====================
    # st.header("📊 Información General de Frutas")
    
    # Obtener parámetros actuales (con overrides aplicados)
    from src.simulator_fruit import get_adjusted_fruit_params, get_fruit_summary_table
    
    params_actuales = get_adjusted_fruit_params(info_df, st.session_state["sim.fruit_overrides"])
    
    # # Mostrar resumen general
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     total_frutas = len(params_actuales)
    #     st.metric(
    #         "Total de Frutas",
    #         total_frutas,
    #         help="Número total de frutas disponibles"
    #     )
    
    # with col2:
    #     precio_promedio = params_actuales["PrecioAjustadoUSD_kg"].mean()
    #     st.metric(
    #         "Precio Promedio",
    #         f"${precio_promedio:.3f}",
    #         help="Precio promedio por kg de todas las frutas"
    #     )
    
    # with col3:
    #     eficiencia_promedio = params_actuales["EficienciaAjustada"].mean()
    #     st.metric(
    #         "Eficiencia Promedio",
    #         f"{eficiencia_promedio:.1%}",
    #         help="Eficiencia promedio de todas las frutas"
    #     )
    
    # with col4:
    #     # Contar frutas con overrides aplicados
    #     frutas_con_overrides = len([ov for ov in st.session_state["sim.fruit_overrides"].values() if ov])
    #     st.metric(
    #         "Frutas Ajustadas",
    #         frutas_con_overrides,
    #         f"{frutas_con_overrides}/{total_frutas}",
    #         help="Número de frutas con ajustes de precio aplicados"
    #     )
    
    # ===================== Ajustes de Precio y Eficiencia =====================
    # st.subheader("⚙️ Ajustes de Precio y Eficiencia")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("Ajuste Individual por Fruta")
        
        # Selector de fruta
        fruta_opts = info_df.assign(
            label=lambda d: d["Fruta_id"] + " — " + d.get("Name", d["Fruta_id"])
        )
        fruta_id = st.selectbox(
            "Seleccionar Fruta:",
            options=fruta_opts["Fruta_id"],
            format_func=lambda fid: fruta_opts.loc[fruta_opts["Fruta_id"]==fid, "label"].iloc[0],
            help="Selecciona la fruta que quieres ajustar"
        )
        
        # Obtener información de la fruta seleccionada
        fruta_info = info_df[info_df["Fruta_id"] == fruta_id].iloc[0]
        precio_actual = fruta_info["Precio"]
        eficiencia_actual = fruta_info["Eficiencia"]
        nombre_fruta = fruta_info.get("Name", fruta_id)
        
        # col11, col12 = st.columns(2)
        # with col11:
        st.info(f"**Precio actual:** ${precio_actual:.3f}/kg")
        # with col12:
        #     st.info(f"**Eficiencia actual:** {eficiencia_actual:.1%}")

    with col2:
        st.subheader("Ajuste de Precio y Eficiencia")

        # --- Estado por defecto de los selectores ---
        st.session_state.setdefault("fruit.tipo", "Precio")                # "Precio" | "Eficiencia"
        st.session_state.setdefault("fruit.metodo", "Porcentaje (%)")      # solo aplica a "Precio"

        tipo_ajuste = st.session_state["fruit.tipo"]
        metodo_precio = st.session_state["fruit.metodo"]

        # --- INPUTS ARRIBA (depende del estado actual) ---
        if tipo_ajuste == "Precio":
            if metodo_precio == "Porcentaje (%)":
                valor_ajuste = st.number_input(
                    "Cambio en porcentaje:",
                    min_value=-100.0, max_value=1000.0, value=0.0,
                    step=0.5, format="%.1f",
                    help="Porcentaje de cambio (-100 a +1000)"
                )
                nuevo_precio = precio_actual * (1 + valor_ajuste / 100)
                override_data = {"price": {"type": "percentage", "value": valor_ajuste}}
            else:
                nuevo_precio = st.number_input(
                    "Nuevo precio (USD/kg):",
                    min_value=0.0, max_value=100.0, value=float(precio_actual),
                    step=0.001, format="%.3f",
                    help="Nuevo valor en dólares por kg"
                )
                cambio_pct = ((nuevo_precio / precio_actual) - 1) * 100 if precio_actual else 0.0
                st.info(f"**Cambio en porcentaje:** {cambio_pct:+.1f}%")

                override_data = {"price": {"type": "dollars", "value": nuevo_precio}}
        else:
            nueva_eficiencia = st.number_input(
                "Nueva eficiencia:",
                min_value=0.01, max_value=1.0, value=float(eficiencia_actual),
                step=0.01, format="%.2f",
                help="Eficiencia debe estar entre 0.01 y 1.0"
            )
            cambio_eficiencia = ((nueva_eficiencia / eficiencia_actual) - 1) * 100 if eficiencia_actual else 0.0
            override_data = {"efficiency": {"type": "absolute", "value": nueva_eficiencia}}

        # --- SELECTORES ABAJO (actualizan estado y relanzan) ---
        sel1, sel2 = st.columns(2)

        with sel1:
            new_tipo = st.radio(
                "Tipo de ajuste:",
                ["Precio", "Eficiencia"],
                horizontal=True,
                index=0 if tipo_ajuste == "Precio" else 1,
                key="fruit.tipo_radio"
            )

        with sel2:
            new_metodo = metodo_precio
            if new_tipo == "Precio":
                new_metodo = st.radio(
                    "Método:",
                    ["Porcentaje (%)", "Valor absoluto (USD/kg)"],
                    horizontal=True,
                    index=0 if metodo_precio == "Porcentaje (%)" else 1,
                    key="fruit.metodo_radio"
                )

        # Sincroniza cambios y vuelve a pintar inputs arriba con el nuevo estado
        if new_tipo != tipo_ajuste or (new_tipo == "Precio" and new_metodo != metodo_precio):
            st.session_state["fruit.tipo"] = new_tipo
            if new_tipo == "Precio":
                st.session_state["fruit.metodo"] = new_metodo
            else:
                # por si venías desde "Precio", deja un método por defecto guardado
                st.session_state["fruit.metodo"] = st.session_state.get("fruit.metodo", "Porcentaje (%)")
            st.rerun()
    
    with col3:
        st.subheader("Aplicar Ajuste")
        
        # Mostrar resumen del ajuste
        if tipo_ajuste == "Precio":
            if metodo_precio == "Porcentaje (%)":
                st.write(f"**Ajuste:** {valor_ajuste:+.1f}%")
                st.write(f"**Precio:** ${precio_actual:.3f} → ${nuevo_precio:.3f}")
            else:
                st.write(f"**Ajuste:** {cambio_pct:+.1f}%")
                st.write(f"**Precio:** ${precio_actual:.3f} → ${nuevo_precio:.3f}")
        else:
            st.write(f"**Ajuste:** {cambio_eficiencia:+.1f}%")
            st.write(f"**Eficiencia:** {eficiencia_actual:.1%} → {nueva_eficiencia:.1%}")
        
        # Botón para aplicar
        if st.button("🚀 Aplicar Ajuste", type="primary", width='stretch'):
            # Tomar snapshot antes de aplicar cambios masivos
            sim_snapshot_push()
            
            # Guardar el override
            st.session_state["sim.fruit_overrides"][fruta_id] = override_data
            
            # Aplicar el override al simulador
            from src.simulator_fruit import apply_fruit_overrides_to_sim
            
            if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                st.session_state["sim.df"] = apply_fruit_overrides_to_sim(
                    st.session_state["sim.df"],
                    receta_df,
                    info_df,
                    st.session_state["sim.fruit_overrides"],
                )
                
                # Marcar como dirty
                st.session_state["sim.dirty"] = True
                
                st.success(f"✅ Ajuste aplicado a {nombre_fruta}")
                st.rerun()
            else:
                st.error("❌ No hay datos de simulación disponibles")
    
    # ===================== Overrides Activos =====================
    if st.session_state["sim.fruit_overrides"]:
        st.subheader("🔧 Overrides Activos")
        
        # Mostrar overrides activos
        for fid, override in st.session_state["sim.fruit_overrides"].items():
            fruta_nombre = info_df[info_df["Fruta_id"] == fid].get("Name", fid).iloc[0] if len(info_df[info_df["Fruta_id"] == fid]) > 0 else fid
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{fruta_nombre}** ({fid})")
            
            with col2:
                if "price" in override:
                    if override["price"]["type"] == "percentage":
                        st.write(f"Precio: {override['price']['value']:+.1f}%")
                    else:
                        st.write(f"Precio: ${override['price']['value']:.3f}/kg")
                elif "efficiency" in override:
                    st.write(f"Eficiencia: {override['efficiency']['value']:.2f}")
            
            with col3:
                # Mostrar impacto en SKUs
                skus_afectados = receta_df[receta_df["Fruta_id"] == fid]["SKU"].nunique()
                st.write(f"SKUs afectados: {skus_afectados}")
            
            with col4:
                if st.button("🗑️", key=f"remove_fruit_{fid}", help=f"Eliminar ajuste de {fruta_nombre}"):
                    # Tomar snapshot antes de aplicar cambios masivos
                    sim_snapshot_push()
                    
                    # Eliminar el override
                    del st.session_state["sim.fruit_overrides"][fid]
                    
                    # Recalcular simulador
                    if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                        st.session_state["sim.df"] = apply_fruit_overrides_to_sim(
                            st.session_state["sim.df"],
                            receta_df,
                            info_df,
                            st.session_state["sim.fruit_overrides"],
                        )
                        
                        # Marcar como dirty
                        st.session_state["sim.dirty"] = True
                        
                        st.success(f"✅ Ajuste de {fruta_nombre} eliminado")
                        st.rerun()
        
        # Botón para limpiar todos
        if st.button("Limpiar Todos los Ajustes", type="secondary"):
            # Tomar snapshot antes de aplicar cambios masivos
            sim_snapshot_push()
            
            # Limpiar todos los overrides
            st.session_state["sim.fruit_overrides"] = {}
            
            # Recalcular simulador
            if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                st.session_state["sim.df"] = apply_fruit_overrides_to_sim(
                    st.session_state["sim.df"],
                    receta_df,
                    info_df,
                    {},
                )
                
                # Marcar como dirty
                st.session_state["sim.dirty"] = True
                
                st.success("✅ Todos los ajustes de frutas eliminados")
                st.rerun()
    
    # ===================== Frutas: Resumen Único + Gráficos Útiles =====================
    st.header("🍎 Resumen de Frutas")

    # 1) Tabla base de resumen (solo usamos SKUsAfectados; ignoramos contribución)
    tabla_resumen = get_fruit_summary_table(
        info_df, receta_df, st.session_state.get("sim.fruit_overrides", {}), skus_visibles=None
    ).rename(columns={"FrutaNombre": "Name"})

    # 2) Enriquecer con cambio de precio (%)
    base = params_actuales.rename(columns={"FrutaNombre":"Name"})[
        ["Fruta_id","Name","PrecioBaseUSD_kg","PrecioAjustadoUSD_kg","EficienciaBase","EficienciaAjustada"]
    ]
    frutas = base.merge(
        tabla_resumen[["Fruta_id","SKUsAfectados"]],
        on="Fruta_id",
        how="left"
    )

    frutas["SKUsAfectados"] = frutas["SKUsAfectados"].fillna(0)
    frutas["Cambio_Precio_%"] = np.where(
        frutas["PrecioBaseUSD_kg"] > 0,
        (frutas["PrecioAjustadoUSD_kg"] / frutas["PrecioBaseUSD_kg"] - 1) * 100,
        0.0
    )

    # 3) KPIs rápidos (sin contribución)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Frutas", len(frutas))
    with k2:
        # total SKUs únicos: suma por fruta puede contar un SKU más de una vez si usa varias frutas.
        # mantenemos el nombre “(únicos)” pero aclaración: es “por fruta”.
        st.metric("SKUs Afectados (por fruta)", int(frutas["SKUsAfectados"].sum()))
    with k3:
        st.metric("ΔPrecio Promedio", f"{frutas['Cambio_Precio_%'].mean():+.1f}%")
    with k4:
        if not frutas.empty:
            top_idx = frutas["SKUsAfectados"].astype(float).idxmax()
            st.metric("Fruta más usada", frutas.loc[top_idx, "Name"])
        else:
            st.metric("Fruta más usada", "—")

    # 4) Tabla única (ordenada por SKUs afectados desc, precio desc)
    cols_tabla = [
        "Name", "PrecioBaseUSD_kg", "PrecioAjustadoUSD_kg", "Cambio_Precio_%", "SKUsAfectados"
    ]
    frutas_view = (frutas
        .sort_values(["SKUsAfectados","PrecioAjustadoUSD_kg"], ascending=[False, False])
        [cols_tabla]
    )

    st.dataframe(
        frutas_view.style.format({
            "PrecioBaseUSD_kg": "{:.3f}",
            "PrecioAjustadoUSD_kg": "{:.3f}",
            "Cambio_Precio_%": "{:+.1f}%",
            "EficienciaBase": "{:.1%}",
            "EficienciaAjustada": "{:.1%}",
            "SKUsAfectados": "{:.0f}",
        }),
        use_container_width=True, hide_index=True
    )

    # 5) Expander: Insights rápidos de frutas (sin contribución)
    with st.expander("📊 Insights rápidos de frutas", expanded=False):
        f = frutas.copy()
        f["SKUsAfectados"] = f["SKUsAfectados"].fillna(0)

        colL, colR = st.columns(2)

        # ---------- Columna izquierda ----------
        with colL:
            st.subheader("🏆 Top 5 por SKUs únicos")
            top_skus = f.sort_values("SKUsAfectados", ascending=False).head(5)
            if top_skus.empty:
                st.write("No hay datos.")
            else:
                for _, r in top_skus.iterrows():
                    st.write(f"• **{r['Name']}** — {int(r['SKUsAfectados'])} SKUs")

            st.subheader("📈 Stats de precios (USD/kg)")
            p = f["PrecioAjustadoUSD_kg"].dropna()
            if not p.empty:
                st.write(
                    f"• **Rango**: ${p.min():.3f} — ${p.max():.3f}/kg\n\n"
                    f"• **p25/mediana/p75**: ${p.quantile(0.25):.3f} / ${p.median():.3f} / ${p.quantile(0.75):.3f}/kg\n\n"
                    f"• **Desv. estándar (σ)**: ${p.std(ddof=1):.3f}/kg"
                )
        # ---------- Columna derecha ----------
        with colR:
            st.subheader("💰 Top 5 precios más altos")
            top_precio = f.sort_values("PrecioAjustadoUSD_kg", ascending=False).head(5)
            for _, r in top_precio.iterrows():
                st.write(f"• **{r['Name']}** — ${r['PrecioAjustadoUSD_kg']:.3f}/kg")

            st.subheader("🧊 Top 5 precios más bajos (>0)")
            low_precio = f[f["PrecioAjustadoUSD_kg"] > 0].sort_values("PrecioAjustadoUSD_kg", ascending=True).head(5)
            for _, r in low_precio.iterrows():
                st.write(f"• **{r['Name']}** — ${r['PrecioAjustadoUSD_kg']:.3f}/kg")


    # 6) Gráfico A: Top N por SKUs afectados (barra)
    try:
        import plotly.express as px
        topN = st.slider("Top N para gráfico de SKUs afectados", 5, min(25, len(frutas_view)), 10)
        g = frutas_view.head(topN).copy()

        fig_bar = px.bar(
            g,
            x="Name",
            y="SKUsAfectados",
            title="Top N frutas por SKUs afectados",
            labels={"Name": "Fruta", "SKUsAfectados": "SKUs afectados"},
            text="SKUsAfectados"
        )
        fig_bar.update_layout(xaxis_tickangle=-30, height=420, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    except ImportError:
        st.info("Para ver gráficos, instala plotly: `pip install plotly`")

# 7) Gráfico B: Dispersión Precio vs Eficiencia (tamaño = SKUs, color = ΔPrecio%)
# try:
#     import plotly.express as px
#     fig_scatter = px.scatter(
#         frutas,
#         x="PrecioAjustadoUSD_kg", y="EficienciaAjustada",
#         size=frutas["SKUsAfectados"].fillna(0).clip(lower=0.1),  # que no desaparezcan
#         color="Cambio_Precio_%",
#         hover_data=["Name","SKUsAfectados"],
#         labels={"PrecioAjustadoUSD_kg":"Precio (USD/kg)","EficienciaAjustada":"Eficiencia"},
#         color_continuous_scale="RdBu_r"
#     )
#     fig_scatter.update_layout(
#         title="Precio vs. Eficiencia (tamaño = SKUs afectados, color = ΔPrecio%)",
#         height=420, showlegend=False
#     )
#     st.plotly_chart(fig_scatter, use_container_width=True)
# except ImportError:
#     pass
    # ===================== Información del Simulador =====================
    st.header("📚 Información del Simulador de Frutas")
    
    with st.expander("ℹ️ Cómo usar el simulador", expanded=False):
        st.markdown("""
        ### 🎯 **Objetivo del Simulador**
        
        Este simulador te permite ajustar precios y eficiencias de frutas para analizar su impacto en:
        - **Costos de MMPP (Fruta)** por SKU
        - **EBITDA** de cada producto
        - **Contribución total** de cada fruta al negocio
        
        ### 🔧 **Cómo usar**
        
        1. **Selecciona una fruta** del dropdown
        2. **Elige el tipo de ajuste**:
           - **Precio**: Porcentaje (%) o valor absoluto (USD/kg)
           - **Eficiencia**: Valor entre 0.01 y 1.0
        3. **Ingresa el valor** del ajuste
        4. **Aplica el cambio** con "🚀 Aplicar Ajuste"
        5. **Revisa el impacto** en tiempo real
        
        ### 📊 **Interpretación de resultados**
        
        - **Contribución positiva**: La fruta contribuye al costo del producto
        - **SKUs afectados**: Número de productos que usan esta fruta
        - **Cambio de precio**: Impacto del ajuste en el costo final
        - **Impacto total**: Suma de todos los cambios aplicados
        
        ### ⚠️ **Consideraciones importantes**
        
        - Los cambios se aplican **inmediatamente** a todos los SKUs que usan esa fruta
        - **hist.df** permanece inmutable (datos históricos originales)
        - **sim.df** contiene todos los cambios aplicados
        - Usa **Undo/Redo** para revertir cambios masivos
        """)
    
    with st.expander("🔍 Detalles técnicos", expanded=False):
        st.markdown("""
        ### 📈 **Fórmulas utilizadas**
        
        **Contribución por SKU:**
        ```
        contrib_pos = PrecioAjustadoUSD_kg × Porcentaje ÷ EficienciaAjustada
        MMPP (Fruta) (USD/kg) = -contrib_pos
        ```
        
        **Impacto del ajuste:**
        ```
        Impacto = Contrib_Nueva - Contrib_Base
        ```
        
        ### 🗄️ **Estructura de datos**
        
        - **RECETA_SKU**: SKU, Fruta_id, Porcentaje
        - **INFO_FRUTA**: Fruta_id, Precio, Eficiencia, Name
        - **Overrides**: {fruta_id: {"price": {"type": "percentage"|"dollars", "value": float}}}
        
        ### 🔄 **Flujo de recálculo**
        
        1. Aplicar overrides a parámetros base
        2. Recalcular contribuciones por SKU
        3. Actualizar MMPP (Fruta) en sim.df
        4. Recalcular totales (EBITDA, costos totales)
        5. Marcar sim.dirty = True
        """)
    
    with st.expander("📋 Estado actual del simulador", expanded=False):
        # Mostrar estado actual
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍓 Frutas disponibles")
            if info_df is not None:
                st.metric("Total frutas", len(info_df))
                st.metric("Frutas con nombre", len(info_df[info_df["Name"].notna()]))
                st.metric("Precio promedio", f"${info_df['Precio'].mean():.3f}/kg")
                st.metric("Eficiencia promedio", f"{info_df['Eficiencia'].mean():.1%}")
            else:
                st.warning("No hay datos de frutas disponibles")
        
        with col2:
            st.subheader("📊 Recetas disponibles")
            if receta_df is not None:
                st.metric("Total recetas", len(receta_df))
                st.metric("SKUs únicos", receta_df["SKU"].nunique())
                st.metric("Frutas únicas", receta_df["Fruta_id"].nunique())
                st.metric("Porcentaje promedio", f"{receta_df['Porcentaje'].mean():.1f}%")
            else:
                st.warning("No hay datos de recetas disponibles")
        
        # Mostrar overrides activos
        if st.session_state["sim.fruit_overrides"]:
            st.subheader("🔧 Ajustes activos")
            for fid, override in st.session_state["sim.fruit_overrides"].items():
                if "price" in override:
                    fruta_info = info_df[info_df["Fruta_id"] == fid].iloc[0]
                    nombre = fruta_info.get("Name", fid)
                    tipo = override["price"]["type"]
                    valor = override["price"]["value"]
                    
                    if tipo == "percentage":
                        st.write(f"• **{nombre}**: {valor:+.1f}%")
                    else:
                        st.write(f"• **{nombre}**: ${valor:.3f}/kg")
        else:
            st.info("No hay ajustes activos")
    
    # Separador visual
    st.markdown("---")
    
    # Footer con información adicional
    st.caption("""
    💡 **Tip**: Usa el botón "🔍 Diagnóstico session_state" en la página principal para ver el estado completo de la aplicación.
    
    📚 **Documentación**: Este simulador está diseñado para trabajar con datos de frutas y recetas de SKUs, permitiendo análisis de sensibilidad y optimización de costos.
    """)

# ===================== ADN - Visor de Recetas por SKU =====================
@st.dialog("🍓 Receta del SKU", width="large")
def ver_receta_dialog(sku: str, receta_df: pd.DataFrame, info_df: pd.DataFrame):
    # Normaliza tipos
    receta = receta_df.copy()
    receta["SKU"] = receta["SKU"].astype(int)
    receta = receta[receta["SKU"] == int(sku)].copy()

    if receta.empty:
        st.info("No hay líneas de receta para este SKU.")
        return

    # Enriquecer con INFO_FRUTA
    if info_df is not None and not info_df.empty:
        info = info_df[["Fruta_id","Precio","Eficiencia","Name"]].copy()
        info["Fruta_id"] = info["Fruta_id"].astype(str).str.strip()
        receta["Fruta_id"] = receta["Fruta_id"].astype(str).str.strip()
        det = receta.merge(info, on="Fruta_id", how="left")

        # Overrides de precio si existen
        overrides = st.session_state.get("sim.fruit_overrides", {})
        if overrides:
            def precio_ajustado(fid, base):
                ov = overrides.get(str(fid))
                if not ov or "price" not in ov: 
                    return base
                kind = ov["price"]["type"]; val = float(ov["price"]["value"])
                return max(0.0, float(base)*(1+val/100.0) if kind=="percentage" else val)

            det["Precio"] = det.apply(lambda r: precio_ajustado(r["Fruta_id"], pd.to_numeric(r["Precio"], errors="coerce")), axis=1)

        # Contribución positiva = Precio * (Porcentaje/100) / Eficiencia
        pct  = pd.to_numeric(det["Porcentaje"], errors="coerce").fillna(0) / 100.0
        pr   = pd.to_numeric(det["Precio"], errors="coerce").fillna(0).clip(lower=0)
        opt  = pd.to_numeric(det["Óptimo"], errors="coerce").fillna(0) / 100.0
        eff  = pd.to_numeric(det["Eficiencia"], errors="coerce").fillna(0).clip(lower=0.01, upper=1.0)
        det["Name"] = det["Name"].fillna(det["Fruta_id"])
        det["Contribucion Original (USD/kg)"] = (pr * pct) / eff
        det["Contribucion Óptima (USD/kg)"] = (pr * opt) / eff
        det.rename(columns={"Porcentaje":"Porcentaje Original", "Óptimo":"Porcentaje Óptimo"}, inplace=True)

        # Cabecera compacta
        c1, c2, c3, c4, c5 = st.columns([1,1,2,2,2])
        with c1: st.metric("SKU", sku)
        with c2: st.metric("Frutas usadas", int(det["Fruta_id"].nunique()))
        with c3: 
            total = det["Contribucion Original (USD/kg)"].sum()
            st.metric("MMPP (Fruta) Simulado - Original", f"{total:.3f} USD/kg")
        with c4:
            if det["Contribucion Óptima (USD/kg)"].sum() > 0:
                total = det["Contribucion Óptima (USD/kg)"].sum()
                st.metric("MMPP (Fruta) Simulado - Óptimo", f"{total:.3f} USD/kg")
            else:
                st.metric("MMPP (Fruta) Simulado - Óptimo", "No hay óptimo", help="Producto no considerado para 2026")
        with c5:
            if det["Contribucion Óptima (USD/kg)"].sum() > 0:
                total = (det["Contribucion Original (USD/kg)"] - det["Contribucion Óptima (USD/kg)"]).sum()
                st.metric("MMPP (Fruta) Simulado - Diferencia", f"{total:.3f} USD/kg")
            else:
                st.metric("MMPP (Fruta) Simulado - Diferencia", "No hay óptimo", help="Producto no considerado para 2026")

        st.subheader("💰 Contribución por fruta (USD/kg)")
        st.dataframe(
            det[["Name","Contribucion Original (USD/kg)","Porcentaje Original","Contribucion Óptima (USD/kg)","Porcentaje Óptimo"]]
                .sort_values("Contribucion Óptima (USD/kg)", ascending=False)
                .style.format({"Contribucion Original (USD/kg)":"{:.3f}","Porcentaje Original":"{:.2f}%","Contribucion Óptima (USD/kg)":"{:.3f}","Porcentaje Óptimo":"{:.2f}%","Precio":"{:.3f}","Eficiencia":"{:.3f}"}),
            width='stretch', hide_index=True
        )

        # Footer pegado
        st.markdown('<div class="modal-footer"></div>', unsafe_allow_html=True)

    else:
        st.dataframe(receta, width='stretch', hide_index=True)


with tab_receta:
    st.header("📖 Visor de Recetas por SKU")
    
    # Verificar que tenemos datos de recetas
    if receta_df is None:
        st.error("❌ No hay datos de recetas disponibles. Sube un archivo con la hoja 'RECETA_SKU' en la página de Inicio.")
        st.stop()
    
    # Obtener SKUs visibles (usar sim.df_filtered si existe, sino sim.df)
    if "sim.df_filtered" in st.session_state and st.session_state["sim.df_filtered"] is not None:
        skus_visibles = st.session_state["sim.df_filtered"]["SKU"].tolist()
    elif "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
        skus_visibles = st.session_state["sim.df"]["SKU"].tolist()
    else:
        skus_visibles = None
    
    # Filtrar recetas por SKUs visibles si hay filtros aplicados
    if skus_visibles:
        receta_filtrada = receta_df[receta_df["SKU"].astype(int).isin(skus_visibles)].copy()
        st.info(f"📊 Mostrando recetas para {len(skus_visibles)} SKUs visibles (filtrados)")
    else:
        receta_filtrada = receta_df.copy()
        st.info(f"📊 Mostrando todas las recetas disponibles ({receta_df['SKU'].nunique()} SKUs únicos)")
    
    # ===================== Resumen de Recetas =====================
    st.subheader("📋 Resumen de Recetas")
    
    col2, col3 = st.columns([2,2])
    
    # with col1:
    #     st.metric(
    #         "Total Recetas",
    #         len(receta_filtrada),
    #         help="Número total de líneas de receta"
    #     )
    
    with col2:
        st.metric(
            "Recetas Disponibles",
            receta_filtrada["SKU"].nunique(),
            help="Número total de recetas"
        )
    
    with col3:
        st.metric(
            "Frutas Únicas",
            receta_filtrada["Fruta_id"].nunique(),
            help="Número de frutas diferentes"
        )
    
    # with col4:
    #     st.metric(
    #         "Porcentaje Promedio",
    #         f"{receta_filtrada['Porcentaje'].mean():.1f}%",
    #         help="Porcentaje promedio por receta"
    #     )
    
    # ===================== Lista paginada con botón "Ver receta" =====================
    st.subheader("📊 SKUs con Recetas Disponibles")
    
    # Crear tabla compacta de SKUs
    skus_summary = receta_filtrada.groupby("SKU").agg({
        "Fruta_id": "count",
        "Porcentaje": "sum"
    }).reset_index()
    skus_summary.columns = ["SKU", "Frutas_Usadas", "Porcentaje_Total"]
    skus_summary["SKU"] = skus_summary["SKU"].astype(int)

    # Enriquecer con información adicional si está disponible
    if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
        sim_df = st.session_state["sim.df"]
        sim_df["SKU"] = sim_df["SKU"].astype(int)
        skus_summary = skus_summary.merge(
            sim_df[["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "MMPP (Fruta) (USD/kg)", "EBITDA (USD/kg)"]],
            on="SKU",
            how="left"
        )
    
    # Asegura tipos consistentes
    skus_summary = skus_summary.copy()
    skus_summary["SKU"] = skus_summary["SKU"].astype(int)
    receta_filtrada["SKU"] = receta_filtrada["SKU"].astype(int)
    if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
        st.session_state["sim.df"]["SKU"] = st.session_state["sim.df"]["SKU"].astype(int)
    
    # --- Estado de paginación ---
    if "adn.page" not in st.session_state:
        st.session_state["adn.page"] = 1
    # Controles de página
    colp1, colp7, colp2, colp3, colp4, colp5, colp6 = st.columns([2,3,1,1,1,1,1])
    with colp1:
        page_size = st.selectbox(placeholder="Filas por página", label="Filas por página", options=[10, 20, 50], index=1, key="adn_page_size", label_visibility="visible", )
        total_rows = len(skus_summary)
        total_pages = max(1, math.ceil(total_rows / page_size))
        st.caption(f"{total_rows} filas • {total_pages} páginas")
    
    with colp2:
        if st.button("⏮️", disabled=st.session_state["adn.page"] <= 1):
            st.session_state["adn.page"] = 1
            st.rerun()
    with colp3:
        if st.button("◀️", disabled=st.session_state["adn.page"] <= 1):
            st.session_state["adn.page"] -= 1
            st.rerun()
    with colp4:
        st.number_input("Página", min_value=1, max_value=total_pages, step=1,
                        value=st.session_state["adn.page"], key="adn_page_num",
                        label_visibility="collapsed")
        # sincroniza si el user escribe un número
        if st.session_state["adn.page"] != st.session_state["adn_page_num"]:
            st.session_state["adn.page"] = st.session_state["adn_page_num"]
            st.rerun()
    with colp5:
        if st.button("▶️", disabled=st.session_state["adn.page"] >= total_pages):
            st.session_state["adn.page"] += 1
            st.rerun()
    with colp6:
        if st.button("⏭️", disabled=st.session_state["adn.page"] >= total_pages):
            st.session_state["adn.page"] = total_pages
            st.rerun()
    
    # Slice de la página actual
    start = (st.session_state["adn.page"] - 1) * page_size
    end = start + page_size
    page_df = skus_summary.iloc[start:end].copy()    
    # Encabezados
    head = st.container()
    with head:
        hc = st.columns([1, 3, 2, 2, 2, 2])
        hc[0].markdown("**SKU**")
        hc[1].markdown("**Descripción**")
        hc[2].markdown("**Marca**")
        hc[3].markdown("**Cliente**")
        hc[4].markdown("**Frutas**")
        hc[5].markdown("**Acción**")
    
    # Filas con botón por fila
    for _, row in page_df.iterrows():
        sku_cliente = str(row["SKU-Cliente"])
        sku = str(row["SKU"])
        desc = row.get("Descripcion", "")
        marca = row.get("Marca", "")
        cliente = row.get("Cliente", "")
        frutas_usadas = int(row.get("Frutas_Usadas", 0))
    
        cols = st.columns([1, 3, 2, 2, 2, 2])
        cols[0].write(sku)
        cols[1].write(desc)
        cols[2].write(marca)
        cols[3].write(cliente)
        cols[4].write(frutas_usadas, help="Número de frutas diferentes usadas en el SKU")
    
        btn_key = f"ver_receta_{sku_cliente}_{start}"
        if cols[5].button("Ver receta", key=btn_key, width='stretch'):
            ver_receta_dialog(sku, st.session_state["fruta.receta_df"], st.session_state["fruta.info_df"])
    # Línea de ayuda
    st.caption("Tip: usa los controles de página para navegar y el botón **Ver receta** para abrir el modal.")
    
    # ===================== Estadísticas por Fruta =====================
    st.subheader("📈 Estadísticas por Fruta")
    
    if info_df is not None and not receta_filtrada.empty:
        # Calcular estadísticas por fruta
        stats_fruta = receta_filtrada.groupby("Fruta_id").agg({
            "SKU": "nunique",
            "Porcentaje": "mean",
            "Óptimo": ["mean", "sum"],
        }).reset_index()
        
        # Flatten column names
        stats_fruta.columns = ["Fruta_id", "SKUs_Usados", "Porcentaje Original Promedio", "Porcentaje Óptimo Promedio", "Porcentaje Óptimo Total"]
        
        # Enriquecer con información de frutas
        stats_fruta = stats_fruta.merge(
            info_df[["Fruta_id", "Precio", "Eficiencia", "Name"]],
            on="Fruta_id",
            how="left"
        )
        
        # Calcular contribución total por fruta
        stats_fruta["Contribucion_Total"] = (
            stats_fruta["Precio"] * 
            stats_fruta["Porcentaje Óptimo Total"] / 100 / 
            stats_fruta["Eficiencia"]
        )
        
        view_fruta = stats_fruta[["Name", "SKUs_Usados", "Porcentaje Original Promedio", "Porcentaje Óptimo Promedio", "Precio"]]
        view_fruta.sort_values(by="SKUs_Usados", ascending=False, inplace=True)
        
        # Mostrar tabla de estadísticas
        st.dataframe(
            view_fruta.style.format({
                "SKUs_Usados": "{:.0f}",
                "Porcentaje Original Promedio": "{:.2f}%",
                "Porcentaje Óptimo Promedio": "{:.2f}%",
                "Porcentaje Óptimo Total": "{:.1f}%",
                "Precio": "{:.3f}",
                "Eficiencia": "{:.3f}",
                "Contribucion_Total": "{:.3f}"
            }),
            column_config={
                "Name": st.column_config.TextColumn(width="small"),
                "SKUs_Usados": st.column_config.NumberColumn(width="small"),
                "Porcentaje Original Promedio": st.column_config.NumberColumn(width="small", help="Este porcentaje considera los monoproductos"),
                "Porcentaje Óptimo Promedio": st.column_config.NumberColumn(width="small", help="Este porcentaje considera los monoproductos"),
                "Precio": st.column_config.NumberColumn(width="small"),
            },
            width='stretch',
            hide_index=True
        )
        
        # Gráfico de top frutas por uso
        st.subheader("🏆 Top Frutas por Uso")
        
        try:
            import plotly.express as px
            
            # Top 10 frutas por número de SKUs
            top_frutas = stats_fruta.nlargest(10, "SKUs_Usados")
            
            fig_top_frutas = px.bar(
                top_frutas,
                x="Name",
                y="SKUs_Usados",
                title="Top 10 Frutas por Número de SKUs que las Usan",
                color="Contribucion_Total",
                color_continuous_scale="Blues"
            )
            
            fig_top_frutas.update_layout(
                xaxis_tickangle=-45,
                showlegend=False
            )
            
            st.plotly_chart(fig_top_frutas, width='stretch')
            
        except ImportError:
            st.info("📊 Para ver gráficos, instala plotly: `pip install plotly`")
    
    # ===================== Descarga de Datos =====================
    st.subheader("📥 Descarga de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Descargar recetas filtradas
        if not receta_filtrada.empty:
            csv_recetas = receta_filtrada.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Recetas Filtradas (CSV)",
                data=csv_recetas,
                file_name=f"recetas_filtradas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No hay recetas para descargar")
    
    with col2:
        # Descargar estadísticas por fruta
        if 'stats_fruta' in locals() and not stats_fruta.empty:
            csv_stats = stats_fruta.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Estadísticas por Fruta (CSV)",
                data=csv_stats,
                file_name=f"stats_frutas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No hay estadísticas para descargar")
    
    # ===================== Información de la Pestaña ADN =====================
    with st.expander("ℹ️ Acerca de la pestaña ADN", expanded=False):
        st.markdown("""
        ### 🧬 **¿Qué es ADN?**
        
        La pestaña **ADN** te permite analizar la "composición genética" de cada SKU:
        
        - **Recetas completas**: Ver qué frutas componen cada producto
        - **Porcentajes de composición**: Entender la proporción de cada ingrediente
        - **Contribución por fruta**: Analizar el impacto de cada fruta en el costo
        - **Estadísticas agregadas**: Resumen de uso de frutas en todos los SKUs
        
        ### 🔍 **Funcionalidades principales**
        
        1. **Visor de recetas**: Tabla compacta de todos los SKUs con recetas
        2. **Modal de receta**: Vista detallada de la composición de cada SKU
        3. **Análisis de contribución**: Impacto en USD/kg de cada fruta
        4. **Estadísticas por fruta**: Uso agregado de cada ingrediente
        5. **Descarga de datos**: Exportar análisis en formato CSV
        
        ### 📊 **Interpretación de datos**
        
        - **Porcentaje**: Proporción de la fruta en el SKU
        - **Contribución**: Impacto en costo por kg del producto
        - **SKUs usados**: Número de productos que usan cada fruta
        - **Eficiencia**: Factor de procesamiento de cada fruta
        """)

# -------- Información de navegación --------
st.markdown("---")

# Expander opcional para diagnóstico de session_state
with st.expander("🔎 Diagnóstico session_state", expanded=False):
    session_state_table()

st.info("💡 **Navegación**: Usa el menú lateral para volver a la página principal.")
st.info("💾 **Datos persistentes**: Los cambios se mantienen durante la sesión.")
st.info("📁 **Requisito**: Debes cargar datos en la página Home antes de usar el simulador.")
