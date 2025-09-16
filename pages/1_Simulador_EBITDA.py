"""
Simulador de EBITDA por SKU (USD/kg)
Página del simulador con filtros, overrides y análisis de rentabilidad.
"""

from altair import renderers
import streamlit as st
import pandas as pd
import numpy as np
import sys
import math
from pathlib import Path
import io
import streamlit.components.v1 as components
import json
from src.state import sync_filters_to_shared, sync_filters_from_shared
from src.dynamic_filters import DynamicFiltersWithList
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode, ColumnsAutoSizeMode, ExcelExportMode

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ===================== Utilidades =====================
def create_subtotal_row(df, position="bottom"):
    """Crea una fila de subtotales con manejo de valores None"""
    subtotal_row = {}
    
    # Llenar columnas de agrupación con "TOTAL" o mensaje genérico
    for col in ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Fruta", "Name"]:
        if col in df.columns:
            if col in ["Marca", "Cliente", "Especie", "Condicion", "Fruta", "Name", "SKU", "Descripcion"]:
                subtotal_row[col] = "TOTAL"
            else:
                subtotal_row[col] = ""  # Mensaje genérico en lugar de None
    
    # Calcular sumas para columnas numéricas
    numeric_cols = ["EBITDA (USD/kg)", "Costos Totales (USD/kg)", "KgEmbarcados", 
                    "MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)", "PrecioVenta (USD/kg)",
                    "Precio (USD/kg)", "Costo (USD/kg)", "Óptimo", "Ponderado",
                    "MMPP Total (USD/kg)", "MO Directa", "MO Indirecta", "MO Total",
                    "Materiales Directos", "Materiales Indirectos", "Materiales Total",
                    "Laboratorio", "Mantención", "Mantencion y Maquinaria" "Utilities", "Fletes Internos",
                    "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
                    "Servicios Generales", "Comex", "Guarda PT", "Almacenaje MMPP",
                    "Gastos Totales (USD/kg)", "Costos Directos", "Costos Indirectos"]
    
    for col in numeric_cols:
        try:
            if col == "KgEmbarcados":
                subtotal_row[col] = df["KgEmbarcados"].sum()
            elif col in df.columns:
                subtotal_row[col] = (df[col]*df["KgEmbarcados"]).sum()/df["KgEmbarcados"].sum()
        except:
            subtotal_row[col] = ""
    
    # Manejar columnas que pueden ser None (como EBITDA Pct)
    for col in ["EBITDA Pct"]:
        if col in df.columns:
            # Para porcentajes, calcular el promedio o mostrar mensaje genérico
            if df[col].notna().any():
                subtotal_row[col] = df[col].mean()
            else:
                subtotal_row[col] = ""  # Mensaje genérico en lugar de None
    
    return subtotal_row
    
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un DataFrame a formato seguro para Arrow/Streamlit.
    
    Args:
        df: DataFrame a convertir
        
    Returns:
        DataFrame con formato seguro para Arrow
    """
    out = df.copy()

    # 1) Asegurar nombres de columnas simples (no MultiIndex / no objetos)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" | ".join(map(str, lvl)) for lvl in out.columns]

    out.columns = [str(c) for c in out.columns]

    # 2) Convertir valores no serializables (tuplas, listas, dicts, arrays, sets) a strings JSON
    # EXCEPTO para la columna Especie que debe mantener las listas para ListColumn
    def _sanitize_val(x, preserve_lists=False):
        if preserve_lists and isinstance(x, list):
            return x  # Preservar listas para ListColumn
        elif isinstance(x, (tuple, list, dict, set, np.ndarray)):
            try:
                return json.dumps(x, default=str)
            except Exception:
                return str(x)
        return x

    for c in out.columns:
        # Si la serie es object, aplica saneo elemento a elemento
        if out[c].dtype == "object":
            # Evita cast innecesario si ya son strings/números/fechas
            sample = out[c].dropna().head(5).tolist()
            if any(isinstance(v, (tuple, list, dict, set, np.ndarray)) for v in sample):
                # Preservar listas solo para la columna Especie
                preserve_lists = (c == "Especie")
                out[c] = out[c].map(lambda x: _sanitize_val(x, preserve_lists=preserve_lists))
        # Opcional: convertir categorías a string
        if pd.api.types.is_categorical_dtype(out[c].dtype):
            out[c] = out[c].astype(str)

    # 3) Índice simple
    if isinstance(out.index, pd.MultiIndex):
        # Convertir MultiIndex a strings usando to_flat_index()
        out.index = [" | ".join(map(str, tup)) for tup in out.index.to_flat_index()]

    return out

# Importar con manejo de errores más robusto
try:
    # Intentar import desde src
    from src.data_io import build_detalle, REQ_SHEETS, columns_config, recalculate_totals, cargar_plan_2026
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
        export_escenario, validate_upload_file,
        apply_granel_filters, get_granel_filter_options, apply_granel_global_overrides,
        recalculate_granel_totals, apply_granel_universal_adjustments, calculate_granel_kpis,
        get_top_bottom_granel, create_granel_cost_chart, export_granel_escenario,
        sync_granel_changes_to_retail, get_data_for_download, get_mime_type, get_file_extension
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
    cargar_plan_2026(st.session_state["hist.file_bytes"])

# Filtrar SKUs sin costos totales (igual a 0) para análisis de EBITDA más preciso
# Guardar los excluidos en variable 'skus_excluidos' para mantenerlos disponibles
if df_base is not None and "Costos Totales (USD/kg)" in df_base.columns:
    original_count = len(df_base)
    # if st.session_state["sim.df_filtered"] is not None:
    #     df_base = st.session_state["sim.df_filtered"]
    # Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
    subproductos = df_base[df_base["Costos Totales (USD/kg)"] == 0].copy()
    sin_ventas = df_base[df_base["Comex"] == 0].copy()
    # Filtrar SKUs que no están en el plan 2026
    no_plan_2026 = df_base[~df_base["SKU-Cliente"].isin(st.session_state["sim.plan_2026"]["SKU-Cliente"])].copy()
    skus_excluidos = pd.concat([subproductos, sin_ventas, no_plan_2026])
    skus_excluidos = skus_excluidos.drop_duplicates(subset=["SKU-Cliente"], keep="first")
    df_base = df_base[~df_base["SKU-Cliente"].isin(skus_excluidos["SKU-Cliente"])].copy()
    # Quiero agregar columnas a SKU_Excluidos con el booleano de si es subproducto, sin ventas o no en plan 2026
    skus_excluidos["Subproducto"] = skus_excluidos["SKU-Cliente"].isin(subproductos["SKU-Cliente"])
    skus_excluidos["Sin Ventas"] = skus_excluidos["SKU-Cliente"].isin(sin_ventas["SKU-Cliente"])
    skus_excluidos["No en Plan 2026"] = skus_excluidos["SKU-Cliente"].isin(no_plan_2026["SKU-Cliente"])
    # Ordenar por SKU-Cliente
    skus_excluidos["SKU-Cliente"] = skus_excluidos["SKU-Cliente"].astype(int)
    skus_excluidos = skus_excluidos.set_index("SKU-Cliente").sort_index()
    filtered_count = len(df_base)
    skus_excluidos_count = len(skus_excluidos)
    
    if original_count > filtered_count:        
        # IMPORTANTE: Recalcular totales en los datos cargados para asegurar que EBITDA Pct esté correcto
        if "EBITDA Pct" in df_base.columns:
            df_base = recalculate_totals(df_base)
        # Mostrar información sobre subproductos excluidos
        with st.expander(f"📋 **SKUs excluidos** ({skus_excluidos_count} SKUs)", expanded=False):
            st.write("**¿Por qué se excluyen estos SKUs?**")
            st.write("Son SKUs sin ventas, con costos totales = 0, o que no están en el plan 2026, que no pueden generar EBITDA real y distorsionan el análisis financiero.")
            
            # Estadísticas de subproductos
            col1, col2, col3 = st.columns(3)
            # # Quiero mostrar los 3 tipos de excluidos (sin ventas, con costos totales = 0, no en plan 2026) de manera separada.
            # sin_ventas_counts = sin_ventas["Marca"].value_counts()
            # subproductos_counts = subproductos["Marca"].value_counts()
            # no_plan_2026_counts = no_plan_2026["Marca"].value_counts()
            # st.write("**Por Marca:**")
            # for marca, count in sin_ventas_counts.head(3).items():
            #     st.write(f"- {marca}: {count}")
            # for marca, count in subproductos_counts.head(3).items():
            #     st.write(f"- {marca}: {count}")
            # for marca, count in no_plan_2026_counts.head(3).items():
            #     st.write(f"- {marca}: {count}")
            
            with col1:
                st.metric("**Sin ventas:**", len(sin_ventas))
                if "Marca" in skus_excluidos.columns:
                    marca_counts = skus_excluidos["Marca"].value_counts()
                    st.write("**Por Marca:**")
                    for marca, count in marca_counts.head(3).items():
                        st.write(f"- {marca}: {count}")
            
            with col2:
                st.metric("**Con costos totales = 0:**", len(subproductos))
                if "Cliente" in skus_excluidos.columns:
                    cliente_counts = skus_excluidos["Cliente"].value_counts()
                    st.write("**Por Cliente:**")
                    for cliente, count in cliente_counts.head(3).items():
                        st.write(f"- {cliente}: {count}")
            
            with col3:
                st.metric("**No en plan 2026:**", len(no_plan_2026))
                if "Especie" in skus_excluidos.columns:
                    especie_counts = skus_excluidos["Especie"].value_counts()
                    st.write("**Por Especie:**")
                    for especie, count in especie_counts.head(3).items():
                        st.write(f"- {especie}: {count}")
            
            # Tabla completa de subproductos
            st.write("**Lista completa de subproductos excluidos:**")
            st.dataframe(
                skus_excluidos[["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Subproducto", "Sin Ventas", "No en Plan 2026"]],
                width='stretch',
                hide_index=True
            )
            
            # Botón de exportación con selector de formato
            col1, col2 = st.columns([1, 1])
            
            with col1:
                format_skus_excluidos = st.selectbox(
                    "Formato:",
                    options=["csv", "excel"],
                    format_func=lambda x: "CSV" if x == "csv" else "Excel",
                    help="Selecciona el formato de descarga",
                    key="skus_excluidos_format_1"
                )
            
            with col2:
                data_skus_excluidos = get_data_for_download(skus_excluidos, format_skus_excluidos)
                mime_type_skus_excluidos = get_mime_type(format_skus_excluidos)
                extension_skus_excluidos = get_file_extension(format_skus_excluidos)
                
            st.download_button(
                    label=f"📥 Descargar Lista Completa de SKUs excluidos ({format_skus_excluidos.upper()})",
                    data=data_skus_excluidos,
                    file_name=f"subproductos_excluidos_completo.{extension_skus_excluidos}",
                    mime=mime_type_skus_excluidos,
                width='stretch',
                key="download_skus_excluidos_sim_1"
            )

# Filtros Dinamicos de la libreria streamlit-dynamic-filters
st.sidebar.header("🔍 Filtros Dinámicos")
with st.sidebar.container():
    if "hist.filters" in st.session_state:
        st.session_state["hist.filters"] = sync_filters_from_shared(page="hist")
        active_filters = st.session_state["hist.filters"]
        active_count = sum(len(v) for v in active_filters.values() if v)
        if active_count > 0:
            if active_count == 1:
                st.sidebar.info(f"🔍 **{active_count} filtro activo**")
            else:
                st.sidebar.info(f"🔍 **{active_count} filtros activos**")
            for logical, values in active_filters.items():
                if values:
                    st.sidebar.write(f"**{logical}**: {', '.join(str(values)[:3])}{'...' if len(values) > 3 else ''}")
    dynamic_filters = DynamicFiltersWithList(df=df_base, filters=['Marca', 'Cliente', 'Especie', 'Condicion', 'SKU'], filters_name='hist.filters')
    dynamic_filters.check_state()
    dynamic_filters.display_filters(location='sidebar')
    df_filtered = dynamic_filters.filter_df()

with st.sidebar.container():
    st.button("Resetear Filtros", on_click=dynamic_filters.reset_filters)

sync_filters_to_shared(page="hist", filters=st.session_state["hist.filters"])

# Orden por SKU-Cliente si existe y sin índice
sku_cliente_col = "SKU-Cliente"
if sku_cliente_col in df_filtered.columns:
    df_filtered = df_filtered.sort_values([sku_cliente_col]).reset_index(drop=True)
else:
    df_filtered = df_filtered.reset_index(drop=True)

# Guardar resultado filtrado en sim.df_filtered
st.session_state["sim.df_filtered"] = df_filtered.copy()

# Los filtros se sincronizan automáticamente via on_change de los widgets

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
tab_granel, tab_sku, tab_precio_frutas, tab_receta = st.tabs(["🏭 Granel (Fruta)", "📊 Retail (SKU)", "🍓 Precio Fruta", "📖 Receta"])

with tab_sku:
    tab_plan, tab_optimos, tab_comparacion = st.tabs(["🔍 Plan 2026", "🏆 Óptimos", "⚖️ Comparación"])
    with tab_plan:
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
            # Asegurar que solo se muestren los SKUs filtrados (usar SKU-Cliente para consistencia)
            filtered_skus = st.session_state["sim.df_filtered"]["SKU-Cliente"].tolist()
            df_current = df_current[df_current["SKU-Cliente"].isin(filtered_skus)].copy()
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
            
            adj_columns = ["Almacenaje MMPP", "MO Directa", "MO Indirecta",
        "Materiales Directos", "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", "Utilities",
        "Fletes Internos", "Comex", "Guarda PT", "PrecioVenta (USD/kg)"]
            
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
                        # Usar text_input para evitar el bug de number_input con valores negativos
                        input_text = st.text_input(
                            "Nuevo valor:",
                            value="",
                            help="Nuevo valor en USD/kg (ej: -0.123, 0.456)",
                            key=f"adjustment_value_{selected_cost}",
                            placeholder="0.000"
                        )
                        
                        # Validar y convertir el valor
                        try:
                            adjustment_value = float(input_text)
                            if adjustment_value < -10.0 or adjustment_value > 100.0:
                                st.error("⚠️ El valor debe estar entre -10.0 y 100.0 USD/kg")
                                adjustment_value = 0.0
                        except ValueError:
                            if input_text.strip() == "":
                                adjustment_value = 0.0
                            else:
                                st.error("⚠️ Por favor, ingrese un número válido (ej: -0.123)")
                                adjustment_value = 0.0
            
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
            
            # ===================== Opciones de Visualización =====================
            st.subheader("📊 Opciones de Visualización")
            col_view1, col_view2 = st.columns([1, 2])
            
            with col_view1:
                show_subtotals_at_top = st.checkbox(
                    "Subtotales al inicio",
                    value=st.session_state.get("sim.show_subtotals_at_top", False),
                    help="Mostrar fila de subtotales al inicio de la tabla",
                    key="sim_show_subtotals_at_top"
                )
                if show_subtotals_at_top != st.session_state.get("sim.show_subtotals_at_top", False):
                    st.session_state["sim.show_subtotals_at_top"] = show_subtotals_at_top
                    st.rerun()
            
            # Preparar datos para la tabla editable usando sim.df (que incluye ajustes universales)
            # Obtener datos de simulación si están disponibles en la sesión
            if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                # Usar sim.df que ya incluye los ajustes universales aplicados
                sim_data = st.session_state["sim.df"].copy()
                # Filtrar por SKUs actuales (usar SKU-Cliente para consistencia)
                filtered_skus = st.session_state["sim.df_filtered"]["SKU-Cliente"].tolist()
                
                # Identificar columnas de costos (excluyendo dimensiones y totales)
                # Nota: SKU-Cliente se incluye en dimension_cols para el procesamiento pero se oculta en la tabla
                dimension_cols = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]

                orden_cols = ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)", "MMPP Total (USD/kg)","MO Directa",
                        "MO Indirecta","MO Total","Materiales Directos","Materiales Indirectos","Materiales Total",
                        "Laboratorio","Mantención","Utilities","Fletes Internos","Retail Costos Directos (USD/kg)",
                        "Retail Costos Indirectos (USD/kg)","Servicios Generales","Comex","Guarda PT","Almacenaje MMPP",
                        "Gastos Totales (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)",
                        "EBITDA Pct","KgEmbarcados"]
        
                # Mover columnas dimensionales al inicio
                display_order = dimension_cols + orden_cols
                available_display_cols = [col for col in display_order if col in sim_data.columns]
                
                # Crear DataFrame para edición
                df_edit = sim_data[available_display_cols].copy()
                df_edit = df_edit[df_edit["SKU-Cliente"].isin(filtered_skus)]
                
                # No aplicar subtotales manuales - usar tabla dinámica en su lugar
                        
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
                total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)", "Costos Totales (USD/kg)",
                "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)", "KgEmbarcados"]
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
                
                # Aplicar estilos especiales a filas de subtotales (código legacy removido)
                
                # El DataFrame ya tiene el índice establecido, solo aplicar estilos
                df_edit_final = df_edit_styled
                
                # Crear fila de subtotales y preparar una vista solo de métricas (ocultar dimensiones)
                subtotal_row = create_subtotal_row(df_edit)
                subtotal_df = pd.DataFrame([subtotal_row])
                # Columnas numéricas presentes en df_edit
                try:
                    numeric_cols = [c for c in df_edit.columns if pd.api.types.is_numeric_dtype(df_edit[c])]
                except Exception:
                    numeric_cols = [c for c in df_edit.columns if c not in ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion"]]
                subtotal_numeric = subtotal_df[[c for c in numeric_cols if c in subtotal_df.columns]].copy()
                # Estilo del subtotal
                sty_sub = subtotal_numeric.style.set_properties(**{"font-weight":"bold","background-color":"#e8f4fd","border-top":"2px solid #1f77b4"})
                
                # Header más alto para permitir 2 líneas y mostrar tablas
                try:
                    from src.data_io import inject_streamlit_dataframe_css
                    inject_streamlit_dataframe_css(header_height=64)
                except Exception:
                    pass

                # Subtotal arriba (solo métricas) si corresponde
                if show_subtotals_at_top:
                    st.caption("Subtotal (ponderado por KgEmbarcados)")
                    st.dataframe(sty_sub, column_config=editable_columns, width='stretch', hide_index=True)

                # Tabla principal
                edited_df = st.dataframe(
                    df_edit_final,
                    column_config=editable_columns,
                    width='stretch',
                    height="auto",
                    key="data_editor_detalle",
                    hide_index=True
                )

                # Subtotal abajo (solo métricas) si corresponde
                if not show_subtotals_at_top:
                    st.caption("Subtotal (ponderado por KgEmbarcados)")
                    st.dataframe(sty_sub, column_config=editable_columns, width='stretch', hide_index=True)
                
                # # Mostrar métricas de subtotales
                # col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                
                # with col_metrics1:
                #     if "EBITDA (USD/kg)" in df_edit.columns:
                #         total_ebitda = df_edit["EBITDA (USD/kg)"].sum()
                #         st.metric("EBITDA Total (USD/kg)", f"{total_ebitda:,.2f}")
                
                # with col_metrics2:
                #     if "Costos Totales (USD/kg)" in df_edit.columns:
                #         total_costos = df_edit["Costos Totales (USD/kg)"].sum()
                #         st.metric("Costos Totales (USD/kg)", f"{total_costos:,.2f}")
                
                # with col_metrics3:
                #     if "KgEmbarcados" in df_edit.columns:
                #         total_kg = df_edit["KgEmbarcados"].sum()
                #         st.metric("Kg Embarcados Total", f"{total_kg:,.0f}")
                
                # Detectar cambios y recalcular totales AUTOMÁTICAMENTE
                # if not edited_df.equals(df_edit_original):
                #     st.info("🔍 Cambios detectados en la tabla editable")
                    
                #     # Restaurar índice para procesamiento
                #     edited_df_reset = edited_df.reset_index()
                    
                #     # Guardar historial de cambios ANTES de procesar
                #     changes_detected = 0
                    
                #     # Comparar contra hist.df (valores originales) para detectar cambios
                #     if "hist.df" in st.session_state:
                #         hist_df = st.session_state["hist.df"]
                        
                #         # Convertir filtered_skus a strings para que coincida con los DataFrames
                #         filtered_skus_str = [str(sku) for sku in filtered_skus]
                        
                #         # Buscar cambios por SKU comparando contra valores originales
                #         for sku in filtered_skus_str:
                #             # Buscar el SKU en hist.df (valores originales)
                #             mask_hist = hist_df["SKU"] == sku
                #             mask_edited = edited_df_reset["SKU"] == sku
                            
                #             if mask_hist.any() and mask_edited.any():
                #                 # Obtener las filas correspondientes
                #                 original_row = hist_df[mask_hist].iloc[0]
                #                 edited_row = edited_df_reset[mask_edited].iloc[0]
                                
                #                 # Comparar columnas numéricas (excluyendo dimensiones)
                #                 for col in edited_df_reset.columns:
                #                     if col not in ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]:
                #                         try:
                #                             # Verificar que la columna existe en ambos DataFrames
                #                             if col in original_row and col in edited_row:
                #                                 original_value = original_row[col]
                #                                 edited_value = edited_row[col]
                                                
                #                                 # Si hay cambio, guardar en historial
                #                                 if abs(original_value - edited_value) > 1e-6:  # Tolerancia para floats
                #                                     save_edit_history(sku, col, original_value, edited_value)
                #                                     changes_detected += 1
                #                             else:
                #                                 st.warning(f"⚠️ Columna {col} no encontrada en uno de los DataFrames")
                #                         except (IndexError, KeyError, TypeError) as e:
                #                             st.warning(f"⚠️ Error comparando {sku} - {col}: {e}")
                #                             continue
                #             else:
                #                 st.warning(f"⚠️ SKU {sku} no encontrado en uno de los DataFrames")
                #     else:
                #         st.warning("⚠️ No hay datos históricos disponibles para comparar cambios")
                    
                #     if changes_detected > 0:
                #         st.success(f"✅ {changes_detected} cambios detectados y guardados en historial")
                        
                #         # Validar y corregir signos antes de procesar
                #         edited_df_reset = validate_and_correct_signs(edited_df_reset)
                        
                #         # IMPORTANTE: Recalcular totales directamente en edited_df_reset
                #         edited_df_recalculated = recalculate_totals(edited_df_reset)
                        
                #         # Actualizar solo sim.df (NO modificar hist.df)
                #         if "sim.df" in st.session_state:
                #             st.session_state["sim.df"] = edited_df_recalculated.copy()
                #             st.session_state["sim.dirty"] = True
                        
                #         st.success("✅ EBITDA recalculado automáticamente")
                        
                #         # Forzar actualización de la vista automáticamente
                #         st.rerun()
                #     else:
                #         st.warning("⚠️ No se detectaron cambios específicos")
                # else:
                #     st.info("ℹ️ No hay cambios en la tabla editable")
                
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
                    
                    # Botón de exportación con selector de formato
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        format_subproductos = st.selectbox(
                            "Formato:",
                            options=["csv", "excel"],
                            format_func=lambda x: "CSV" if x == "csv" else "Excel",
                            help="Selecciona el formato de descarga",
                            key="subproductos_format_2"
                        )
                    
                    with col2:
                        data_subproductos = get_data_for_download(subproductos, format_subproductos)
                        mime_type_subproductos = get_mime_type(format_subproductos)
                        extension_subproductos = get_file_extension(format_subproductos)
                        
                    st.download_button(
                            label=f"📥 Descargar Lista Completa de Subproductos ({format_subproductos.upper()})",
                            data=data_subproductos,
                            file_name=f"subproductos_excluidos_completo.{extension_subproductos}",
                            mime=mime_type_subproductos,
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
        st.header(" Top 10 y Bottom 10 SKUs por EBITDA")

        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 SKUs")
                top_skus, _ = get_top_bottom_skus(df_current, 10)
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
                        width='stretch',
                        hide_index=True
                    )
                else:
                    st.info("No hay datos para mostrar")
            
            with col2:
                st.subheader("Bottom 10 SKUs")
                _, bottom_skus = get_top_bottom_skus(df_current, 10)
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
                        width='stretch',
                        hide_index=True
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

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            filename_prefix = st.text_input(
                "Prefijo del archivo:",
                value="escenario_ebitda",
                help="Nombre base para el archivo de exportación"
            )

        with col2:
            export_format = st.selectbox(
                "Formato:",
                options=["csv", "excel"],
                format_func=lambda x: "CSV" if x == "csv" else "Excel",
                help="Selecciona el formato de exportación"
            )

        with col3:
            if st.button("📥 Exportar", type="primary"):
                try:
                    # Generar datos para descarga
                    data = get_data_for_download(df_current, export_format)
                    mime_type = get_mime_type(export_format)
                    extension = get_file_extension(export_format)
                    
                    # Generar nombre de archivo con timestamp
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
                    filename = f"{filename_prefix}_{timestamp}.{extension}"
                    
                    # Botón de descarga
                    st.download_button(
                        label=f"⬇️ Descargar {export_format.upper()}",
                        data=data,
                        file_name=filename,
                        mime=mime_type,
                        key="download_escenario"
                    )
                    
                    st.success(f"✅ Escenario listo para descarga en formato {export_format.upper()}")
                    
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
        
        ### 🌾 **Simulador de Granel**
        
        El simulador de granel te permite:
        
        1. **Filtrar frutas** por tipo específico
        2. **Aplicar overrides globales** con cambios porcentuales en costos de granel
        3. **Ajustes universales** por tipo de costo específico
        4. **Analizar costos** de MO, Materiales, Laboratorio, etc.
        5. **Visualizar resultados** con gráficos de costos por fruta
        6. **Exportar escenarios** de granel para análisis posterior
        7. **🔄 Sincronización automática** con el simulador de retail
        
        ### 🔄 **Sincronización Granel ↔ Retail**
        
        Cuando editas costos de granel, el sistema automáticamente:
        - Recalcula el "Proceso Granel (USD/kg)" para cada SKU
        - Actualiza los totales en el simulador de retail
        - Mantiene la consistencia entre ambos simuladores
        - Requiere datos de recetas (RECETA_SKU) y frutas (INFO_FRUTA) para funcionar
        
        ### 🔧 **Cómo usar el simulador de granel**
        
        1. **Carga datos** en la página Home con la hoja 'FACT_GRANEL_POND'
        2. **Navega a la pestaña Granel** en el simulador
        3. **Aplica filtros** por fruta si deseas enfocar el análisis
        4. **Configura overrides globales** para cambios porcentuales masivos
        5. **Usa ajustes universales** para modificar costos específicos
        6. **Analiza KPIs** y gráficos para tomar decisiones
        7. **Exporta el escenario** para compartir o analizar
        """)
    with tab_optimos:
        st.header("🏆 Óptimos (no editable)")
        # Obtener datos optimos si están disponibles en la sesión
        if "hist.df_optimo" in st.session_state and st.session_state["hist.df_optimo"] is not None:
            # Usar hist.optimos que ya incluye los ajustes universales aplicados
            optimos_data = st.session_state["hist.df_optimo"].copy()
            # Filtrar por SKUs actuales
            filtered_skus = st.session_state["sim.df_filtered"]["SKU-Cliente"].tolist()
            
            # Identificar columnas de costos (excluyendo dimensiones y totales)
            # Nota: SKU-Cliente se incluye en dimension_cols para el procesamiento pero se oculta en la tabla
            dimension_cols = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]
            
            orden_cols = ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)", "MMPP Total (USD/kg)","MO Directa",
                        "MO Indirecta","MO Total","Materiales Directos","Materiales Indirectos","Materiales Total",
                        "Laboratorio","Mantención","Utilities","Fletes Internos","Retail Costos Directos (USD/kg)",
                        "Retail Costos Indirectos (USD/kg)","Servicios Generales","Comex","Guarda PT","Almacenaje MMPP",
                        "Gastos Totales (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
        
            # Mover columnas dimensionales al inicio
            display_order_optimos = dimension_cols + orden_cols
            available_display_cols_optimos = [col for col in display_order_optimos if col in optimos_data.columns]
            
            # Crear DataFrame para edición
            df_optimos = optimos_data[available_display_cols_optimos].copy()
            df_optimos = df_optimos[df_optimos["SKU-Cliente"].isin(filtered_skus)]
                    
            # Configurar columnas editables (solo costos individuales, no totales)
            editable_columns = columns_config(editable=True)
            # Aplicar estilos antes de mostrar la tabla editable (igual que en datos históricos)
            df_optimos_styled = df_optimos.copy()
            
            # ESTABLECER EL ÍNDICE ANTES de aplicar estilos
            df_optimos_styled = df_optimos_styled.set_index("SKU-Cliente")
            
            # IMPORTANTE: Guardar una copia del DataFrame original (con índice) ANTES de convertir a Styler
            df_optimos_original = df_optimos_styled.copy()
            
            # Aplicar formato numérico ANTES de convertir a Styler
            fmt_cols = {}
            for c in df_optimos_styled.columns:
                if c not in ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]:
                    if "Pct" in c or "Porcentaje" in c:
                        fmt_cols[c] = "{:.1%}"  # Formato de porcentaje
                    elif np.issubdtype(df_optimos_styled[c].dtype, np.number):
                        fmt_cols[c] = "{:.3f}"   # Formato numérico
            
            # Aplicar formato numérico al DataFrame
            if fmt_cols:
                df_optimos_styled = df_optimos_styled.style.format(fmt_cols)
            
            # Aplicar negritas a las columnas de totales
            total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)", "Costos Totales (USD/kg)",
            "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)"]
            existing_total_columns = [col for col in total_columns if col in df_optimos.columns]
            
            if existing_total_columns:
                df_optimos_styled = df_optimos_styled.set_properties(
                    subset=existing_total_columns,
                    **{"font-weight": "bold", "background-color": "#f8f9fa"}
                )
            
            # Aplicar estilos a columnas EBITDA
            ebitda_columns = ["EBITDA (USD/kg)", "EBITDA Pct"]
            existing_ebitda_columns = [col for col in ebitda_columns if col in df_optimos.columns]
            
            if existing_ebitda_columns:
                df_optimos_styled = df_optimos_styled.set_properties(
                    subset=existing_ebitda_columns,
                    **{"font-weight": "bold", "background-color": "#fff7ed"}
                )
            
            # Crear fila de subtotales
            subtotal_row = create_subtotal_row(df_optimos)
            subtotal_df = pd.DataFrame([subtotal_row])
            
            # Concatenar subtotales al final (por defecto)
            df_optimos_with_subtotals = pd.concat([df_optimos, subtotal_df], ignore_index=True)
            subtotal_position = len(df_optimos)  # Última fila
            
            # Aplicar estilos a la tabla con subtotales
            df_optimos_styled = df_optimos_with_subtotals.style
            
            # Aplicar negritas a las columnas de totales
            total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)",
            "Costos Totales (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
            "KgEmbarcados"]
            existing_total_columns = [col for col in total_columns if col in df_optimos_with_subtotals.columns]

            if existing_total_columns:
                df_optimos_styled = df_optimos_styled.set_properties(
                    subset=existing_total_columns,
                    **{"font-weight": "bold", "background-color": "#f8f9fa"}
                )

            # Aplicar estilos a columnas EBITDA
            ebitda_columns = ["EBITDA (USD/kg)", "EBITDA Pct"]
            existing_ebitda_columns = [col for col in ebitda_columns if col in df_optimos_with_subtotals.columns]
            
            if existing_ebitda_columns:
                df_optimos_styled = df_optimos_styled.set_properties(
                    subset=existing_ebitda_columns,
                    **{"font-weight": "bold", "background-color": "#fff7ed"}
                )
            
            # Aplicar estilo especial a la fila de subtotales
            from pandas import IndexSlice as idx
            df_optimos_styled = df_optimos_styled.set_properties(
                subset=idx[subtotal_position, :],  # Fila de subtotales, todas las columnas
                **{
                    "font-weight": "bold",
                    "background-color": "#e8f4fd",
                    "border-top": "2px solid #1f77b4",
                },
            )
            
            # Mostrar tabla con subtotales
            edited_df = st.dataframe(
                df_optimos_styled,
                column_config=editable_columns,
                width='stretch',
                height="auto",
                key="data_editor_detalle",
                hide_index=True
            )
            
            # # Mostrar métricas de subtotales
            # col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            # with col_metrics1:
            #     if "EBITDA (USD/kg)" in df_optimos.columns:
            #         total_ebitda = df_optimos["EBITDA (USD/kg)"].sum()
            #         st.metric("EBITDA Total (USD/kg)", f"{total_ebitda:,.2f}")
            
            # with col_metrics2:
            #     if "Costos Totales (USD/kg)" in df_optimos.columns:
            #         total_costos = df_optimos["Costos Totales (USD/kg)"].sum()
            #         st.metric("Costos Totales (USD/kg)", f"{total_costos:,.2f}")
            
            # with col_metrics3:
            #     if "KgEmbarcados" in df_optimos.columns:
            #         total_kg = df_optimos["KgEmbarcados"].sum()
            #         st.metric("Kg Embarcados Total", f"{total_kg:,.0f}")

# ====== GRANEL ======
with tab_granel:
    # ------------------------------------------------------
    # Helpers (solo para esta tab; usan tus funciones de src/simulator.py)
    # ------------------------------------------------------
    def _ensure_sim_df_from_hist():
        """Inicializa sim.granel_df desde el histórico si no existe (sin mutar el histórico)."""
        if "sim.granel_df" not in st.session_state or st.session_state["sim.granel_df"] is None:
            st.session_state["sim.granel_df"] = st.session_state["hist.granel_ponderado"].copy()
            st.session_state["sim.granel_df"] = recalculate_granel_totals(st.session_state["sim.granel_df"])

    def _cost_editable_columns(df: pd.DataFrame) -> list:
        """Columnas de costos editables manualmente (excluye IDs, títulos y totales)."""
        lock = {"Fruta_id", "Fruta", "Precio", "Rendimiento", "Precio Efectivo", "Costos Directos", "Costos Indirectos"}
        return [c for c in df.columns if c not in lock and not c.endswith("Total")]

    def _merge_back_sim(sim_df: pd.DataFrame, edited_view: pd.DataFrame, pk="Fruta_id") -> pd.DataFrame:
        """Escribe de vuelta SOLO lo editado (columnas editables) en sim.granel_df, recalculando totales."""
        editable = [c for c in _cost_editable_columns(edited_view) if c in sim_df.columns]
        base = sim_df.copy()
        # Trae solo pk + columnas editables desde la vista
        left = base.merge(edited_view[[pk] + editable], on=pk, how="left", suffixes=("", "_new"))
        for c in editable:
            nc = f"{c}_new"
            if nc in left.columns:
                left[c] = left[nc].combine_first(left[c])
                left.drop(columns=[nc], inplace=True)
        left = recalculate_granel_totals(left)
        return left

    def _sync_retail_using_hist_copy(overrides: dict):
        """
        Sincroniza retail usando:
        - COPIA del histórico + overrides universales aplicados al vuelo
        (sin mutar el histórico), y recalcula Proceso Granel en retail.
        """
        receta_df = st.session_state.get("fruta.receta_df")
        info_df   = st.session_state.get("fruta.plan_2026")
        retail_df = st.session_state.get("sim.df")
        if receta_df is None or info_df is None or retail_df is None:
            return False, "Faltan datos de recetas, info de frutas o sim.df"

        hist = st.session_state["hist.granel_ponderado"]  # inmutable
        hist_for_sync = apply_granel_universal_adjustments(hist, overrides or {})
        # (No usamos ediciones manuales aquí para evitar mezclar filtros; si quisieras
        # también reflejar ediciones locales, podrías fusionar columnas editadas desde sim.granel_df)
        try:
            st.session_state["sim.df"] = sync_granel_changes_to_retail(
                hist_for_sync, receta_df, info_df, retail_df
            )
            st.session_state["sim.dirty"] = True
            return True, None
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------
    # Tabs internas
    # ------------------------------------------------------
    tab_granel_real, tab_granel_optimos = st.tabs(["🔍 Real", "🏆 Óptimos"])

    # ======================================================
    # TAB: REAL
    # ======================================================
    with tab_granel_real:
        # --------- Carga de datos base ---------
        granel_hist = st.session_state.get("hist.granel_ponderado")
        if granel_hist is None or granel_hist.empty:
            st.error("❌ **No hay datos de granel disponibles**")
            st.info("💡 **Para usar el simulador de granel, primero debes:**\n\n1. 📁 Ir a **Inicio**\n2. 📤 Cargar Excel con la hoja **FACT_GRANEL_POND**\n3. 🔄 Regresar al simulador")
            if st.button("Ir a Inicio", type="primary"):
                st.switch_page("Inicio.py")
            st.stop()

        # Asegura escenario sim inicial (NUNCA tocar histórico)
        _ensure_sim_df_from_hist()

        # --------- Filtros ---------
        st.subheader("🔍 Filtros de Granel")
        filter_options = get_granel_filter_options(granel_hist)
        col1, col2 = st.columns(2)
        with col1:
            if "Fruta" in filter_options:
                frutas_seleccionadas = st.multiselect(
                    "Frutas:",
                    options=filter_options["Fruta"],
                    default=[],
                    help="Selecciona las frutas a incluir en el análisis"
                )
            else:
                frutas_seleccionadas = []
        with col2:
            st.write("")

        # Mapear las frutas seleccionadas a sus ids correspondientes
        ids_seleccionadas = []
        if "Fruta" in filter_options and "id" in filter_options:
            # Crear un mapeo directo de nombre de fruta a ID
            fruta_to_id_map = {}
            for i, nombre in enumerate(filter_options["Fruta"]):
                if i < len(filter_options["id"]):
                    fruta_to_id_map[nombre] = filter_options["id"][i]
            
            # Mapear las frutas seleccionadas a sus IDs
            for fruta in frutas_seleccionadas:
                if fruta in fruta_to_id_map:
                    ids_seleccionadas.append(fruta_to_id_map[fruta])

        # Mostrar SIEMPRE el sim (no el histórico), filtrado
        sim_df = st.session_state["sim.granel_df"].copy()
        granel_filtrado = apply_granel_filters(sim_df, ids_seleccionadas)

        # --------- Ajustes universales ---------
        st.subheader("⚙️ Ajustes Universales por Costo de Granel")

        # Columnas de costos disponibles (en el SIM filtrado para UX, se aplican globalmente)
        cost_columns_granel = _cost_editable_columns(sim_df)
        if cost_columns_granel:
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                selected_cost_granel = st.selectbox(
                    "Seleccionar costo a ajustar:",
                    options=cost_columns_granel,
                    help="Ajuste universal sobre esta columna"
                )
            with c2:
                adjustment_type_granel = st.selectbox(
                    "Tipo de ajuste:",
                    options=["Porcentaje (%)", "Dólares por kg (USD/kg)"],
                    help="Aplicar % o fijar un nuevo valor absoluto"
                )
            with c3:
                if adjustment_type_granel == "Porcentaje (%)":
                    adjustment_value_granel = st.number_input(
                        "Valor del ajuste:",
                        key=f"adjustment_value_granel_{selected_cost_granel}",
                        min_value=-100.0, max_value=1000.0, value=0.0,
                        step=0.5, format="%.1f",
                        help="Porcentaje de cambio (-100 a +1000)"
                    )
                else:
                    # Usar text_input para evitar el bug de number_input con valores negativos
                    input_text = st.text_input(
                        "Nuevo valor:",
                        value="",
                        help="Nuevo valor en USD/kg (ej: -0.123, 0.456)",
                        key=f"adjustment_value_granel_dollars_{selected_cost_granel}",
                        placeholder="0.000"
                    )
                    
                    # Validar y convertir el valor
                    try:
                        adjustment_value_granel = float(input_text)
                        if adjustment_value_granel < -10.0 or adjustment_value_granel > 100.0:
                            st.error("⚠️ El valor debe estar entre -10.0 y 100.0 USD/kg")
                            adjustment_value_granel = 0.0
                    except ValueError:
                        if input_text.strip() == "":
                            adjustment_value_granel = 0.0
                        else:
                            st.error("⚠️ Por favor, ingrese un número válido (ej: -0.123)")
                            adjustment_value_granel = 0.0
            with c4:
                if st.button("Aplicar Ajuste Granel", type="primary"):
                    sim_snapshot_push()

                    # GUARDAR VALORES ORIGINALES ANTES DE APLICAR AJUSTES
                    if "sim.granel_original_values" not in st.session_state:
                        st.session_state["sim.granel_original_values"] = {}
                    
                    # Guardar valores originales solo si es la primera vez que se aplica este ajuste
                    if selected_cost_granel not in st.session_state["sim.granel_original_values"]:
                        original_values = {}
                        # Usar sim.granel_df si existe, sino usar hist.granel_ponderado
                        source_df = st.session_state.get("sim.granel_df", st.session_state["hist.granel_ponderado"])
                        for fruta_id in source_df["Fruta_id"]:
                            mask = source_df["Fruta_id"] == fruta_id
                            if mask.any() and selected_cost_granel in source_df.columns:
                                original_values[fruta_id] = source_df.loc[mask, selected_cost_granel].iloc[0]
                        st.session_state["sim.granel_original_values"][selected_cost_granel] = original_values

                    st.session_state.setdefault("sim.granel_overrides_row", {})
                    st.session_state["sim.granel_overrides_row"][selected_cost_granel] = {
                        "type": "percentage" if adjustment_type_granel == "Porcentaje (%)" else "dollars",
                        "value": float(adjustment_value_granel),
                        "timestamp": pd.Timestamp.now()
                    }

                    # Recalcular el ESCENARIO completo desde histórico + overrides (histórico intocable)
                    base = st.session_state["hist.granel_ponderado"].copy()
                    base = apply_granel_universal_adjustments(base, st.session_state["sim.granel_overrides_row"])
                    st.session_state["sim.granel_df"] = recalculate_granel_totals(base)
                    
                    # DEBUG: Verificar que los valores se aplicaron correctamente
                    if selected_cost_granel in st.session_state["sim.granel_df"].columns:
                        st.write(f"🔍 DEBUG: Valores de {selected_cost_granel} después de aplicar ajuste:")
                        st.write(st.session_state["sim.granel_df"][selected_cost_granel].head().tolist())

                    # Sync retail con COPIA del histórico + overrides
                    ok, err = _sync_retail_using_hist_copy(st.session_state["sim.granel_overrides_row"])
                    if ok:
                        st.success(f"✅ Ajuste aplicado y sincronizado: {selected_cost_granel}")
                    else:
                        st.warning(f"⚠️ Ajuste aplicado en granel; retail no sincronizado: {err}")
                    st.rerun()

        # Listado y gestión de overrides activos
        if st.session_state.get("sim.granel_overrides_row"):
            st.subheader("Ajustes Universales Activos")
            for cost_column, adj in st.session_state["sim.granel_overrides_row"].items():
                tipo = "Porcentaje" if adj["type"] == "percentage" else "Dólares"
                valor = f"{adj['value']:+.1f}%" if adj["type"] == "percentage" else f"{adj['value']:+.3f} USD/kg"
                ts = adj["timestamp"].strftime("%H:%M:%S") if isinstance(adj.get("timestamp"), pd.Timestamp) else "—"
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.write(f"**{cost_column}**: {valor}")
                with c2:
                    st.write(f"**Tipo**: {tipo}")
                with c3:
                    st.write(f"**Aplicado**: {ts}")
                with c4:
                    if st.button("🗑️", key=f"del_ovr_{cost_column}", help=f"Eliminar ajuste de {cost_column}"):
                        sim_snapshot_push()
                        
                        # RESTAURAR VALORES ORIGINALES
                        if "sim.granel_original_values" in st.session_state and cost_column in st.session_state["sim.granel_original_values"]:
                            original_values = st.session_state["sim.granel_original_values"][cost_column]
                            
                            # CORREGIDO: NO modificar hist.granel_ponderado (debe ser inmutable)
                            # En su lugar, recalcular sim.granel_df desde histórico limpio
                            
                            # Eliminar valores originales guardados
                            del st.session_state["sim.granel_original_values"][cost_column]
                        
                        # Eliminar el ajuste
                        del st.session_state["sim.granel_overrides_row"][cost_column]

                        # Recalcular escenario desde histórico limpio + overrides restantes
                        base = st.session_state["hist.granel_ponderado"].copy()
                        base = apply_granel_universal_adjustments(base, st.session_state["sim.granel_overrides_row"])
                        st.session_state["sim.granel_df"] = recalculate_granel_totals(base)
                        
                        # DEBUG: Verificar que los valores se restauraron correctamente
                        if cost_column in st.session_state["sim.granel_df"].columns:
                            st.write(f"🔍 DEBUG: Valores de {cost_column} después de eliminar ajuste:")
                            st.write(st.session_state["sim.granel_df"][cost_column].head().tolist())

                        ok, err = _sync_retail_using_hist_copy(st.session_state["sim.granel_overrides_row"])
                        if ok:
                            st.success(f"✅ Ajuste eliminado y sincronizado: {cost_column}")
                        else:
                            st.warning(f"⚠️ Ajuste eliminado; retail no sincronizado: {err}")
                        st.rerun()

            if st.button("Limpiar todos los ajustes", type="secondary"):
                sim_snapshot_push()
                
                # CORREGIDO: NO modificar hist.granel_ponderado (debe ser inmutable)
                # Simplemente limpiar valores originales guardados
                if "sim.granel_original_values" in st.session_state:
                    st.session_state["sim.granel_original_values"] = {}
                
                # Limpiar overrides
                st.session_state["sim.granel_overrides_row"] = {}
                st.session_state["sim.granel_df"] = recalculate_granel_totals(st.session_state["hist.granel_ponderado"].copy())

                ok, err = _sync_retail_using_hist_copy({})
                if ok:
                    st.success("✅ Overrides limpiados y retail sincronizado")
                else:
                    st.warning(f"⚠️ Overrides limpiados; retail no sincronizado: {err}")
                st.rerun()

        # --------- Tabla editable (sobre el SIM filtrado) ---------
        st.subheader("✏️ Editar Costos de Granel (editable)")
        editable_view = granel_filtrado.copy()

        # Casting numérico seguro para estilos/ediciones
        numeric_cols = [c for c in editable_view.columns if c not in ["Fruta_id", "Name"]]
        for c in numeric_cols:
            editable_view[c] = pd.to_numeric(editable_view[c], errors="coerce")

        # Orden preferido (mostrar si existen)
        order_cols = ["Fruta_id", "Name", "Precio", "Rendimiento", "Precio Efectivo",
                      "MO Directa", "MO Indirecta", "MO Total",
                      "Materiales Directos", "Materiales Indirectos", "Materiales Total",
                      "Laboratorio", "Mantencion y Maquinaria",
                      "Costos Directos", "Costos Indirectos", "Servicios Generales", "Proceso Granel (USD/kg)"]
        available_cols = [c for c in order_cols if c in editable_view.columns]
        # Añadir columnas que no estaban en el orden
        editable_view = editable_view[available_cols]
        editable_view = editable_view.set_index("Fruta_id").sort_index()
        editable_view = editable_view.sort_values(by="Proceso Granel (USD/kg)")
        total_columns = ["MO Total", "Materiales Total", "Costos Directos", "Costos Indirectos"]
        editable_columns = editable_view.style
        if total_columns:
                editable_columns = editable_columns.set_properties(
                    subset=total_columns,
                    **{"font-weight": "bold", "background-color": "#f8f9fa"}
                )
        if "Proceso Granel (USD/kg)" in editable_columns.columns:
            editable_columns = editable_columns.set_properties(
                subset=["Proceso Granel (USD/kg)"],
                    **{"font-weight": "bold", "background-color": "#fff7ed"}
                )
        config = {}
        for c in editable_columns.columns:
            if c not in ["Fruta_id", "Name", "Proceso Granel (USD/kg)", "Precio Efectivo"]:
                config[c] = st.column_config.NumberColumn(
                    c,
                    format="%.3f"
                )
            else:
                if c == "Proceso Granel (USD/kg)":
                    config[c] = st.column_config.NumberColumn(
                    "Proceso Granel (USD/kg)",
                    disabled=True,
                    pinned="left",
                    format="%.3f"
                )
                elif c == "Name":
                    config[c] = st.column_config.TextColumn(
                        "Fruta",
                        disabled=True,
                        pinned="left",
                )
                else:
                    config[c] = st.column_config.TextColumn(
                        c,
                        disabled=True,
                        pinned="left",
                    )

        # Crear fila de subtotales
        subtotal_row = create_subtotal_row(editable_view)
        subtotal_df = pd.DataFrame([subtotal_row])
        
        # Concatenar subtotales al final (por defecto)
        editable_view_with_subtotals = pd.concat([editable_view, subtotal_df], ignore_index=True)
        subtotal_position = len(editable_view)  # Última fila
        
        # Aplicar estilos a la tabla con subtotales
        editable_columns = editable_view_with_subtotals.style
        
        # Aplicar negritas a las columnas de totales
        total_columns = ["MO Total", "Materiales Total", "Costos Directos", "Costos Indirectos"]
        existing_total_columns = [col for col in total_columns if col in editable_view_with_subtotals.columns]

        if existing_total_columns:
            editable_columns = editable_columns.set_properties(
                subset=existing_total_columns,
                **{"font-weight": "bold", "background-color": "#f8f9fa"}
            )

        # Aplicar estilos a columnas especiales
        if "Proceso Granel (USD/kg)" in editable_view_with_subtotals.columns:
            editable_columns = editable_columns.set_properties(
                subset=["Proceso Granel (USD/kg)"],
                **{"font-weight": "bold", "background-color": "#fff7ed"}
            )
        
        # Aplicar estilo especial a la fila de subtotales
        from pandas import IndexSlice as idx
        editable_columns = editable_columns.set_properties(
            subset=idx[subtotal_position, :],  # Fila de subtotales, todas las columnas
            **{
                "font-weight": "bold",
                "background-color": "#e8f4fd",
                "border-top": "2px solid #1f77b4",
            },
        )
        # Mostrar tabla con subtotales
        edited_df = st.dataframe(
            editable_columns,
            column_config=config,
            width="stretch",
            hide_index=True,
            key="data_editor_granel"
        )
        
        # # Mostrar métricas de subtotales
        # col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        # with col_metrics1:
        #     if "Costos Directos" in editable_view.columns:
        #         total_directos = editable_view["Costos Directos"].sum()
        #         st.metric("Costos Directos Total", f"{total_directos:,.2f}")
        
        # with col_metrics2:
        #     if "Costos Indirectos" in editable_view.columns:
        #         total_indirectos = editable_view["Costos Indirectos"].sum()
        #         st.metric("Costos Indirectos Total", f"{total_indirectos:,.2f}")
        
        # with col_metrics3:
        #     if "Proceso Granel (USD/kg)" in editable_view.columns:
        #         total_proceso = editable_view["Proceso Granel (USD/kg)"].sum()
        #         st.metric("Proceso Granel Total", f"{total_proceso:,.2f}")

        # --------- KPIs ---------
        st.subheader("📈 KPIs de Granel (escenario)")
        editable_view_con_proceso = editable_view[editable_view["Proceso Granel (USD/kg)"] < 0]
        try:
            kpis_granel = calculate_granel_kpis(editable_view_con_proceso)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Frutas", kpis_granel.get("Total Frutas", 0))
            with c2:
                st.metric("MO Promedio", f"${kpis_granel.get('MO Promedio (USD/kg)', float('nan')):.3f}/kg" if "MO Promedio (USD/kg)" in kpis_granel else "N/A")
            with c3:
                st.metric("Materiales Promedio", f"${kpis_granel.get('Materiales Promedio (USD/kg)', float('nan')):.3f}/kg" if "Materiales Promedio (USD/kg)" in kpis_granel else "N/A")
            with c4:
                st.metric("Precio Efectivo Promedio", f"${kpis_granel.get('Precio Efectivo Promedio (USD/kg)', float('nan')):.3f}/kg" if "Precio Efectivo Promedio (USD/kg)" in kpis_granel else "N/A")
        except Exception as e:
            st.error(f"❌ Error calculando KPIs: {e}")

        # # --------- Top/Bottom ---------
        # st.subheader("🏆 Top 10 y Bottom 10 Frutas por Costo")
        # try:
        #     c1, c2 = st.columns(2)
        #     config = {}
        #     config["Name"] = st.column_config.TextColumn(
        #         "Fruta",
        #         disabled=True,
        #         pinned="left",
        #     )
        #     with c1:
        #         st.subheader("Top 5")
        #         top_frutas, _ = get_top_bottom_granel(editable_view_con_proceso, 5)
        #         if not top_frutas.empty:
        #             show_cols = [c for c in ["Name", "Proceso Granel (USD/kg)", "Costos Directos"] if c in top_frutas.columns]
        #             st.dataframe(top_frutas[show_cols].style.format({k:"{:.3f}" for k in show_cols if k not in ["Name"]}), use_container_width=True, column_config=config, hide_index=True)
        #         else:
        #             st.info("No hay datos")
        #     with c2:
        #         st.subheader("Bottom 5")
        #         _, bottom_frutas = get_top_bottom_granel(editable_view_con_proceso, 5)
        #         if not bottom_frutas.empty:
        #             show_cols = [c for c in ["Name", "Proceso Granel (USD/kg)", "Costos Directos"] if c in bottom_frutas.columns]
        #             st.dataframe(bottom_frutas[show_cols].style.format({k:"{:.3f}" for k in show_cols if k not in ["Name"]}), use_container_width=True, column_config=config, hide_index=True)
        #         else:
        #             st.info("No hay datos")
        # except Exception as e:
        #     st.error(f"❌ Error obteniendo top/bottom: {e}")

        # --------- Gráfico ---------
        st.subheader("📈 Gráfico de Costos")
        c1, c2 = st.columns([1, 3])
        with c1:
            top_n_granel = st.number_input(
                "Número de frutas",
                min_value=5, max_value=50, value=10, step=5,
                help="Top N por costo"
            )
        with c2:
            st.write("")
        try:
            chart = create_granel_cost_chart(editable_view_con_proceso, int(top_n_granel))
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No se pudo crear el gráfico.")
        except Exception as e:
            st.error(f"❌ Error creando gráfico: {e}")

        # --------- Export ---------
        st.subheader("💾 Exportar Escenario de Granel")
        ec1, ec2, ec3 = st.columns([2, 1, 1])
        with ec1:
            filename_prefix_granel = st.text_input(
                "Prefijo del archivo:",
                value="escenario_granel",
                help="Nombre base para el archivo"
            )
        with ec2:
            export_format_granel = st.selectbox(
                "Formato:",
                options=["csv", "excel"],
                format_func=lambda x: "CSV" if x == "csv" else "Excel",
                help="Selecciona el formato de exportación",
                key="granel_format"
            )
        with ec3:
            if st.button("📥 Exportar", type="primary", key="export_granel"):
                try:
                    # Generar datos para descarga
                    data = get_data_for_download(st.session_state["sim.granel_df"], export_format_granel)
                    mime_type = get_mime_type(export_format_granel)
                    extension = get_file_extension(export_format_granel)
                    
                    # Generar nombre de archivo con timestamp
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
                    filename = f"{filename_prefix_granel}_{timestamp}.{extension}"
                    
                    st.download_button(
                        label=f"⬇️ Descargar {export_format_granel.upper()}",
                        data=data,
                        file_name=filename,
                        mime=mime_type,
                        key="download_granel"
                    )
                    st.success(f"✅ Escenario de granel listo para descarga en formato {export_format_granel.upper()}")
                except Exception as e:
                    st.error(f"❌ Error exportando: {e}")

    # ======================================================
    # TAB: ÓPTIMOS
    # ======================================================
    with tab_granel_optimos:
        st.header("🏆 Granel Óptimos (no editable)")
        granel_optimo = st.session_state.get("hist.granel_optimo")
        if granel_optimo is None or granel_optimo.empty:
            st.info("ℹ️ **Datos óptimos de granel no disponibles**. Se generan al cargar datos base.")
        else:
            st.subheader("📊 Datos Óptimos de Granel")
            opt_view = granel_optimo.copy()
            # Casting numérico
            num_cols = [c for c in opt_view.columns if c not in ["Fruta_id", "Fruta", "Name"]]
            for c in num_cols:
                opt_view[c] = pd.to_numeric(opt_view[c], errors="coerce")

            order_cols = ["Fruta_id", "Name", "Precio", "Rendimiento", "Precio Efectivo",
                          "MO Directa", "MO Indirecta", "MO Total",
                          "Materiales Directos", "Materiales Indirectos", "Materiales Total",
                          "Laboratorio", "Mantencion y Maquinaria",
                          "Costos Directos", "Costos Indirectos", "Servicios Generales"]
            avail_cols = [c for c in order_cols if c in opt_view.columns]
            opt_view = opt_view[available_cols]
            opt_view = opt_view.set_index("Fruta_id").sort_index()
            opt_view = opt_view.sort_values(by="Proceso Granel (USD/kg)")
            total_columns = ["MO Total", "Materiales Total", "Costos Directos", "Costos Indirectos"]
            opt_view_styled = opt_view.style
            if total_columns:
                    opt_view_styled = opt_view_styled.set_properties(
                        subset=total_columns,
                        **{"font-weight": "bold", "background-color": "#f8f9fa"}
                    )
            if "Proceso Granel (USD/kg)" in opt_view.columns:
                opt_view_styled = opt_view_styled.set_properties(
                    subset=["Proceso Granel (USD/kg)"],
                        **{"font-weight": "bold", "background-color": "#fff7ed"}
                    )
            config = {}
            for c in opt_view_styled.columns:
                if c not in ["Fruta_id", "Name", "Proceso Granel (USD/kg)", "Precio Efectivo"]:
                    config[c] = st.column_config.NumberColumn(
                        c,
                        format="%.3f"
                    )
                else:
                    if c == "Proceso Granel (USD/kg)":
                        config[c] = st.column_config.NumberColumn(
                        "Proceso Granel (USD/kg)",
                        disabled=True,
                        pinned="left",
                        format="%.3f"
                    )
                    elif c == "Name":
                        config[c] = st.column_config.TextColumn(
                            "Fruta",
                            disabled=True,
                            pinned="left",
                        )
            # Aplicar subtotales según la configuración
            if "Name" in opt_view.columns:
                # Crear fila de subtotales
                subtotal_row = create_subtotal_row(opt_view)
                subtotal_df = pd.DataFrame([subtotal_row])
                
                # Concatenar subtotales al final (por defecto)
                opt_view_with_subtotals = pd.concat([opt_view, subtotal_df], ignore_index=True)
                subtotal_position = len(opt_view)  # Última fila
                
                # Aplicar estilos a la tabla con subtotales
                opt_view_styled = opt_view_with_subtotals.style
                
                # Aplicar negritas a las columnas de totales
                total_columns = ["MO Total", "Materiales Total", "Costos Directos", "Costos Indirectos"]
                existing_total_columns = [col for col in total_columns if col in opt_view_with_subtotals.columns]

                if existing_total_columns:
                    opt_view_styled = opt_view_styled.set_properties(
                        subset=existing_total_columns,
                        **{"font-weight": "bold", "background-color": "#f8f9fa"}
                    )

                # Aplicar estilos a columnas especiales
                if "Proceso Granel (USD/kg)" in opt_view_with_subtotals.columns:
                    opt_view_styled = opt_view_styled.set_properties(
                        subset=["Proceso Granel (USD/kg)"],
                        **{"font-weight": "bold", "background-color": "#fff7ed"}
                    )
                
                # Aplicar estilo especial a la fila de subtotales
                from pandas import IndexSlice as idx
                opt_view_styled = opt_view_styled.set_properties(
                    subset=idx[subtotal_position, :],  # Fila de subtotales, todas las columnas
                    **{
                        "font-weight": "bold",
                        "background-color": "#e8f4fd",
                        "border-top": "2px solid #1f77b4",
                    },
                )
                
                # Mostrar tabla con subtotales
                opt_df = st.dataframe(
                    opt_view_styled,
                    column_config=config,
                    width="stretch",
                    hide_index=True,
                    key="data_editor_granel"
                )
                
                # # Mostrar métricas de subtotales
                # col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                
                # with col_metrics1:
                #     if "Costos Directos" in opt_view.columns:
                #         total_directos = opt_view["Costos Directos"].sum()
                #         st.metric("Costos Directos Total", f"{total_directos:,.2f}")
                
                # with col_metrics2:
                #     if "Costos Indirectos" in opt_view.columns:
                #         total_indirectos = opt_view["Costos Indirectos"].sum()
                #         st.metric("Costos Indirectos Total", f"{total_indirectos:,.2f}")
                
                # with col_metrics3:
                #     if "Proceso Granel (USD/kg)" in opt_view.columns:
                #         total_proceso = opt_view["Proceso Granel (USD/kg)"].sum()
                #         st.metric("Proceso Granel Total", f"{total_proceso:,.2f}")
            
            else:
                # Mostrar tabla normal
                opt_df = st.dataframe(
                    opt_view_styled,
                    column_config=config,
                    width="stretch",
                    hide_index=True,
                    key="data_editor_granel"
                )

            # KPIs
            st.subheader("📈 KPIs Óptimos")
            try:
                kpi_opt = calculate_granel_kpis(opt_view)
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Frutas Óptimas", kpi_opt.get("Total Frutas", 0))
                with k2:
                    st.metric("MO Promedio Óptimo", f"${kpi_opt.get('MO Promedio (USD/kg)', float('nan')):.3f}/kg" if "MO Promedio (USD/kg)" in kpi_opt else "N/A")
                with k3:
                    st.metric("Materiales Promedio Óptimo", f"${kpi_opt.get('Materiales Promedio (USD/kg)', float('nan')):.3f}/kg" if "Materiales Promedio (USD/kg)" in kpi_opt else "N/A")
                with k4:
                    st.metric("Precio Efectivo Promedio Óptimo", f"${kpi_opt.get('Precio Efectivo Promedio (USD/kg)', float('nan')):.3f}/kg" if "Precio Efectivo Promedio (USD/kg)" in kpi_opt else "N/A")
            except Exception as e:
                st.error(f"❌ Error KPIs óptimos: {e}")
    # ===================== PESTAÑA DE COMPARACIÓN =====================
    with tab_comparacion:
        st.header("⚖️ Comparación Simulador vs Histórico")
        st.markdown("Comparación de métricas clave entre el simulador y los datos históricos.")
        
        # Verificar que tenemos datos disponibles
        if "sim.df_filtered" not in st.session_state or st.session_state["sim.df_filtered"] is None:
            st.error("❌ No hay datos de simulación disponibles")
            st.info("💡 Aplica filtros en el sidebar para ver los datos")
            st.stop()
        
        if "hist.df_filtered" not in st.session_state or st.session_state["hist.df_filtered"] is None:
            st.error("❌ No hay datos históricos disponibles")
            st.info("💡 Ve a la página de Datos Históricos y carga un archivo")
            st.stop()
        
        # Obtener datos filtrados
        opt_data = st.session_state["hist.df_optimo"].copy()
        opt_data = opt_data[opt_data["SKU"].isin(st.session_state["sim.df_filtered"]["SKU"])]
        hist_data = st.session_state["hist.df_filtered"].copy()
        hist_data = hist_data[hist_data["SKU"].isin(st.session_state["sim.df_filtered"]["SKU"])]
        
        # Asegurar que SKU sea string en ambos datasets para el merge
        opt_data["SKU"] = opt_data["SKU"].astype(str)
        hist_data["SKU"] = hist_data["SKU"].astype(str)
        
        # Merge de datos simulador e histórico por SKU
        comparison_data = opt_data.merge(
            hist_data[["SKU", "PrecioVenta (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]], 
            on="SKU", 
            how="inner", 
            suffixes=("_sim", "_hist")
        )
        
        if comparison_data.empty:
            st.warning("⚠️ No hay SKUs coincidentes entre simulador e histórico")
            st.info("💡 Verifica que los filtros permitan ver SKUs comunes")
            st.stop()
        
        # # Mostrar métricas de resumen
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     st.metric("SKUs Comparados", len(comparison_data))
        
        # with col2:
        #     total_sim = comparison_data["EBITDA (USD/kg)_sim"].sum()
        #     total_hist = comparison_data["EBITDA (USD/kg)_hist"].sum()
        #     diff = total_sim - total_hist
        #     st.metric("EBITDA Total Sim", f"${total_sim:,.2f}", f"{diff:+,.2f} vs Hist")
        
        # with col3:
        #     avg_sim = comparison_data["EBITDA Pct_sim"].mean()
        #     avg_hist = comparison_data["EBITDA Pct_hist"].mean()
        #     diff_pct = avg_sim - avg_hist
        #     st.metric("EBITDA % Promedio Sim", f"{avg_sim:.1f}%", f"{diff_pct:+.1f}pp vs Hist")
        
        # with col4:
        #     price_sim = comparison_data["PrecioVenta (USD/kg)_sim"].mean()
        #     price_hist = comparison_data["PrecioVenta (USD/kg)_hist"].mean()
        #     diff_price = price_sim - price_hist
        #     st.metric("Precio Promedio Sim", f"${price_sim:.2f}", f"{diff_price:+.2f} vs Hist")
        
        # st.markdown("---")
        
        # Preparar datos para la tabla de comparación
        comparison_display = comparison_data[[
            "SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion",
            "PrecioVenta (USD/kg)_sim", "PrecioVenta (USD/kg)_hist",
            "Costos Totales (USD/kg)_sim", "Costos Totales (USD/kg)_hist", 
            "EBITDA (USD/kg)_sim", "EBITDA (USD/kg)_hist",
            "EBITDA Pct_sim", "EBITDA Pct_hist"
        ]].copy()
        
        # Renombrar columnas para mejor visualización
        comparison_display.columns = [
            "SKU", "Descripción", "Marca", "Cliente", "Especie", "Condición",
            "Precio 2026", "Precio Hist", 
            "Costo Óptimo", "Costo Hist",
            "EBITDA Óptimo", "EBITDA Hist", 
            "EBITDA % Óptimo", "EBITDA % Hist"
        ]
        
        # Calcular diferencias
        comparison_display["Δ Precio"] = comparison_display["Precio 2026"] - comparison_display["Precio Hist"]
        comparison_display["Δ Costo"] = comparison_display["Costo Óptimo"] - comparison_display["Costo Hist"]
        comparison_display["Δ EBITDA"] = comparison_display["EBITDA Óptimo"] - comparison_display["EBITDA Hist"]
        comparison_display["Δ EBITDA %"] = comparison_display["EBITDA % Óptimo"] - comparison_display["EBITDA % Hist"]
        
        # Aplicar formato a las columnas numéricas
        numeric_cols = ["Precio 2026", "Precio Hist", "Costo Óptimo", "Costo Hist", 
                    "EBITDA Óptimo", "EBITDA Hist", "EBITDA % Óptimo", "EBITDA % Hist",
                    "Δ Precio", "Δ Costo", "Δ EBITDA", "Δ EBITDA %"]
        
        for col in numeric_cols:
            if col in comparison_display.columns:
                if "%" in col:
                    comparison_display[col] = comparison_display[col].round(1)
                else:
                    comparison_display[col] = comparison_display[col].round(2)
        
        # Crear tabla con estilos
        st.subheader("📊 Tabla de Comparación Detallada")
        
        # Aplicar estilos a la tabla
        styled_comparison = comparison_display.style
        
        # Resaltar diferencias significativas
        def highlight_differences(val):
            if col == "Δ EBITDA %":
                if abs(val) > 10:
                    return 'background-color: #ffebee'
                elif abs(val) > 5:
                    return 'background-color: #fff3e0'
            else:
                if isinstance(val, (int, float)):
                    if abs(val) > 0.1:  # Diferencia significativa
                        return 'background-color: #ffebee'  # Rojo claro
                    elif abs(val) > 0.05:  # Diferencia moderada
                        return 'background-color: #fff3e0'  # Naranja claro
                return ''
        
        # Aplicar estilos a las columnas de diferencias
        diff_cols = ["Δ Precio", "Δ Costo", "Δ EBITDA", "Δ EBITDA %"]
        for col in diff_cols:
            if col in comparison_display.columns:
                styled_comparison = styled_comparison.applymap(highlight_differences, subset=[col])
        
        config = {}
        for col in styled_comparison.columns:
            if col not in ["SKU", "Descripción", "Marca", "Cliente", "Especie", "Condición"]:
                if col == "Δ EBITDA %":
                    config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.1f%%"
                    )
                else:
                    config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.2f"
                    )

        # Mostrar tabla
        st.dataframe(
            styled_comparison,
            column_config=config,
            width="stretch",
            height=400,
            hide_index=True
        )
        
        # Mostrar estadísticas de diferencias
        st.subheader("📈 Análisis de Diferencias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Diferencias en EBITDA (USD/kg):**")
            ebitda_diff = comparison_display["Δ EBITDA"]
            st.write(f"- Promedio: {ebitda_diff.mean():.2f}")
            st.write(f"- Mediana: {ebitda_diff.median():.2f}")
            st.write(f"- Desviación: {ebitda_diff.std():.2f}")
            st.write(f"- Rango: {ebitda_diff.min():.2f} a {ebitda_diff.max():.2f}")
        
        with col2:
            st.write("**Diferencias en EBITDA %:**")
            ebitda_pct_diff = comparison_display["Δ EBITDA %"]
            st.write(f"- Promedio: {ebitda_pct_diff.mean():.1f}pp")
            st.write(f"- Mediana: {ebitda_pct_diff.median():.1f}pp")
            st.write(f"- Desviación: {ebitda_pct_diff.std():.1f}pp")
            st.write(f"- Rango: {ebitda_pct_diff.min():.1f}pp a {ebitda_pct_diff.max():.1f}pp")
        
        # Botón de descarga con selector de formato
        col1, col2 = st.columns([1, 1])
        
        with col1:
            export_format_comparison = st.selectbox(
                "Formato de descarga:",
                options=["csv", "excel"],
                format_func=lambda x: "CSV" if x == "csv" else "Excel",
                help="Selecciona el formato de descarga",
                key="comparison_format"
            )
        
        with col2:
            data = get_data_for_download(comparison_display, export_format_comparison)
            mime_type = get_mime_type(export_format_comparison)
            extension = get_file_extension(export_format_comparison)
            
        st.download_button(
                label=f"📥 Descargar Comparación ({export_format_comparison.upper()})",
                data=data,
                file_name=f"comparacion_sim_vs_hist_{len(comparison_data)}_skus.{extension}",
                mime=mime_type,
                key="download_comparison"
        )

    
with tab_precio_frutas:
    st.header("🍓 Simulador de Precios de Frutas")
    
    # Verificar que los datos de frutas estén disponibles
    receta_df = st.session_state.get("fruta.receta_df")
    info_df = st.session_state.get("fruta.plan_2026")
    
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
    #     Rendimiento_promedio = params_actuales["RendimientoAjustada"].mean()
    #     st.metric(
    #         "Rendimiento Promedio",
    #         f"{Rendimiento_promedio:.1%}",
    #         help="Rendimiento promedio de todas las frutas"
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
    
    # ===================== Ajustes de Precio y Rendimiento =====================
    # st.subheader("⚙️ Ajustes de Precio y Rendimiento")
    
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
        Rendimiento_actual = fruta_info["Rendimiento"]
        nombre_fruta = fruta_info.get("Name", fruta_id)
        
        # col11, col12 = st.columns(2)
        # with col11:
        st.info(f"**Precio actual:** ${precio_actual:.3f}/kg")
        # with col12:
        #     st.info(f"**Rendimiento actual:** {Rendimiento_actual:.1%}")

    with col2:
        st.subheader("Ajuste de Precio y Rendimiento")

        # --- Estado por defecto de los selectores ---
        st.session_state.setdefault("fruit.tipo", "Precio")                # "Precio" | "Rendimiento"
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
            nuevo_Rendimiento = st.number_input(
                "Nuevo Rendimiento:",
                min_value=0.01, max_value=1.0, value=float(Rendimiento_actual),
                step=0.01, format="%.2f",
                help="Rendimiento debe estar entre 0.01 y 1.0"
            )
            cambio_Rendimiento = ((nuevo_Rendimiento / Rendimiento_actual) - 1) * 100 if Rendimiento_actual else 0.0
            override_data = {"rendimiento": {"type": "absolute", "value": nuevo_Rendimiento}}

        # --- SELECTORES ABAJO (actualizan estado y relanzan) ---
        sel1, sel2 = st.columns(2)

        with sel1:
            new_tipo = st.radio(
                "Tipo de ajuste:",
                ["Precio", "Rendimiento"],
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
            st.write(f"**Ajuste:** {cambio_Rendimiento:+.1f}%")
            st.write(f"**Rendimiento:** {Rendimiento_actual:.1%} → {nuevo_Rendimiento:.1%}")
        
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
                    st.write(f"Rendimiento: {override['efficiency']['value']:.2f}")
            
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
    base = params_actuales.rename(columns={"FrutaNombre":"Name",
                                            "PrecioBaseUSD_kg":"Costo Base (USD/kg)",
                                            "PrecioAjustadoUSD_kg":"Costo Ajustado (USD/kg)",
                                            "RendimientoBase":"Rendimiento Base",
                                            "RendimientoAjustado":"Rendimiento Ajustado",
                                            "CostoEfectivoBase":"Costo Efectivo Base (USD/kg)",
                                            "CostoEfectivoAjustado":"Costo Efectivo Ajustado (USD/kg)"})[
        ["Fruta_id","Name","Costo Base (USD/kg)","Costo Ajustado (USD/kg)","Rendimiento Base",
        "Rendimiento Ajustado","Costo Efectivo Base (USD/kg)","Costo Efectivo Ajustado (USD/kg)"]
    ]
    frutas = base.merge(
        tabla_resumen[["Fruta_id","SKUsAfectados"]],
        on="Fruta_id",
        how="left"
    )

    frutas["SKUsAfectados"] = frutas["SKUsAfectados"].fillna(0)
    frutas["Variación Costo %"] = np.where(
        frutas["Costo Base (USD/kg)"] > 0,
        (frutas["Costo Ajustado (USD/kg)"] / frutas["Costo Base (USD/kg)"] - 1) * 100,
        0.0
    )
    frutas["Variación Rendimiento %"] = np.where(
        frutas["Rendimiento Base"] > 0,
        (frutas["Rendimiento Ajustado"] / frutas["Rendimiento Base"] - 1) * 100,
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
        st.metric("Δ Costo Promedio", f"{frutas['Variación Costo %'].mean():+.1f}%")
    with k4:
        if not frutas.empty:
            top_idx = frutas["SKUsAfectados"].astype(float).idxmax()
            st.metric("Fruta más usada", frutas.loc[top_idx, "Name"])
        else:
            st.metric("Fruta más usada", "—")

    # 4) Tabla única (ordenada por SKUs afectados desc, precio desc)
    cols_tabla = [
        "Name", "Costo Base (USD/kg)", "Costo Ajustado (USD/kg)", "Costo Efectivo Base (USD/kg)",
        "Costo Efectivo Ajustado (USD/kg)", "Variación Costo %", "Variación Rendimiento %",
        "Rendimiento Base", "Rendimiento Ajustado", "SKUsAfectados"
    ]
    frutas_view = (frutas
        .sort_values(["SKUsAfectados","Costo Ajustado (USD/kg)"], ascending=[False, False])
        [cols_tabla]
    )

    st.dataframe(
        frutas_view.style.format({
            "Costo Base (USD/kg)": "{:.3f}",
            "Costo Ajustado (USD/kg)": "{:.3f}",
            "Variación Costo %": "{:+.1f}%",
            "Variación Rendimiento %": "{:+.1f}%",
            "Rendimiento Base": "{:.1%}",
            "Rendimiento Ajustado": "{:.1%}",
            "SKUsAfectados": "{:.0f}",
            "Costo Efectivo Base (USD/kg)": "{:.3f}",
            "Costo Efectivo Ajustado (USD/kg)": "{:.3f}",
        }),
        width="stretch", hide_index=True
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
            p = f["Costo Ajustado (USD/kg)"].dropna()
            if not p.empty:
                st.write(
                    f"• **Rango**: ${p.min():.3f} — ${p.max():.3f}/kg\n\n"
                    f"• **p25/mediana/p75**: ${p.quantile(0.25):.3f} / ${p.median():.3f} / ${p.quantile(0.75):.3f}/kg\n\n"
                    f"• **Desv. estándar (σ)**: ${p.std(ddof=1):.3f}/kg"
                )
        # ---------- Columna derecha ----------
        with colR:
            st.subheader("💰 Top 5 precios más altos")
            top_precio = f.sort_values("Costo Ajustado (USD/kg)", ascending=False).head(5)
            for _, r in top_precio.iterrows():
                st.write(f"• **{r['Name']}** — ${r['Costo Ajustado (USD/kg)']:.3f}/kg")

            st.subheader("🧊 Top 5 precios más bajos (>0)")
            low_precio = f[f["Costo Ajustado (USD/kg)"] > 0].sort_values("Costo Ajustado (USD/kg)", ascending=True).head(5)
            for _, r in low_precio.iterrows():
                st.write(f"• **{r['Name']}** — ${r['Costo Ajustado (USD/kg)']:.3f}/kg")


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

# 7) Gráfico B: Dispersión Precio vs Rendimiento (tamaño = SKUs, color = ΔPrecio%)
# try:
#     import plotly.express as px
#     fig_scatter = px.scatter(
#         frutas,
#         x="PrecioAjustadoUSD_kg", y="RendimientoAjustada",
#         size=frutas["SKUsAfectados"].fillna(0).clip(lower=0.1),  # que no desaparezcan
#         color="Cambio_Precio_%",
#         hover_data=["Name","SKUsAfectados"],
#         labels={"PrecioAjustadoUSD_kg":"Precio (USD/kg)","RendimientoAjustada":"Rendimiento"},
#         color_continuous_scale="RdBu_r"
#     )
#     fig_scatter.update_layout(
#         title="Precio vs. Rendimiento (tamaño = SKUs afectados, color = ΔPrecio%)",
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
        
        Este simulador te permite ajustar precios y Rendimientos de frutas para analizar su impacto en:
        - **Costos de MMPP (Fruta)** por SKU
        - **EBITDA** de cada producto
        - **Contribución total** de cada fruta al negocio
        
        ### 🔧 **Cómo usar**
        
        1. **Selecciona una fruta** del dropdown
        2. **Elige el tipo de ajuste**:
           - **Precio**: Porcentaje (%) o valor absoluto (USD/kg)
           - **Rendimiento**: Valor entre 0.01 y 1.0
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
        contrib_pos = PrecioAjustadoUSD_kg × Porcentaje ÷ RendimientoAjustada
        MMPP (Fruta) (USD/kg) = -contrib_pos
        ```
        
        **Impacto del ajuste:**
        ```
        Impacto = Contrib_Nueva - Contrib_Base
        ```
        
        ### 🗄️ **Estructura de datos**
        
        - **RECETA_SKU**: SKU, Fruta_id, Porcentaje
        - **INFO_FRUTA**: Fruta_id, Precio, Rendimiento, Name
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
                st.metric("Rendimiento promedio", f"{info_df['Rendimiento'].mean():.1%}")
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
        info = info_df[["Fruta_id","Precio","Rendimiento","Name"]].copy()
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

        # Contribución positiva = Precio * (Porcentaje/100) / Rendimiento
        pct  = pd.to_numeric(det["Porcentaje"], errors="coerce").fillna(0) / 100.0
        pr   = pd.to_numeric(det["Precio"], errors="coerce").fillna(0).clip(lower=0)
        opt  = pd.to_numeric(det["Óptimo"], errors="coerce").fillna(0) / 100.0
        rend  = pd.to_numeric(det["Rendimiento"], errors="coerce").fillna(0).clip(lower=0.01, upper=1.0)
        det["Name"] = det["Name"].fillna(det["Fruta_id"])
        det["Contribucion Original (USD/kg)"] = (pr * pct) / rend
        det["Contribucion Óptima (USD/kg)"] = (pr * opt) / rend
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
                .style.format({"Contribucion Original (USD/kg)":"{:.3f}","Porcentaje Original":"{:.2f}%","Contribucion Óptima (USD/kg)":"{:.3f}","Porcentaje Óptimo":"{:.2f}%","Precio":"{:.3f}","Rendimiento":"{:.3f}"}),
            width='stretch', hide_index=True
        )

        # Footer pegado
        st.markdown('<div class="modal-footer"></div>', unsafe_allow_html=True)

    else:
        st.dataframe(receta, width='stretch', hide_index=True)


with tab_receta:
    st.header("📖 Visor de Recetas por SKU")
    info_df = st.session_state.get("fruta.plan_2026")

    
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
    
    # --- AgGrid con botones de acción por fila ---
    st.caption(f"📊 Mostrando {len(skus_summary)} SKUs únicos con recetas")

    display_df = skus_summary.drop_duplicates(subset=["SKU"]).copy()
    display_df = display_df.drop(
        ["SKU-Cliente", "EBITDA (USD/kg)", "Porcentaje_Total"], 
        axis=1, 
        errors="ignore"
    )
    display_df["Ver Receta"] = "Ver Receta"

    gb = GridOptionsBuilder.from_dataframe(display_df)

    # columnas
    gb.configure_column("SKU", width=90, pinned="left", filter=False, suppressSizeToFit=True)
    if "Descripcion" in display_df.columns:
        gb.configure_column("Descripcion", minWidth=300,
        maxWidth=500, header_name="Descripción", filter=False)
    if "Marca" in display_df.columns:
        gb.configure_column("Marca", minWidth=200,
         maxWidth=400, filter=False)
    if "Cliente" in display_df.columns:
        gb.configure_column("Cliente", minWidth=200,
         maxWidth=400, filter=False)
    if "Frutas_Usadas" in display_df.columns:
        FRUTAS_CELL_STYLE = JsCode("""
        function(params){
            return {fontSize: '18px', fontWeight: '600'};
        }
        """)
        gb.configure_column(
            "Frutas_Usadas",
            width=100,
            header_name="Frutas",
            format="{:.0f}",
            pinned="right",
            suppressMovable=True,
            cellStyle=FRUTAS_CELL_STYLE,
            suppressSizeToFit=True
        ) 
    if "MMPP (Fruta) (USD/kg)" in display_df.columns:
        FORMATTER_3_DEC = JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) { return ''; }
            const num = Number(params.value);
            return num.toLocaleString('es-CL', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
        }
        """)
        gb.configure_column(
            "MMPP (Fruta) (USD/kg)",
            minWidth=150,
            maxWidth=250,
            header_name="MMPP Fruta",
            type=["numericColumn", "numberColumnFilter"],
            valueFormatter=FORMATTER_3_DEC,
            suppressSizeToFit=True
        )

    # selección (IMPORTANTE)
    gb.configure_selection(selection_mode="single", use_checkbox=False, suppressRowClickSelection=True)

    # botón que selecciona la fila y cubre todo el espacio
    BTN_RENDERER = JsCode("""
    class BtnRenderer {
    init(params){
        this.params = params;
        const b = document.createElement('button');
        b.textContent = 'Ver Receta';
        b.className = 'vf-btn-receta';
        b.style.cursor = 'pointer';
        b.style.padding = '6px 10px';
        b.style.border = '1px solid #d1d5db';
        b.style.borderRadius = '6px';
        b.style.background = '#e5e7eb';
        b.style.color = '#111827';
        b.style.fontSize = '12px';
        b.style.boxSizing = 'border-box';
        b.style.width = '100%';
        b.style.maxWidth = '100%';
        b.style.height = '28px';
        b.style.minHeight = '28px';
        b.style.margin = '0';
        b.style.overflow = 'hidden';
        b.style.textOverflow = 'ellipsis';
        b.style.whiteSpace = 'nowrap';
        b.style.display = 'flex';
        b.style.alignItems = 'center';
        b.style.justifyContent = 'center';
        b.onclick = () => {
            // Solo seleccionar cuando se hace click en el botón
            params.api.deselectAll();
            params.node.setSelected(true);
        };
        this.eGui = b;
    }
    getGui(){ return this.eGui; }
    destroy(){ this.eGui = null; }
    }
    """)

    gb.configure_column(
         "Ver Receta",
         header_name="Ver Receta",
         width=120,
         minWidth=110,
         maxWidth=140,
         filter=False,
         sortable=False,
         editable=False,
         cellRenderer=BTN_RENDERER,
         pinned="right",
         suppressMovable=True,
    )
    ## 1) CSS para forzar elipsis (overflow) y así tener casos con tooltip
    ELIPSIS_CSS = """
    <style>
    .ag-theme-balham .ag-cell.elipsis {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    }
    </style>
    """
    st.markdown(ELIPSIS_CSS, unsafe_allow_html=True)

    # 2) Getter simple (nada de DOM)
    TOOLTIP_VALUE = JsCode("""
    function(params){
    if (params.value === null || params.value === undefined) return '';
    if (Array.isArray(params.value)) return params.value.join(', ');
    return String(params.value);
    }
    """)


    # 3) Columna por defecto con tooltip
    gb.configure_default_column(
        resizable=True, sortable=True, filter=True,
        tooltipValueGetter=TOOLTIP_VALUE,
        cellClass="elipsis"   # activa la clase que hace overflow/ellipsis
    )

    # 4) No expandas siempre a todo el ancho: crea overflow real
    #    (si de verdad necesitas sizeColumnsToFit, aplícalo sólo una vez o
    #     marca columnas con 'suppressSizeToFit:true' para que queden estrechas)
    on_ready = JsCode("""
    function(p){
    // Si tu versión soporta 'tooltipShowMode', puedes dejarlo en gridOptions;
    // si no, no pasa nada, esto solo ajusta un pelín.
    // p.api.sizeColumnsToFit();  // <- comenta esto si quieres que haya truncamiento
    }
    """)
    

    gb.configure_grid_options(
        onFirstDataRendered=on_ready,
        tooltipShowDelay=100,
        tooltipHideDelay=8000,
        # Si tu versión lo soporta, déjalo. Si no, elimínalo sin problema.
        tooltipShowMode="whenTruncated",
        # enableCellTextSelection=True,
        # ensureDomOrder=True
        )

    gridOptions = gb.build()
    grid_response = AgGrid(
        display_df,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        excel_export_mode=ExcelExportMode.MANUAL,
        height=600,
        width='100%',
        allow_unsafe_jscode=True,
        theme="balham",
        custom_css={
            ".ag-header-cell-label": {"justify-content": "center"},
            ".ag-cell": {"display": "flex", "align-items": "center", "justify-content": "center", "overflow": "hidden"},
            ".ag-row": {"min-height": "40px"},
            ".ag-cell-value": {"display": "flex", "align-items": "center", "justify-content": "center"},
            ".ag-header-cell[col-id='Frutas_Usadas'] .ag-header-cell-label": {"font-size": "14px", "font-weight": "600"},
            ".vf-btn-receta": {"box-sizing": "border-box", "width": "100%", "max-width": "100%"},
            ".ag-pinned-right-cols-container .ag-cell": {"overflow": "hidden"}
    }
    )

    # Corregir pin de columna agregado manualmente si existiera
    try:
        gb.configure_column("Frutas", pinned=None)  # no existe campo "Frutas"; evita efectos colaterales
    except Exception:
        pass

    # leer selección de forma robusta
    sel_raw = grid_response.get("selected_rows", None)
    if isinstance(sel_raw, list):
        sel_records = sel_raw
    elif isinstance(sel_raw, pd.DataFrame):
        sel_records = sel_raw.to_dict("records")
    else:
        sel_records = []

    if sel_records:
        sku_to_view = str(sel_records[0].get("SKU", "")).strip()
        if sku_to_view:
            ver_receta_dialog(sku_to_view, st.session_state["fruta.receta_df"], st.session_state["fruta.plan_2026"])

    st.caption("💡 Haz clic en **Ver Receta** para abrir el modal con los detalles de la receta.")

    # ===================== Estadísticas por Fruta =====================
    st.subheader("📈 Estadísticas por Fruta (Mixes)")
    
    if info_df is not None and not receta_filtrada.empty:
        # Calcular estadísticas por fruta
        receta_filtrada_mixes = receta_filtrada[receta_filtrada["Porcentaje"] < 100]
        stats_fruta = receta_filtrada_mixes.groupby("Fruta_id").agg({
            "SKU": "nunique",
            "Porcentaje": "mean",
            "Óptimo": ["mean", "sum"],
        }).reset_index()
        
        # Flatten column names
        stats_fruta.columns = ["Fruta_id", "SKUs con Fruta", "Porcentaje Original Promedio", "Porcentaje Óptimo Promedio", "Porcentaje Óptimo Total"]
        
        # Enriquecer con información de frutas
        stats_fruta = stats_fruta.merge(
            info_df[["Fruta_id", "Precio", "Rendimiento", "Name"]],
            on="Fruta_id",
            how="left"
        )
        
        # Calcular contribución total por fruta
        stats_fruta["Contribucion_Total"] = (
            stats_fruta["Precio"] * 
            stats_fruta["Porcentaje Óptimo Total"] / 100 / 
            stats_fruta["Rendimiento"]
        )
        stats_fruta["Costo efectivo"] = stats_fruta["Precio"] / stats_fruta["Rendimiento"]


        view_fruta = stats_fruta[["Name", "SKUs con Fruta", "Porcentaje Original Promedio", "Porcentaje Óptimo Promedio", "Costo efectivo"]]
        view_fruta.rename(columns={"Costo efectivo": "Precio efectivo"}, inplace=True)
        view_fruta.sort_values(by="SKUs con Fruta", ascending=False, inplace=True)
        
        # Mostrar tabla de estadísticas
        st.dataframe(
            view_fruta.style.format({
                "SKUs con Fruta": "{:.0f}",
                "Porcentaje Original Promedio": "{:.2f}%",
                "Porcentaje Óptimo Promedio": "{:.2f}%",
                "Porcentaje Óptimo Total": "{:.1f}%",
                "Precio efectivo": "{:.3f}",
                "Rendimiento": "{:.3f}",
                "Contribucion_Total": "{:.3f}"
            }),
            column_config={
                "Name": st.column_config.TextColumn(width="small"),
                "SKUs con Fruta": st.column_config.NumberColumn(width="small"),
                "Porcentaje Original Promedio": st.column_config.NumberColumn(width="small", help="Este porcentaje considera los monoproductos"),
                "Porcentaje Óptimo Promedio": st.column_config.NumberColumn(width="small", help="Este porcentaje considera los monoproductos"),
                "Precio": st.column_config.NumberColumn(width="small"),
            },
            width='stretch',
            hide_index=True
        )
        
        # Gráfico de top frutas por uso
        st.subheader("🏆 Top Frutas por Uso (Mixes)")
        
        try:
            import plotly.express as px
            
            # Top 10 frutas por número de SKUs
            top_frutas = stats_fruta.nlargest(10, "SKUs con Fruta")
            
            fig_top_frutas = px.bar(
                top_frutas,
                x="Name",
                y="SKUs con Fruta",
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
            st.write("**Recetas Filtradas:**")
            format_recetas = st.selectbox(
                "Formato:",
                options=["csv", "excel"],
                format_func=lambda x: "CSV" if x == "csv" else "Excel",
                help="Selecciona el formato de descarga",
                key="recetas_format"
            )
            
            data_recetas = get_data_for_download(receta_filtrada, format_recetas)
            mime_type_recetas = get_mime_type(format_recetas)
            extension_recetas = get_file_extension(format_recetas)
            
            st.download_button(
                label=f"📥 Descargar Recetas Filtradas ({format_recetas.upper()})",
                data=data_recetas,
                file_name=f"recetas_filtradas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{extension_recetas}",
                mime=mime_type_recetas,
                key="download_recetas"
            )
        else:
            st.info("No hay recetas para descargar")
    
    with col2:
        # Descargar estadísticas por fruta
        if 'stats_fruta' in locals() and not stats_fruta.empty:
            st.write("**Estadísticas por Fruta:**")
            format_stats = st.selectbox(
                "Formato:",
                options=["csv", "excel"],
                format_func=lambda x: "CSV" if x == "csv" else "Excel",
                help="Selecciona el formato de descarga",
                key="stats_format"
            )
            
            data_stats = get_data_for_download(stats_fruta, format_stats)
            mime_type_stats = get_mime_type(format_stats)
            extension_stats = get_file_extension(format_stats)
            
            st.download_button(
                label=f"📥 Descargar Estadísticas por Fruta ({format_stats.upper()})",
                data=data_stats,
                file_name=f"stats_frutas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{extension_stats}",
                mime=mime_type_stats,
                key="download_stats"
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
        - **Rendimiento**: Factor de procesamiento de cada fruta
        """)

# -------- Información de navegación --------
st.markdown("---")

# Expander opcional para diagnóstico de session_state
with st.expander("🔎 Diagnóstico session_state", expanded=False):
    session_state_table()

st.info("💡 **Navegación**: Usa el menú lateral para volver a la página principal.")
st.info("💾 **Datos persistentes**: Los cambios se mantienen durante la sesión.")
st.info("📁 **Requisito**: Debes cargar datos en la página Home antes de usar el simulador.")
