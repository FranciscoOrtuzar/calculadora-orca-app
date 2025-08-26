"""
Simulador de EBITDA por SKU (USD/kg)
Página del simulador con filtros, overrides y análisis de rentabilidad.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Importar con manejo de errores más robusto
try:
    # Intentar import desde src
    from src.data_io import build_detalle, REQ_SHEETS
    from src.state import (
        ensure_session_state, session_state_table, sim_snapshot_push, 
        sim_undo, sim_redo, apply_fruit_override,
        get_sim_undo_count, get_sim_redo_count, is_sim_dirty
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



# ===================== Función para Recalcular Totales =====================
def recalculate_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula costos totales, gastos totales y EBITDA basándose en costos individuales.
    
    Args:
        df: DataFrame con costos individuales
        
    Returns:
        DataFrame con totales recalculados
    """
    df_calc = df.copy()
    
    # Convertir columnas numéricas y limpiar valores inválido
    numeric_columns = df_calc.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0.0)
    
    # FORZAR SIGNOS CORRECTOS
    # Los costos siempre deben ser negativos
    cost_columns = [col for col in df_calc.columns if "USD/kg" in col and "Precio" not in col]
    for col in cost_columns:
        if col in df_calc.columns:
            # Convertir valores a negativos (costos siempre negativos)
            df_calc[col] = -abs(df_calc[col])
    
    # También corregir columnas de costos sin USD/kg
    other_cost_columns = ["MO Directa", "MO Indirecta", "Materiales Cajas y Bolsas", 
                         "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", 
                         "Utilities", "Fletes Internos", "Comex", "Guarda PT"]
    for col in other_cost_columns:
        if col in df_calc.columns:
            # Convertir valores a negativos (costos siempre negativos)
            df_calc[col] = -abs(df_calc[col])
    
    # El precio de venta siempre debe ser positivo
    if "PrecioVenta (USD/kg)" in df_calc.columns:
        df_calc["PrecioVenta (USD/kg)"] = abs(df_calc["PrecioVenta (USD/kg)"])
    
    # 1. Recalcular MMPP Total si están los componentes
    mmpp_components = [
        "MMPP (Fruta) (USD/kg)",
        "Proceso Granel (USD/kg)"
    ]
    
    if all(col in df_calc.columns for col in mmpp_components):
        df_calc["MMPP Total (USD/kg)"] = df_calc[mmpp_components].sum(axis=1)
    
    # 2. Recalcular MO Total si están los componentes
    mo_components = [
        "MO Directa",
        "MO Indirecta"
    ]
    
    if all(col in df_calc.columns for col in mo_components):
        df_calc["MO Total"] = df_calc[mo_components].sum(axis=1)
    
    # 3. Recalcular Materiales Total si están los componentes
    materiales_components = [
        "Materiales Cajas y Bolsas",
        "Materiales Indirectos"
    ]
    
    if all(col in df_calc.columns for col in materiales_components):
        df_calc["Materiales Total"] = df_calc[materiales_components].sum(axis=1)
    
    # 4. Recalcular Gastos Totales (costos indirectos - NO incluye MMPP)
    gastos_components = [
        "Guarda MMPP",
        "MO Directa",
        "MO Indirecta",
        "Materiales Cajas y Bolsas",
        "Materiales Indirectos",
        "Calidad",
        "Mantencion",
        "Servicios Generales",
        "Utilities",
        "Fletes",
        "Comex",
        "Guarda PT",
        "Proceso Granel (USD/kg)"
    ]
    
    # Solo incluir componentes que existan en el DataFrame
    available_gastos = [col for col in gastos_components if col in df_calc.columns]
    if available_gastos:
        df_calc["Gastos Totales (USD/kg)"] = df_calc[available_gastos].sum(axis=1)
    
    # 5. Recalcular Costos Totales (MMPP + Gastos)
    costos_components = []
    
    # Agregar MMPP Total si existe
    if "MMPP (Fruta) (USD/kg)" in df_calc.columns:
        costos_components.append("MMPP (Fruta) (USD/kg)")

    # Agregar Gastos Totales si existe
    if "Gastos Totales (USD/kg)" in df_calc.columns:
        costos_components.append("Gastos Totales (USD/kg)")
    
    # Calcular costos totales
    if costos_components:
        df_calc["Costos Totales (USD/kg)"] = df_calc[costos_components].sum(axis=1)
    
    # 6. Recalcular EBITDA usando COSTOS TOTALES
    if "PrecioVenta (USD/kg)" in df_calc.columns and "Costos Totales (USD/kg)" in df_calc.columns:
        # Asegurar que los valores sean numéricos válidos
        precio = pd.to_numeric(df_calc["PrecioVenta (USD/kg)"], errors='coerce').fillna(0.0)
        costos = pd.to_numeric(df_calc["Costos Totales (USD/kg)"], errors='coerce').fillna(0.0)
        
        # Los costos ya están en valor absoluto (negativos), pero para el cálculo EBITDA los convertimos a positivos
        costos_abs = abs(costos)
        
        df_calc["EBITDA (USD/kg)"] = precio - costos_abs
        
        # Recalcular EBITDA Pct
        df_calc["EBITDA Pct"] = np.where(
            precio > 1e-12,
            (df_calc["EBITDA (USD/kg)"] / precio) * 100,
            0.0
        )
    return df_calc

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
    """
    Aplica ajustes universales a un DataFrame y recalcula totales.
    
    Args:
        df: DataFrame al que aplicar ajustes
        adjustments: Diccionario con ajustes universales
        
    Returns:
        DataFrame con ajustes aplicados y totales recalculados
    """
    if not adjustments:
        return df
    
    df_adjusted = df.copy()
    
    # Debug: mostrar información de entrada
    st.write(f"🔍 **Debug apply_universal_adjustments**:")
    st.write(f"  - DataFrame de entrada: {len(df)} filas")
    st.write(f"  - Ajustes a aplicar: {list(adjustments.keys())}")
    
    # Aplicar cada ajuste universal
    for cost_column, adjustment_info in adjustments.items():
        if cost_column in df_adjusted.columns:
            st.write(f"  - Aplicando ajuste a {cost_column}: {adjustment_info}")
            
            if adjustment_info["type"] == "percentage":
                # Aplicar ajuste porcentual manteniendo el signo negativo de los costos
                before_values = df_adjusted[cost_column].head(3).tolist()
                df_adjusted[cost_column] = df_adjusted[cost_column] * (1 + adjustment_info["value"] / 100)
                after_values = df_adjusted[cost_column].head(3).tolist()
                st.write(f"    - Valores antes: {before_values}")
                st.write(f"    - Valores después: {after_values}")
            else:  # dollars
                # Aplicar ajuste en dólares manteniendo el signo negativo de los costos
                before_values = df_adjusted[cost_column].head(3).tolist()
                df_adjusted[cost_column] = adjustment_info["value"]
                after_values = df_adjusted[cost_column].head(3).tolist()
                st.write(f"    - Valores antes: {before_values}")
                st.write(f"    - Valores después: {after_values}")
        else:
            st.write(f"  - ⚠️ Columna {cost_column} no encontrada en DataFrame")
    
    # Recalcular totales después de aplicar ajustes
    st.write(f"  - Recalculando totales...")
    df_adjusted = recalculate_totals(df_adjusted)
    st.write(f"  - Final: {len(df_adjusted)} filas")
    
    return df_adjusted

# ===================== Configuración de la página =====================
st.set_page_config(
    page_title="Simulador de EBITDA por SKU (USD/kg)",
    page_icon="📊",
    layout="wide"
)

# ===================== Inicialización de Variables de Sesión =====================
def initialize_session_state():
    """Inicializa todas las variables de sesión necesarias"""
    # Esta función ya no es necesaria ya que ensure_session_state() maneja todo
    pass

# Inicializar sesión
initialize_session_state()

# Inicializar y migrar todas las variables de session_state del sistema
ensure_session_state()

# ===================== Navegación =====================
# def show_navigation():
#     """Muestra la navegación entre páginas"""
#     st.sidebar.markdown("---")
#     st.sidebar.header("Navegación")
    
#     if st.sidebar.button("Datos Históricos"):
#         st.session_state.current_page = "home"
#         st.rerun()
    
#     if st.sidebar.button("Simulador EBITDA", type="primary"):
#         st.rerun()

st.title("Simulador de EBITDA por SKU (USD/kg)")
st.markdown("Simula escenarios de variación en costos y analiza impacto en rentabilidad por SKU.")

# ===================== Carga de datos =====================
@st.cache_data
def load_base_data():
    """Carga los datos base desde archivo local o sesión."""
    
    # CAMBIO: Priorizar 'hist.df' para el simulador
    if "hist.df" in st.session_state and len(st.session_state["hist.df"]) > 0:
        return st.session_state["hist.df"]
    
    # Si no hay datos en sesión, mostrar mensaje para cargar desde Home
    st.warning("⚠️ No hay datos cargados en la sesión")
    st.info("💡 Ve a la página Home y carga tu archivo Excel primero")
    
    # Mostrar botón para recargar
    if st.button("🔄 Recargar página"):
        st.rerun()
    
    return None

# Cargar datos base
df_base = load_base_data()

# Inicializar sim.df una sola vez
if df_base is not None and st.session_state["sim.df"] is None:
    st.session_state["sim.df"] = df_base.copy()

# Nunca sobrescribas sim.df por estar "dirty"
if df_base is not None and st.session_state.get("mmpp.dirty"):
    # aquí dispara solo lo que deba recalcularse (si aplica),
    # pero NO reasignes sim.df = df_base.copy()
    st.session_state["mmpp.dirty"] = False

# Filtrar SKUs sin costos totales (igual a 0) para análisis de EBITDA más preciso
# Guardar los excluidos en variable 'subproductos' para mantenerlos disponibles
if df_base is not None and "Costos Totales (USD/kg)" in df_base.columns:
    original_count = len(df_base)
    
    # Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
    subproductos = df_base[df_base["Costos Totales (USD/kg)"] == 0].copy()
    df_base = df_base[df_base["Costos Totales (USD/kg)"] != 0].copy()
    
    filtered_count = len(df_base)
    subproductos_count = len(subproductos)
    
    if original_count > filtered_count:        
        # IMPORTANTE: Recalcular totales en los datos cargados para asegurar que EBITDA Pct esté correcto
        if "EBITDA Pct" in df_base.columns:
            df_base = recalculate_totals(df_base)
        # Mostrar información sobre subproductos excluidos
        if st.session_state.get("show_subproductos_info", True):
            with st.expander(f"📋 **Subproductos excluidos** ({subproductos_count} SKUs)", expanded=False):
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
                    use_container_width=True
                )
                
                # Botón de exportación
                csv_subproductos = subproductos.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Lista Completa de Subproductos (CSV)",
                    data=csv_subproductos,
                    file_name="subproductos_excluidos_completo.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_subproductos_sim_1"
                )

# Si no hay datos base, intentar cargar desde la sesión directamente
if df_base is None:
    # CAMBIO: Priorizar 'hist.df' para el simulador ya que contiene los costos detallados
    if "hist.df" in st.session_state and st.session_state["hist.df"] is not None:
        df_base = st.session_state["hist.df"]
        # Filtrar SKUs sin costos totales también desde la sesión
        # Guardar los excluidos en variable 'subproductos' para mantenerlos disponibles
        if "Costos Totales (USD/kg)" in df_base.columns:
            original_count = len(df_base)
            
            # Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
            subproductos = df_base[df_base["Costos Totales (USD/kg)"] == 0].copy()
            df_base = df_base[df_base["Costos Totales (USD/kg)"] != 0].copy()
            
            filtered_count = len(df_base)
            subproductos_count = len(subproductos)
            
            # IMPORTANTE: Recalcular totales en los datos cargados para asegurar que EBITDA Pct esté correcto
        if "EBITDA Pct" in df_base.columns:
            df_base = recalculate_totals(df_base)
        
        st.success("✅ Datos cargados desde sesión (hist.df) - Fuente correcta para simulador")
    else:
        st.info("📁 Para usar el simulador, primero carga tu archivo Excel en la página Home.")
        st.info("🔄 Luego regresa aquí para simular escenarios.")
        st.stop()



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

# Debug: mostrar información de los datos
# Información de SKUs mostrados (con botón de cierre)
if st.session_state.get("show_skus_info", True):
    # Información sobre subproductos excluidos (si existen)
    if 'subproductos' in locals() and len(subproductos) > 0:
        if st.session_state.get("show_subproductos_sidebar", True):
            # Expander con detalles de subproductos en el sidebar
            with st.sidebar.expander(f"📋 Ver {len(subproductos)} subproductos", expanded=False):
                st.write("**SKUs excluidos del análisis de EBITDA:**")
                st.write(f"- **Total**: {len(subproductos)} SKUs")
                st.write(f"- **Razón**: Costos totales = 0")
                
                # Mostrar algunos ejemplos
                if len(subproductos) > 0:
                    sample_subproductos = subproductos[["SKU", "Descripcion", "Marca", "Cliente"]].head(5)
                    st.dataframe(sample_subproductos, use_container_width=True)
                    
                    if len(subproductos) > 5:
                        st.write(f"... y {len(subproductos) - 5} SKUs más")
                    
                    # Botón para exportar subproductos desde el sidebar
                    csv_subproductos = subproductos.to_csv(index=False)
                    st.download_button(
                        label="📥 Exportar Subproductos",
                        data=csv_subproductos,
                        file_name="subproductos_sin_costos.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_subproductos_sidebar"
                    )
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

# ===================== Bloque 1 - Carga de Planilla =====================
st.header("📁 Carga de Planilla (SKU-CostoNuevo)")

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
                st.dataframe(df_upload.head(10), use_container_width=True)
            
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
    filtered_skus = df_filtered["SKU"].tolist()
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

# Información sobre signos (con botón de cierre)
if st.session_state.get("show_signs_info", True):
    with st.container():
        col1, col2 = st.columns([20, 1])
        with col1:
            st.info("⚠️ **Importante sobre signos**: Los costos se muestran como valores negativos (ej: -$1.50/kg) y el precio de venta como positivo. Los signos se corrigen automáticamente.")
        with col2:
            if st.button("✕", key="close_signs_info", help="Cerrar aviso"):
                st.session_state.show_signs_info = False
                st.rerun()

# Mostrar información de filtros aplicados
if len(df_filtered) < len(df_base):
    # Información de filtros activos (con botón de cierre)
    if st.session_state.get("show_filters_info", True):
        with st.container():
            col1, col2 = st.columns([20, 1])
            with col1:
                st.success(f"🔍 **Filtros activos**: Mostrando {len(df_filtered)} SKUs de {len(df_base)} totales")
            with col2:
                if st.button("✕", key="close_filters_info", help="Cerrar aviso"):
                    st.session_state.show_filters_info = False
                    st.rerun()
    
    # Mostrar filtros activos
    active_filters = []
    for logical in FILTER_FIELDS:
        selections = SELECTIONS.get(logical, [])
        if selections:
            real_col = RESOLVED[logical]
            active_filters.append(f"**{logical}**: {', '.join(selections)}")
    
else:
    # Información de sin filtros (con botón de cierre)
    if st.session_state.get("show_no_filters_info", True):
        with st.container():
            col1, col2 = st.columns([20, 1])
            with col1:
                st.info("📊 **Sin filtros**: Mostrando todos los SKUs")
            with col2:
                if st.button("✕", key="close_no_filters_info", help="Cerrar aviso"):
                    st.session_state.show_no_filters_info = False
                    st.rerun()

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
                adjustment_value = st.number_input(
                    "Nuevo valor:",
                    min_value=-10.0,
                    max_value=0.0,
                    value=0.0,
                    step=0.01,
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
                
                # Debug: mostrar información antes de aplicar ajustes
                st.write(f"🔍 **Debug**: Aplicando ajuste a {selected_cost}")
                st.write(f"🔍 **Debug**: df_base tiene {len(df_base)} filas")
                st.write(f"🔍 **Debug**: Tipo de ajuste: {adjustment_type}")
                st.write(f"🔍 **Debug**: Valor del ajuste: {adjustment_value}")
                
                df_current_updated = apply_universal_adjustments(df_base, st.session_state["sim.overrides_row"])
                
                # Debug: mostrar información después de aplicar ajustes
                st.write(f"🔍 **Debug**: Después de apply_universal_adjustments: {len(df_current_updated)} filas")
                
                # IMPORTANTE: Excluir SKUs sin costos totales (igual que en df_base)
                if "Costos Totales (USD/kg)" in df_current_updated.columns:
                    before_filter = len(df_current_updated)
                    df_current_updated = df_current_updated[df_current_updated["Costos Totales (USD/kg)"] != 0].copy()
                    after_filter = len(df_current_updated)
                    st.write(f"🔍 **Debug**: Filtrado SKUs sin costos: {before_filter} → {after_filter} filas")
                
                # Recalcular totales en sim.df
                df_current_updated = recalculate_totals(df_current_updated)
                
                # Debug: mostrar información final
                st.write(f"🔍 **Debug**: Final: {len(df_current_updated)} filas en sim.df")
                
                # Guardar en sim.df
                st.session_state["sim.df"] = df_current_updated.copy()
                
                # Marcar como dirty
                st.session_state["sim.dirty"] = True
                
                st.success(f"✅ Ajuste universal aplicado a {len(filtered_skus)} SKUs filtrados y guardado en sesión")
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
                        st.success(f"✅ Ajuste de {cost_column} eliminado - Valores originales restaurados")
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
    
    # ===================== Tabla Editable Completa =====================
    st.subheader("📊 Tabla Editable - Todos los Costos")
    
    # Verificar que sim.df esté disponible
    if "sim.df" not in st.session_state or st.session_state["sim.df"] is None:
        st.error("❌ **No hay datos de simulación disponibles**")
        st.info("💡 **Para usar la tabla editable, primero debes:**")
        st.info("1. 📁 Cargar datos en la página Home")
        st.info("2. 🔄 Regresar al simulador")
        st.stop()
    
    # Mostrar información sobre ajustes universales aplicados
    if st.session_state.get("sim.overrides_row"):
        active_overrides = list(st.session_state["sim.overrides_row"].keys())
        st.info(f"🔧 **Ajustes universales activos**: {', '.join(active_overrides)} - Los valores mostrados incluyen estos ajustes")
        
        # Debug: mostrar información del estado de los datos
        with st.expander("🔍 Debug: Estado de los datos", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**df_base (original):**")
                if df_base is not None:
                    st.write(f"- Filas: {len(df_base)}")
                    st.write(f"- Columnas: {len(df_base.columns)}")
                else:
                    st.write("❌ No disponible")
            
            with col2:
                st.write("**sim.df (con ajustes):**")
                if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                    st.write(f"- Filas: {len(st.session_state['sim.df'])}")
                    st.write(f"- Columnas: {len(st.session_state['sim.df'].columns)}")
                    st.write(f"- Dirty: {st.session_state.get('sim.dirty', False)}")
                else:
                    st.write("❌ No disponible")
            
            with col3:
                st.write("**df_filtered (filtrado):**")
                if 'df_filtered' in locals():
                    st.write(f"- Filas: {len(df_filtered)}")
                    st.write(f"- Columnas: {len(df_filtered.columns)}")
                else:
                    st.write("❌ No disponible")
            
            # Mostrar algunos valores de ejemplo para el primer SKU
            if df_base is not None and len(df_base) > 0:
                st.write("**Ejemplo de valores (primer SKU):**")
                first_sku = df_base.iloc[0]
                if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
                    sim_first_sku = st.session_state["sim.df"][st.session_state["sim.df"]["SKU"] == first_sku["SKU"]]
                    if len(sim_first_sku) > 0:
                        sim_first_sku = sim_first_sku.iloc[0]
                        for col in ["MMPP Total (USD/kg)", "MO Total", "Materiales Total"]:
                            if col in df_base.columns and col in sim_first_sku:
                                original_val = first_sku[col]
                                adjusted_val = sim_first_sku[col]
                                st.write(f"- {col}: {original_val:.3f} → {adjusted_val:.3f}")

    # Botón para forzar recálculo manual
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **Edita cualquier costo individual y los totales se recalcularán automáticamente**")
    with col2:
        if st.button("🔄 Recalcular Totales", type="secondary", help="Fuerza el recálculo de todos los totales"):
            # Recalcular totales en el detalle filtrado
            detalle_filtrado = recalculate_totals(detalle_filtrado)
            
            # Actualizar solo sim.df (NO modificar hist.df)
            if "sim.df" in st.session_state:
                # Obtener todos los SKUs del detalle filtrado
                all_skus = detalle_filtrado["SKU"].tolist()
                sim_df_actualizado = st.session_state["sim.df"].copy()
                
                # Aplicar los cambios a los SKUs filtrados
                for sku in filtered_skus:
                    if sku in all_skus:
                        idx_sim = sim_df_actualizado[sim_df_actualizado["SKU"] == sku].index
                        if len(idx_sim) > 0:
                            idx = idx_sim[0]
                            # Copiar todos los valores actualizados
                            for col in detalle_filtrado.columns:
                                if col in sim_df_actualizado.columns:
                                    sim_df_actualizado.loc[idx, col] = detalle_filtrado[detalle_filtrado["SKU"] == sku][col].iloc[0]
                
                # Recalcular totales en sim.df
                sim_df_actualizado = recalculate_totals(sim_df_actualizado)
                st.session_state["sim.df"] = sim_df_actualizado
                st.session_state["sim.dirty"] = True
            
            st.success("✅ Totales recalculados manualmente")
            st.rerun()
    
    # Preparar datos para la tabla editable usando sim.df (que incluye ajustes universales)
    # Obtener datos de simulación si están disponibles en la sesión
    if "sim.df" in st.session_state and st.session_state["sim.df"] is not None:
        # Usar sim.df que ya incluye los ajustes universales aplicados
        sim_data = st.session_state["sim.df"].copy()
        # Filtrar por SKUs actuales
        filtered_skus = df_filtered["SKU"].tolist()
        
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
                
        # Configurar columnas editables (solo costos individuales, no totales)
        editable_columns = {}
        
        # Definir tipos de columnas para la configuración
        dimension_cols_edit = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]  # Columnas dimensionales visibles
        total_cols_edit = ["Costos Totales (USD/kg)", "Gastos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]
        intermediate_cols_edit = ["PrecioVenta (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
                                 "MO Total", "Materiales Total", "MMPP Total (USD/kg)"]
        
        for col in cost_columns:
            if col in df_edit.columns:
                editable_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor de {col} (los costos se muestran como negativos)",
                    format="%.3f",
                    step=0.001
                )
        
        # Configurar columnas dimensionales (visibles pero no editables)
        for col in dimension_cols_edit:
            if col in df_edit.columns and (col == "SKU" or col == "Descripcion"):
                editable_columns[col] = st.column_config.TextColumn(
                    col,
                    disabled=True,
                    help=f"{col} (no editable)",
                    pinned="left"
                )
            else:
                editable_columns[col] = st.column_config.TextColumn(
                    col,
                    disabled=True,
                    help=f"{col} (no editable)",
                )

        # Configurar la columna SKU-Cliente (oculta pero necesaria para el índice)
        if "SKU-Cliente" in df_edit.columns:
            editable_columns["SKU-Cliente"] = st.column_config.TextColumn(
                "SKU-Cliente",
                disabled=True,
                help="Identificador único SKU-Cliente (oculto)",
            )
        # Configurar columnas intermedias (no editables)
        for col in intermediate_cols_edit:
            if col == "PrecioVenta (USD/kg)":
                editable_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor intermedio de {col} (no editable)",
                    format="%.3f",
                    step=0.01,
                    min_value=0.0,
                    max_value=10.0
                )
            elif col in df_edit.columns:
                editable_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor intermedio de {col} (no editable)",
                    format="%.3f",
                    step=0.01,
                    disabled=True
                )
        
        for col in total_cols_edit:
            if col in df_edit.columns:
                editable_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor total de {col} (no editable)",
                    format="%.3f",
                    step=0.01,
                    disabled=True
                )
        
        # Configurar columnas de totales específicas con estilo especial
        total_columns_special = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total"]
        for col in total_columns_special:
            if col in df_edit.columns:
                editable_columns[col] = st.column_config.NumberColumn(
                    f"**{col}**",  # Título en negritas
                    help=f"Valor total de {col} (calculado automáticamente)",
                    format="%.3f",
                    step=0.01,
                    disabled=True
                )

        # Formato especial para EBITDA Pct como porcentaje
        if "EBITDA Pct" in df_edit.columns:
            editable_columns["EBITDA Pct"] = st.column_config.NumberColumn(
                "EBITDA Pct",
                help="Margen EBITDA en porcentaje (no editable)",
                format="%.1f%%",
                min_value=-100.0,
                step=0.1,
                disabled=True
            )
        
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
            use_container_width=True,
            height=500,
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
    if st.button("Ir a Inicio", type="primary", use_container_width=True):
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
                use_container_width=True
            )
            
            # Botón de exportación
            csv_subproductos = subproductos.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Lista Completa de Subproductos (CSV)",
                data=csv_subproductos,
                file_name="subproductos_excluidos_completo.csv",
                mime="text/csv",
                use_container_width=True,
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
                use_container_width=True
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
                use_container_width=True
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
        st.altair_chart(ebitda_chart, use_container_width=True)
    else:
        st.warning("⚠️ No se pudo crear el gráfico de EBITDA")
except Exception as e:
    st.error(f"❌ Error creando gráfico de EBITDA: {e}")

# Gráfico de distribución de márgenes
st.subheader("📊 Distribución de Márgenes")
try:
    margin_chart = create_margin_distribution_chart(df_current)
    if margin_chart:
        st.altair_chart(margin_chart, use_container_width=True)
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

# -------- Información de navegación --------
st.markdown("---")

# Expander opcional para diagnóstico de session_state
with st.expander("🔎 Diagnóstico session_state", expanded=False):
    session_state_table()

st.info("💡 **Navegación**: Usa el menú lateral para volver a la página principal.")
st.info("💾 **Datos persistentes**: Los cambios se mantienen durante la sesión.")
st.info("📁 **Requisito**: Debes cargar datos en la página Home antes de usar el simulador.")
