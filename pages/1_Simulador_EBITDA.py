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
    from src.simulator import (
        apply_filters, get_filter_options, apply_global_overrides, 
        apply_upload_overrides, compute_ebitda, calculate_kpis,
        get_top_bottom_skus, create_ebitda_chart, create_margin_distribution_chart,
        export_escenario, validate_upload_file
    )
except ImportError as e:
    st.warning(f"⚠️ Error importando desde src/: {e}")
    
    try:
        # Intentar import directo
        from data_io import build_detalle, REQ_SHEETS
        from simulator import (
            apply_filters, get_filter_options, apply_global_overrides, 
            apply_upload_overrides, compute_ebitda, calculate_kpis,
            get_top_bottom_skus, create_ebitda_chart, create_margin_distribution_chart,
            export_escenario, validate_upload_file
        )
        st.success("✅ Módulos importados correctamente (import directo)")
    except ImportError as e2:
        st.error(f"❌ Error en import directo: {e2}")
        
        # Crear funciones stub para evitar errores
        st.warning("⚠️ Creando funciones stub para continuar...")
        
        def apply_filters(df, **kwargs): return df
        def get_filter_options(df): return {}
        def apply_global_overrides(df, pct, enabled): return df
        def apply_upload_overrides(df, upload_df): return df, 0
        def compute_ebitda(df): return df
        def calculate_kpis(df): return {}
        def get_top_bottom_skus(df, n): return pd.DataFrame(), pd.DataFrame()
        def create_ebitda_chart(df, n): return None
        def create_margin_distribution_chart(df): return None
        def export_escenario(df, prefix): return Path("output.csv")
        def validate_upload_file(file): return True, "Archivo válido", None
        
        st.info("💡 Funciones stub creadas. Algunas funcionalidades pueden no funcionar correctamente.")

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
    
    # Agregar MMPP Total si existe (NO MMPP Fruta individual)
    if "MMPP Total (USD/kg)" in df_calc.columns:
        costos_components.append("MMPP Total (USD/kg)")
    # Si no existe MMPP Total, usar MMPP (Fruta) + Proceso Granel
    elif all(col in df_calc.columns for col in ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)"]):
        costos_components.extend(["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)"])

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
        old_value: Valor anterior
        new_value: Nuevo valor
    """
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = {}
    
    change_key = f"{sku}_{column}"
    st.session_state.edit_history[change_key] = {
        "sku": sku,
        "column": column,
        "old_value": old_value,
        "new_value": new_value,
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
    if "edit_history" not in st.session_state:
        return False
    
    change_key = f"{sku}_{column}"
    if change_key not in st.session_state.edit_history:
        return False
    
    change_info = st.session_state.edit_history[change_key]
    old_value = change_info["old_value"]
    
    # Revertir en el detalle de la sesión
    if "detalle" in st.session_state:
        mask = st.session_state.detalle["SKU"] == sku
        if mask.any():
            idx = st.session_state.detalle[mask].index[0]
            st.session_state.detalle.loc[idx, column] = old_value
            
            # Recalcular totales
            st.session_state.detalle = recalculate_totals(st.session_state.detalle)
            
            # Actualizar df_current
            if "df_current" in st.session_state:
                # Aplicar ajustes universales si existen
                if st.session_state.universal_adjustments:
                    st.session_state.df_current = apply_universal_adjustments(
                        st.session_state.detalle, 
                        st.session_state.universal_adjustments
                    )
                else:
                    st.session_state.df_current = st.session_state.detalle.copy()
            
            # Eliminar del historial
            del st.session_state.edit_history[change_key]
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
    
    # Aplicar cada ajuste universal
    for cost_column, adjustment_info in adjustments.items():
        if cost_column in df_adjusted.columns:
            if adjustment_info["type"] == "percentage":
                # Aplicar ajuste porcentual manteniendo el signo negativo de los costos
                df_adjusted[cost_column] = df_adjusted[cost_column] * (1 + adjustment_info["value"] / 100)
            else:  # dollars
                # Aplicar ajuste en dólares manteniendo el signo negativo de los costos
                df_adjusted[cost_column] = adjustment_info["value"]
    
    # Recalcular totales después de aplicar ajustes
    df_adjusted = recalculate_totals(df_adjusted)
    return df_adjusted

def ensure_adjustments_applied(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que los ajustes universales estén aplicados al DataFrame.
    
    Args:
        df: DataFrame base
        
    Returns:
        DataFrame con ajustes aplicados
    """
    if st.session_state.get("universal_adjustments"):
        return apply_universal_adjustments(df, st.session_state.universal_adjustments)
    return df

def recalculate_mmpp_from_fruit_adjustments(detalle_df: pd.DataFrame, recipe_data: dict, fruit_adjustments: dict) -> pd.DataFrame:
    """
    Recalcula MMPP (Fruta) desde recetas aplicando todos los ajustes de frutas activos.
    
    Args:
        detalle_df: DataFrame con datos de detalle
        recipe_data: Datos de recetas y frutas
        fruit_adjustments: Ajustes de frutas activos
        
    Returns:
        DataFrame con MMPP (Fruta) recalculado
    """
    if not fruit_adjustments or not recipe_data.get("success"):
        return detalle_df
    
    # Crear copia del DataFrame
    detalle_updated = detalle_df.copy()
    
    # Obtener datos de recetas
    recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
    fruit_prices = recipe_data.get("fruit_prices", {}).copy()  # Copia para modificar
    fruit_efficiency = recipe_data.get("fruit_efficiency", {})
    
    if recipe_df.empty or not fruit_prices:
        return detalle_df
    
    # Aplicar TODOS los ajustes de frutas activos a los precios
    for adjustment_key, adjustment_info in fruit_adjustments.items():
        fruta_id = adjustment_info["fruta_id"]
        if adjustment_info["type"] == "percentage":
            # Ajuste porcentual
            old_price = fruit_prices.get(fruta_id, 0)
            new_price = old_price * (1 + adjustment_info["value"] / 100)
            fruit_prices[fruta_id] = new_price
        else:
            # Ajuste en dólares
            fruit_prices[fruta_id] = adjustment_info["value"]
    
    # Obtener todos los SKUs únicos que tienen recetas
    all_skus_with_recipes = recipe_df["SKU"].unique().tolist()
    
    # Recalcular MMPP (Fruta) para TODOS los SKUs con recetas
    for sku in all_skus_with_recipes:
        sku_recipes = recipe_df[recipe_df["SKU"] == sku]
        total_mmpp = 0.0
        
        for _, recipe_row in sku_recipes.iterrows():
            fruta_id = recipe_row.get("fruta_id", "")
            porcentaje = recipe_row.get("Porcentaje", 0)
            
            if fruta_id and porcentaje > 0 and fruta_id in fruit_prices:
                fruit_cost_per_kg = fruit_prices[fruta_id]
                efficiency = fruit_efficiency.get(fruta_id, 0.9)
                
                # MMPP (Fruta) = (Porcentaje / 100) × Precio fruta / Eficiencia
                mmpp_fruta = (porcentaje / 100) * fruit_cost_per_kg / efficiency
                total_mmpp += mmpp_fruta
        
        # Actualizar MMPP (Fruta) en el detalle
        if "MMPP (Fruta) (USD/kg)" in detalle_updated.columns:
            mask = detalle_updated["SKU"] == sku
            if mask.any():
                detalle_updated.loc[mask, "MMPP (Fruta) (USD/kg)"] = total_mmpp
    
    # Recalcular MMPP Total si existe la columna
    if "MMPP (Fruta) (USD/kg)" in detalle_updated.columns and "Proceso Granel (USD/kg)" in detalle_updated.columns:
        detalle_updated["MMPP Total (USD/kg)"] = (
            detalle_updated["MMPP (Fruta) (USD/kg)"] + 
            detalle_updated["Proceso Granel (USD/kg)"]
        )
    
    # Recalcular todos los totales
    detalle_updated = recalculate_totals(detalle_updated)
    
    return detalle_updated

# ===================== Configuración de la página =====================
st.set_page_config(
    page_title="Simulador de EBITDA por SKU (USD/kg)",
    page_icon="📊",
    layout="wide"
)

# CSS para estilos de tabla mejorados
st.markdown("""
<style>
/* Estilos base para encabezados de tabla */
.stDataFrame th, .stDataEditor th {
    font-weight: bold !important;
    text-align: center !important;
    border: 1px solid #d1d5db !important;
    background-color: #f3f4f6 !important;
    color: #374151 !important;
}

/* Mejorar la legibilidad de las celdas de datos */
.stDataFrame td, .stDataEditor td {
    border: 1px solid #e5e7eb !important;
    padding: 8px !important;
    background-color: white !important;
}

/* Resaltar filas al pasar el mouse */
.stDataFrame tbody tr:hover td, .stDataEditor tbody tr:hover td {
    background-color: #f9fafb !important;
}

/* Estilos específicos para celdas editables */
.stDataEditor input {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ===================== Inicialización de Variables de Sesión =====================
def initialize_session_state():
    """Inicializa todas las variables de sesión necesarias"""
    if "upload_applied" not in st.session_state:
        st.session_state.upload_applied = False
    
    if "df_current" not in st.session_state:
        st.session_state.df_current = None
    
    if "universal_adjustments" not in st.session_state:
        st.session_state.universal_adjustments = {}
    
    if "detalle" not in st.session_state:
        st.session_state.detalle = None

# Inicializar sesión
initialize_session_state()

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

# Indicador del sistema de recetas
if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
    recipe_data = st.session_state.recipe_data
    recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
    fruit_prices = recipe_data.get("fruit_prices", {})
    
    if not recipe_df.empty and fruit_prices:
        # Contar frutas totales vs frutas con precios
        total_frutas = len(fruit_prices)
        frutas_con_precio = len([p for p in fruit_prices.values() if p > 0])
        
        st.success(f"🍎 **Sistema de Recetas Activo**: {len(recipe_df)} SKUs con recetas, {total_frutas} frutas base disponibles")
        st.info("💡 **MMPP (Fruta) se calcula automáticamente** desde recetas usando precios y eficiencias de INFO_FRUTA")
        
        if frutas_con_precio < total_frutas:
            st.warning(f"⚠️ **Atención**: Solo {frutas_con_precio} de {total_frutas} frutas tienen precios definidos. Las demás usarán precio 0.")
    else:
        st.warning("⚠️ **Sistema de Recetas**: Datos incompletos - verificar INFO_FRUTA y RECETA_SKU")
else:
    st.info("ℹ️ **Sistema de Recetas**: No disponible - cargar archivo con hojas INFO_FRUTA y RECETA_SKU para cálculo automático de MMPP")

# Resumen ejecutivo de ajustes activos
adjustments_summary = []
if st.session_state.get("universal_adjustments"):
    adjustments_summary.append(f"🔧 **{len(st.session_state.universal_adjustments)} ajustes de costos**")
if st.session_state.get("fruit_adjustments"):
    adjustments_summary.append(f"🍎 **{len(st.session_state.fruit_adjustments)} ajustes de frutas**")

if adjustments_summary:
    st.markdown("---")
    st.subheader("📊 **Resumen de Ajustes Activos**")
    for summary in adjustments_summary:
        st.info(summary)
    
    # Mostrar impacto total
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
        
        if not recipe_df.empty:
            # Contar SKUs afectados por ajustes de frutas
            affected_skus_fruits = set()
            if st.session_state.get("fruit_adjustments"):
                for adjustment_info in st.session_state.fruit_adjustments.values():
                    fruta_id = adjustment_info["fruta_id"]
                    fruta_skus = recipe_df[recipe_df["fruta_id"] == fruta_id]["SKU"].unique().tolist()
                    affected_skus_fruits.update(fruta_skus)
            
            # Contar SKUs afectados por ajustes de costos
            affected_skus_costs = set()
            if st.session_state.get("universal_adjustments"):
                for adjustment_info in st.session_state.universal_adjustments.values():
                    affected_skus_costs.update(adjustment_info.get("applied_skus", []))
            
            # SKUs totales afectados
            total_affected_skus = len(affected_skus_fruits.union(affected_skus_costs))
            
            if total_affected_skus > 0:
                st.success(f"✅ **Impacto Total**: {total_affected_skus} SKUs afectados por ajustes")
                
                with st.expander("🔍 Detalle de Impacto", expanded=False):
                    if affected_skus_fruits:
                        st.write(f"**🍎 Ajustes de Frutas**: {len(affected_skus_fruits)} SKUs afectados")
                    if affected_skus_costs:
                        st.write(f"**🔧 Ajustes de Costos**: {len(affected_skus_costs)} SKUs afectados")
                    
                    # SKUs afectados por ambos tipos de ajustes
                    both_affected = affected_skus_fruits.intersection(affected_skus_costs)
                    if both_affected:
                        st.write(f"**⚠️ SKUs con ambos ajustes**: {len(both_affected)} SKUs")
                        st.write("Estos SKUs tienen tanto ajustes de costos como de frutas aplicados")

# ===================== PESTAÑAS DEL SIMULADOR =====================
tab1, tab2 = st.tabs(["📊 Simulación por SKU", "🍎 Simulación por Receta"])

with tab1:
    st.header("📊 Simulación por SKU")
    st.markdown("Simula cambios en costos individuales por SKU.")
    
    # # Mostrar navegación
    # show_navigation()

# ===================== Carga de datos =====================
@st.cache_data
def load_base_data():
    """Carga los datos base desde archivo local o sesión."""
    
    # CAMBIO: Priorizar 'detalle' para el simulador
    if "detalle" in st.session_state and len(st.session_state.detalle) > 0:
        return st.session_state.detalle
    
    # Si no hay datos en sesión, mostrar mensaje para cargar desde Home
    st.warning("⚠️ No hay datos cargados en la sesión")
    st.info("💡 Ve a la página Home y carga tu archivo Excel primero")
    
    # Mostrar botón para recargar
    if st.button("🔄 Recargar página"):
        st.rerun()
    
    return None

# Cargar datos base
df_base = load_base_data()

# INICIALIZAR df_current si no existe en la sesión
if "df_current" not in st.session_state or st.session_state.df_current is None:
    if df_base is not None:
        st.session_state.df_current = df_base.copy()
        st.info("🔄 df_current inicializado desde datos base")
    elif "detalle" in st.session_state and st.session_state.detalle is not None:
        st.session_state.df_current = st.session_state.detalle.copy()
        st.info("🔄 df_current inicializado desde detalle de sesión")

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
    # CAMBIO: Priorizar 'detalle' para el simulador ya que contiene los costos detallados
    if "detalle" in st.session_state and st.session_state.detalle is not None:
        df_base = st.session_state.detalle
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
        
        st.success("✅ Datos cargados desde sesión (detalle) - Fuente correcta para simulador")
        
        # INICIALIZAR df_current si no existe
        if "df_current" not in st.session_state or st.session_state.df_current is None:
            st.session_state.df_current = df_base.copy()
            st.info("🔄 df_current inicializado desde datos de sesión")
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
df_filtered = _apply_filters(df_base, SELECTIONS).copy()

# Orden por SKU-Cliente si existe y sin índice
sku_cliente_col = "SKU-Cliente"
if sku_cliente_col in df_filtered.columns:
    df_filtered = df_filtered.sort_values([sku_cliente_col]).reset_index(drop=True)
else:
    df_filtered = df_filtered.reset_index(drop=True)


# # Botón para forzar actualización de datos cuando cambien los filtros
# if st.session_state.universal_adjustments:
#     if st.sidebar.button("🔄 Actualizar Datos con Filtros", type="secondary", 
#                          help="Fuerza la actualización de datos cuando cambian los filtros"):
#         # Reaplicar ajustes universales a los nuevos datos filtrados
#         df_current = apply_universal_adjustments(df_filtered, st.session_state.universal_adjustments)
        
#         # Filtrar SKUs sin costos totales también en datos simulados
#         if "Costos Totales (USD/kg)" in df_current.columns:
#             original_count = len(df_current)
#             df_current = df_current[df_current["Costos Totales (USD/kg)"] != 0].copy()
#             filtered_count = len(df_current)
#             if original_count > filtered_count:
#                 st.sidebar.info(f"🔍 Se excluyeron {original_count - filtered_count} SKUs sin costos de la simulación")
        
#         st.session_state.df_current = df_current.copy()
#         st.sidebar.success("✅ Datos actualizados con filtros actuales")
#         st.rerun()

# ===================== Sidebar - Overrides Globales =====================
st.sidebar.header("Overrides Globales")

# Checkbox para habilitar overrides globales
enable_global = st.sidebar.checkbox("Aplicar % global a costos", value=False)

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
# Inicializar TODAS las variables de sesión al principio
if "upload_applied" not in st.session_state:
    st.session_state.upload_applied = False

if "df_current" not in st.session_state:
    st.session_state.df_current = None

if "universal_adjustments" not in st.session_state:
    st.session_state.universal_adjustments = {}

# DataFrame actual para trabajar (usar df_global que ya tiene los filtros aplicados)
# Si hay datos en sesión, aplicarlos sobre los filtros actuales
if st.session_state.upload_applied and "df_current" in st.session_state and st.session_state.df_current is not None:
    # Aplicar los overrides de sesión sobre los datos filtrados
    df_current = st.session_state.df_current.copy()
    # Asegurar que solo se muestren los SKUs filtrados
    filtered_skus = df_filtered["SKU"].tolist()
    df_current = df_current[df_current["SKU"].isin(filtered_skus)].copy()
else:
    df_current = df_global.copy()

# VERIFICACIÓN DE SEGURIDAD: Asegurar que df_current no sea None
if df_current is None:
    st.error("❌ **Error crítico**: df_current es None después de filtros. Reiniciando...")
    df_current = df_global.copy()
    if "df_current" not in st.session_state:
        st.session_state.df_current = df_current.copy()
    st.warning("⚠️ df_current reinicializado desde df_global")

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
if st.session_state.universal_adjustments:
    # Aplicar ajustes universales a los datos filtrados
    df_current_with_adjustments = ensure_adjustments_applied(df_current)
    
    # Filtrar SKUs sin costos totales después de aplicar ajustes universales
    if "Costos Totales (USD/kg)" in df_current_with_adjustments.columns:
        df_current_with_adjustments = df_current_with_adjustments[df_current_with_adjustments["Costos Totales (USD/kg)"] != 0].copy()
    
    df_current = df_current_with_adjustments.copy()

# ===================== Bloque 2 - Tabla Editable con Todos los Costos =====================
st.header("Detalle de Costos Simulados")

# Mostrar indicador de ajustes universales activos
if st.session_state.universal_adjustments:
    st.success(f"🔧 **Ajustes Universales Activos**: {len(st.session_state.universal_adjustments)} costos modificados")
    with st.expander("📊 Ver Ajustes Aplicados", expanded=False):
        for cost_column, adjustment_info in st.session_state.universal_adjustments.items():
            adjustment_type_str = "Porcentaje" if adjustment_info["type"] == "percentage" else "Dólares"
            value_str = f"{adjustment_info['value']:+.1f}%" if adjustment_info["type"] == "percentage" else f"{adjustment_info['value']:+.3f} USD/kg"
            st.write(f"**{cost_column}**: {value_str} ({adjustment_type_str})")
else:
    st.info("ℹ️ **Sin ajustes universales** - Los costos se muestran en su valor original")

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
    
    # if active_filters:
    #     with st.expander("🔍 Ver filtros activos"):
    #         for filter_info in active_filters:
    #             st.write(filter_info)
        
    #     # Botón para limpiar filtros
    #     if st.button("🧹 Limpiar todos los filtros", type="secondary"):
    #         for logical in FILTER_FIELDS:
    #             if f"ms_sim_{logical}" in st.session_state:
    #                 del st.session_state[f"ms_sim_{logical}"]
    #         st.rerun()
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
if "detalle" in st.session_state and st.session_state.detalle is not None:
    detalle_data = st.session_state.detalle.copy()
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
                # Aplicar ajuste universal
                if adjustment_type == "Porcentaje (%)":
                    # Ajuste porcentual
                    # detalle_filtrado[f"{selected_cost}_Original"] = detalle_filtrado[selected_cost]
                    detalle_filtrado[selected_cost] = detalle_filtrado[selected_cost] * (1 + adjustment_value / 100)
                    st.success(f"✅ Ajuste aplicado: {adjustment_value:+.1f}% a {selected_cost}")
                else:
                    # Ajuste en dólares
                    # detalle_filtrado[f"{selected_cost}_Original"] = detalle_filtrado[selected_cost]
                    detalle_filtrado[selected_cost] = adjustment_value
                    st.success(f"✅ Ajuste aplicado: {adjustment_value:+.3f} USD/kg a {selected_cost}")
                
                # Recalcular totales
                detalle_filtrado = recalculate_totals(detalle_filtrado)
                
                # GUARDAR EL AJUSTE UNIVERSAL EN LA SESIÓN
                adjustment_key = f"{selected_cost}"
                
                # IMPORTANTE: Guardar valores originales ANTES de aplicar el ajuste
                # Usar el detalle original de la sesión, no el filtrado ya modificado
                original_values = {}
                for sku in filtered_skus:
                    if sku in st.session_state.detalle["SKU"].values:
                        idx_original = st.session_state.detalle[st.session_state.detalle["SKU"] == sku].index[0]
                        original_values[sku] = st.session_state.detalle.loc[idx_original, selected_cost]
                
                st.session_state.universal_adjustments[adjustment_key] = {
                    "type": "percentage" if adjustment_type == "Porcentaje (%)" else "dollars",
                    "value": adjustment_value,
                    "applied_skus": filtered_skus.copy(),  # Guardar SKUs afectados
                    "original_values": original_values,  # Guardar valores originales
                    "timestamp": pd.Timestamp.now()
                }
                

                
                # ACTUALIZAR LOS DATOS ORIGINALES EN LA SESIÓN
                # Obtener todos los SKUs del detalle original
                all_skus = st.session_state.detalle["SKU"].tolist()
                
                # Crear una copia del detalle original
                detalle_actualizado = st.session_state.detalle.copy()
                
                # Aplicar los cambios solo a los SKUs filtrados
                for sku in filtered_skus:
                    if sku in all_skus:
                        # Encontrar el índice en el detalle original
                        idx_original = detalle_actualizado[detalle_actualizado["SKU"] == sku].index
                        if len(idx_original) > 0:
                            idx = idx_original[0]
                            # Aplicar el cambio al costo seleccionado
                            if adjustment_type == "Porcentaje (%)":
                                # detalle_actualizado.loc[idx, f"{selected_cost}_Original"] = detalle_actualizado.loc[idx, selected_cost]
                                detalle_actualizado.loc[idx, selected_cost] = detalle_actualizado.loc[idx, selected_cost] * (1 + adjustment_value / 100)
                            else:
                                # detalle_actualizado.loc[idx, f"{selected_cost}_Original"] = detalle_actualizado.loc[idx, selected_cost]
                                # Para dólares por kg, sobreescribir completamente el valor
                                detalle_actualizado.loc[idx, selected_cost] = adjustment_value
                
                # Recalcular totales en el detalle completo
                detalle_actualizado = recalculate_totals(detalle_actualizado)
                
                # Actualizar la sesión con el detalle completo actualizado
                st.session_state.detalle = detalle_actualizado
                
                # ACTUALIZAR df_current para que se refleje en la tabla editable y KPIs
                # Aplicar los ajustes universales actualizados a df_global
                df_current_updated = apply_universal_adjustments(df_global, st.session_state.universal_adjustments)
                st.session_state.df_current = df_current_updated.copy()
                
                # IMPORTANTE: Aplicar filtros a df_current_updated para mantener consistencia
                filtered_skus = df_filtered["SKU"].tolist()
                df_current_filtered = df_current_updated[df_current_updated["SKU"].isin(filtered_skus)].copy()
                
                # Filtrar SKUs sin costos totales
                if "Costos Totales (USD/kg)" in df_current_filtered.columns:
                    df_current_filtered = df_current_filtered[df_current_filtered["Costos Totales (USD/kg)"] != 0].copy()
                
                st.session_state.df_current = df_current_filtered.copy()
                
                st.success(f"✅ Ajuste universal aplicado a {len(filtered_skus)} SKUs filtrados y guardado en sesión")
                st.rerun()
    
    # Mostrar ajustes universales activos
    if st.session_state.universal_adjustments:
        st.subheader("Ajustes Universales Activos")
        
                # Información sobre restauración (con botón de cierre)
        if st.session_state.get("show_restoration_info", True):
            with st.container():
                col1, col2 = st.columns([20, 1])
                with col1:
                    st.info("💡 **Restauración automática**: Al eliminar un ajuste, se restauran automáticamente los valores originales del detalle histórico.")
                with col2:
                    if st.button("✕", key="close_restoration_info", help="Cerrar aviso"):
                        st.session_state.show_restoration_info = False
                        st.rerun()
        
        for cost_column, adjustment_info in st.session_state.universal_adjustments.items():
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
                    # Restaurar valores originales del detalle
                    if "detalle" in st.session_state:
                        # Obtener valores originales del detalle
                        detalle_original = st.session_state.detalle.copy()
                        
                        # Aplicar valores originales a los SKUs afectados
                        for sku in adjustment_info["applied_skus"]:
                            if sku in detalle_original["SKU"].values:
                                idx = detalle_original[detalle_original["SKU"] == sku].index[0]
                                # Restaurar el valor original del costo usando los valores guardados
                                if cost_column in detalle_original.columns and "original_values" in adjustment_info:
                                    if sku in adjustment_info["original_values"]:
                                        original_value = adjustment_info["original_values"][sku]
                                        # Aplicar el valor original
                                        detalle_original.loc[idx, cost_column] = original_value
                        
                        # Recalcular totales
                        detalle_original = recalculate_totals(detalle_original)
                        st.session_state.detalle = detalle_original
                        
                        # Actualizar df_current
                        if "df_current" in st.session_state:
                            # Aplicar los ajustes universales restantes
                            remaining_adjustments = {k: v for k, v in st.session_state.universal_adjustments.items() if k != cost_column}
                            if remaining_adjustments:
                                df_current_updated = apply_universal_adjustments(detalle_original, remaining_adjustments)
                                st.session_state.df_current = df_current_updated.copy()
                            else:
                                st.session_state.df_current = detalle_original.copy()
                        

                    
                    # Eliminar el ajuste
                    del st.session_state.universal_adjustments[cost_column]
                    st.success(f"✅ Ajuste de {cost_column} eliminado - Valores originales restaurados")
                    st.rerun()
        
        # Botón para limpiar todos los ajustes
        if st.button("Limpiar todos los ajustes", type="secondary"):
            st.session_state.universal_adjustments = {}
            st.success("✅ Todos los ajustes universales eliminados")
            st.rerun()
    
    # ===================== AJUSTES UNIVERSALES POR FRUTA =====================
    st.subheader("🍎 Ajustes Universales por Fruta")
    st.markdown("Modifica los precios de las frutas base para recalcular automáticamente el MMPP (Fruta) desde recetas.")
    
    # Verificar si hay datos de recetas disponibles
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        fruit_prices = recipe_data.get("fruit_prices", {})
        fruit_efficiency = recipe_data.get("fruit_efficiency", {})
        fruit_names = recipe_data.get("fruit_names", {})
        
        if fruit_prices and fruit_names:
            # Inicializar ajustes de frutas en la sesión si no existen
            if "fruit_adjustments" not in st.session_state:
                st.session_state.fruit_adjustments = {}
            
            # Mostrar información sobre frutas disponibles
            total_frutas = len(fruit_prices)
            frutas_con_precio = len([p for p in fruit_prices.values() if p > 0])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"🌍 **{total_frutas} frutas disponibles** para ajuste de precios")
            with col2:
                st.metric("Frutas con precio", frutas_con_precio, f"de {total_frutas}")
            
            # Crear selector de fruta con nombres en inglés
            fruit_options = []
            for fruta_id, price in fruit_prices.items():
                fruit_name = fruit_names.get(fruta_id, fruta_id)
                fruit_options.append(f"{fruit_name} ({fruta_id})")
            
            # Ordenar por nombre de fruta
            fruit_options.sort()
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                selected_fruit_display = st.selectbox(
                    "Seleccionar fruta a modificar:",
                    options=fruit_options,
                    help="Selecciona la fruta base cuyo precio quieres modificar"
                )
                
                # Extraer fruta_id del display seleccionado
                selected_fruit = selected_fruit_display.split("(")[-1].rstrip(")")
            
            with col2:
                # Obtener precio actual
                current_price = fruit_prices.get(selected_fruit, 0)
                st.metric("Precio actual", f"${current_price:.3f}/kg")
            
            with col3:
                adjustment_type = st.selectbox(
                    "Tipo de ajuste:",
                    options=["Porcentaje (%)", "Dólares por kg (USD/kg)"],
                    help="Ajuste por porcentaje o nuevo valor en dólares por kg",
                    key="adjustment_type_fruit"
                )
            
            with col4:
                if adjustment_type == "Porcentaje (%)":
                    adjustment_value = st.number_input(
                        "Valor del ajuste:",
                        min_value=-100.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.5,
                        format="%.1f",
                        help="Porcentaje de cambio (-100 a +1000)",
                        key="adjustment_value_fruit"
                    )
                else:
                    adjustment_value = st.number_input(
                        "Nuevo precio:",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_price),
                        step=0.01,
                        format="%.3f",
                        help="Nuevo precio en dólares por kg",
                        key="adjustment_value_fruit"
                    )
            
            # Botón para aplicar ajuste
            if st.button("🚀 Aplicar Ajuste de Fruta", type="primary"):
                # Calcular nuevo precio
                if adjustment_type == "Porcentaje (%)":
                    new_price = current_price * (1 + adjustment_value / 100)
                    adjustment_key = f"fruta_{selected_fruit}_pct"
                else:
                    new_price = adjustment_value
                    adjustment_key = f"fruta_{selected_fruit}_dollars"
                
                # Guardar ajuste en la sesión
                st.session_state.fruit_adjustments[adjustment_key] = {
                    "fruta_id": selected_fruit,
                    "fruit_name": fruit_names.get(selected_fruit, selected_fruit),
                    "type": "percentage" if adjustment_type == "Porcentaje (%)" else "dollars",
                    "value": adjustment_value,
                    "old_price": current_price,
                    "new_price": new_price,
                    "timestamp": pd.Timestamp.now()
                }
                
                # Recalcular MMPP (Fruta) desde recetas con TODOS los ajustes activos
                if "detalle" in st.session_state:
                    # Usar la nueva función que recalcula todo correctamente
                    detalle_updated = recalculate_mmpp_from_fruit_adjustments(
                        st.session_state.detalle, 
                        recipe_data, 
                        st.session_state.fruit_adjustments
                    )
                    
                    # Actualizar la sesión
                    st.session_state.detalle = detalle_updated
                    
                    # Actualizar df_current
                    if "df_current" in st.session_state:
                        # Aplicar ajustes universales existentes si los hay
                        if st.session_state.universal_adjustments:
                            df_current_updated = apply_universal_adjustments(detalle_updated, st.session_state.universal_adjustments)
                            st.session_state.df_current = df_current_updated.copy()
                        else:
                            st.session_state.df_current = detalle_updated.copy()
                
                st.success(f"✅ Ajuste aplicado: {fruit_names.get(selected_fruit, selected_fruit)} → ${new_price:.3f}/kg")
                
                # Mostrar información de debug
                with st.expander("🔍 Debug: Ver Cambios Aplicados", expanded=False):
                    st.write(f"**Fruta modificada**: {fruit_names.get(selected_fruit, selected_fruit)} ({selected_fruit})")
                    st.write(f"**Precio anterior**: ${current_price:.3f}/kg")
                    st.write(f"**Precio nuevo**: ${new_price:.3f}/kg")
                    
                    # Mostrar SKUs afectados
                    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
                        recipe_df = st.session_state.recipe_data.get("recipe_df", pd.DataFrame())
                        if not recipe_df.empty:
                            skus_using_fruit = recipe_df[recipe_df["fruta_id"] == selected_fruit]["SKU"].unique().tolist()
                            st.write(f"**SKUs afectados**: {len(skus_using_fruit)}")
                            if len(skus_using_fruit) <= 10:
                                st.write(f"**Lista**: {', '.join(map(str, skus_using_fruit))}")
                            else:
                                st.write(f"**Primeros 10**: {', '.join(map(str, skus_using_fruit[:10]))}...")
                    
                    # Mostrar valores antes y después del cambio
                    if "detalle" in st.session_state:
                        detalle_before = st.session_state.detalle.copy()
                        # Mostrar algunos valores de ejemplo
                        sample_skus = detalle_before[detalle_before["SKU"].isin(skus_using_fruit[:3])][["SKU", "MMPP (Fruta) (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)"]]
                        st.write("**Valores antes del cambio (primeros 3 SKUs):**")
                        st.dataframe(sample_skus)
                
                st.rerun()
            
            # Mostrar ajustes de frutas activos
            if st.session_state.fruit_adjustments:
                st.subheader("🍎 Ajustes de Frutas Activos")
                
                for adjustment_key, adjustment_info in st.session_state.fruit_adjustments.items():
                    adjustment_type_str = "Porcentaje" if adjustment_info["type"] == "percentage" else "Dólares"
                    value_str = f"{adjustment_info['value']:+.1f}%" if adjustment_info["type"] == "percentage" else f"{adjustment_info['value']:+.3f} USD/kg"
                    fruit_name = adjustment_info["fruit_name"]
                    old_price = adjustment_info["old_price"]
                    new_price = adjustment_info["new_price"]
                    timestamp = adjustment_info["timestamp"].strftime("%H:%M:%S")
                    
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    with col1:
                        st.write(f"**{fruit_name}**: ${old_price:.3f} → ${new_price:.3f}/kg")
                    with col2:
                        st.write(f"**{value_str}**")
                    with col3:
                        st.write(f"**{adjustment_type_str}**")
                    with col4:
                        st.write(f"**{timestamp}**")
                    with col5:
                        if st.button("🗑️", key=f"remove_fruit_{adjustment_key}", help=f"Eliminar ajuste de {fruit_name}"):
                            # Restaurar precio original
                            if "detalle" in st.session_state:
                                # Eliminar el ajuste de la sesión
                                del st.session_state.fruit_adjustments[adjustment_key]
                                
                                # Recalcular MMPP (Fruta) con los ajustes restantes
                                detalle_restored = recalculate_mmpp_from_fruit_adjustments(
                                    st.session_state.detalle, 
                                    recipe_data, 
                                    st.session_state.fruit_adjustments
                                )
                                
                                # Actualizar la sesión
                                st.session_state.detalle = detalle_restored
                                
                                # Actualizar df_current
                                if "df_current" in st.session_state:
                                    # Aplicar ajustes universales existentes si los hay
                                    if st.session_state.universal_adjustments:
                                        df_current_updated = apply_universal_adjustments(detalle_restored, st.session_state.universal_adjustments)
                                        st.session_state.df_current = df_current_updated.copy()
                                    else:
                                        st.session_state.df_current = detalle_restored.copy()
                            
                            st.success(f"✅ Ajuste de {fruit_name} eliminado - Precio original restaurado")
                            st.rerun()
                
                # Botón para limpiar todos los ajustes de frutas
                if st.button("Limpiar todos los ajustes de frutas", type="secondary"):
                    # Limpiar todos los ajustes de frutas
                    st.session_state.fruit_adjustments = {}
                    
                    # Recalcular MMPP (Fruta) sin ajustes
                    if "detalle" in st.session_state:
                        detalle_restored = recalculate_mmpp_from_fruit_adjustments(
                            st.session_state.detalle, 
                            recipe_data, 
                            {}  # Sin ajustes
                        )
                        
                        # Actualizar la sesión
                        st.session_state.detalle = detalle_restored
                        
                        # Actualizar df_current
                        if "df_current" in st.session_state:
                            # Aplicar ajustes universales existentes si los hay
                            if st.session_state.universal_adjustments:
                                df_current_updated = apply_universal_adjustments(detalle_restored, st.session_state.universal_adjustments)
                                st.session_state.df_current = df_current_updated.copy()
                            else:
                                st.session_state.df_current = detalle_restored.copy()
                    
                    st.success("✅ Todos los ajustes de frutas eliminados")
                    st.rerun()
                
                # Botón de prueba para verificar ajustes
                if st.button("🧪 Probar Ajustes de Frutas", type="secondary", help="Verifica que los ajustes se estén aplicando correctamente"):
                    if "detalle" in st.session_state and st.session_state.get("fruit_adjustments"):
                        st.subheader("🔍 **Prueba de Ajustes de Frutas**")
                        
                        # Mostrar resumen de ajustes activos
                        st.write("**Ajustes activos:**")
                        for adjustment_key, adjustment_info in st.session_state.fruit_adjustments.items():
                            fruit_name = adjustment_info["fruit_name"]
                            old_price = adjustment_info["old_price"]
                            new_price = adjustment_info["new_price"]
                            st.write(f"- {fruit_name}: ${old_price:.3f} → ${new_price:.3f}/kg")
                        
                        # Verificar que MMPP (Fruta) se haya actualizado
                        if "MMPP (Fruta) (USD/kg)" in st.session_state.detalle.columns:
                            # Obtener algunos SKUs con recetas para verificar
                            recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
                            if not recipe_df.empty:
                                sample_skus = recipe_df["SKU"].unique()[:5]  # Primeros 5 SKUs
                                sample_data = st.session_state.detalle[st.session_state.detalle["SKU"].isin(sample_skus)][["SKU", "MMPP (Fruta) (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)"]]
                                
                                st.write("**Valores actuales (primeros 3 SKUs con recetas):**")
                                st.dataframe(sample_data)
                                
                                # Verificar que los valores no sean todos 0
                                mmpp_values = sample_data["MMPP (Fruta) (USD/kg)"]
                                if mmpp_values.sum() > 0:
                                    st.success("✅ **MMPP (Fruta) actualizado correctamente**")
                                else:
                                    st.warning("⚠️ **MMPP (Fruta) parece estar en 0** - Verificar recetas")
                            else:
                                st.warning("⚠️ No hay datos de recetas disponibles")
                        else:
                            st.error("❌ **Columna MMPP (Fruta) no encontrada** en los datos")
                    else:
                        st.warning("⚠️ No hay ajustes de frutas activos para probar")
                
                # Botón para forzar sincronización de tabla editable
                if st.button("🔄 Forzar Sincronización de Tabla Editable", type="secondary", help="Fuerza la actualización de la tabla editable con los ajustes de frutas"):
                    if "detalle" in st.session_state and st.session_state.get("fruit_adjustments"):
                        # Recalcular MMPP desde recetas
                        detalle_updated = recalculate_mmpp_from_fruit_adjustments(
                            st.session_state.detalle,
                            recipe_data,
                            st.session_state.fruit_adjustments
                        )
                        
                        # Actualizar la sesión
                        st.session_state.detalle = detalle_updated
                        
                        # Actualizar df_current
                        if st.session_state.get("universal_adjustments"):
                            df_current_updated = apply_universal_adjustments(detalle_updated, st.session_state.universal_adjustments)
                            st.session_state.df_current = df_current_updated.copy()
                        else:
                            st.session_state.df_current = detalle_updated.copy()
                        
                        st.success("✅ **Tabla editable sincronizada** con ajustes de frutas")
                        st.rerun()
                    else:
                        st.warning("⚠️ No hay ajustes de frutas activos para sincronizar")
                
                # Botón de emergencia para reinicializar datos
                if st.button("🚨 Reinicializar Datos de Emergencia", type="secondary", help="Reinicializa completamente todos los datos si hay problemas"):
                    if "detalle" in st.session_state:
                        # Reinicializar df_current desde detalle
                        st.session_state.df_current = st.session_state.detalle.copy()
                        
                        # Limpiar ajustes problemáticos
                        if "fruit_adjustments" in st.session_state:
                            st.session_state.fruit_adjustments = {}
                        
                        st.success("✅ **Datos reinicializados** - df_current restaurado desde detalle")
                        st.rerun()
                    else:
                        st.error("❌ No hay datos de detalle disponibles para reinicializar")
        else:
            st.warning("⚠️ No hay datos de frutas disponibles para ajustes")
    else:
        st.info("ℹ️ **Sistema de Recetas**: No disponible - cargar archivo con hojas INFO_FRUTA y RECETA_SKU para ajustes de frutas")
    
    # ===================== Tabla Editable Completa =====================
    st.subheader("📊 Tabla Editable - Todos los Costos")
    
    # Información sobre MMPP calculado desde recetas
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
        
        if not recipe_df.empty:
            # Contar SKUs con recetas disponibles
            skus_with_recipes = len(recipe_df)
            total_skus = len(df_current)
            
            st.info(f"🍎 **MMPP (Fruta) calculado desde recetas**: {skus_with_recipes} de {total_skus} SKUs tienen recetas disponibles")
            st.info("💡 Los valores de MMPP (Fruta) se calculan automáticamente usando: (Porcentaje/100) × Precio Fruta ÷ Eficiencia")
            
            # Mostrar indicador de ajustes de frutas activos
            if st.session_state.get("fruit_adjustments"):
                fruit_adjustments_count = len(st.session_state.fruit_adjustments)
                st.success(f"🔧 **Ajustes de Frutas Activos**: {fruit_adjustments_count} frutas modificadas")
                with st.expander("📊 Ver Ajustes de Frutas Aplicados", expanded=False):
                    for adjustment_key, adjustment_info in st.session_state.fruit_adjustments.items():
                        fruit_name = adjustment_info["fruit_name"]
                        old_price = adjustment_info["old_price"]
                        new_price = adjustment_info["new_price"]
                        adjustment_type_str = "Porcentaje" if adjustment_info["type"] == "percentage" else "Dólares"
                        value_str = f"{adjustment_info['value']:+.1f}%" if adjustment_info["type"] == "percentage" else f"{adjustment_info['value']:+.3f} USD/kg"
                        st.write(f"**{fruit_name}**: ${old_price:.3f} → ${new_price:.3f}/kg ({value_str})")
                
                # Mostrar impacto en MMPP
                st.info(f"💡 **MMPP (Fruta) recalculado automáticamente** para todos los SKUs con recetas")
                st.info(f"🔍 **Ver cambios**: Los valores de MMPP (Fruta) en la tabla editable reflejan los nuevos precios de frutas")
                
                # ⚠️ ADVERTENCIA IMPORTANTE
                st.warning("⚠️ **IMPORTANTE**: Si no ves los cambios en MMPP (Fruta) en la tabla editable, usa el botón '🔄 Forzar Sincronización de Tabla Editable' abajo")
            else:
                st.info("ℹ️ **Sin ajustes de frutas** - MMPP (Fruta) usa precios originales de INFO_FRUTA")
            
            # Mostrar ejemplo de cálculo si hay datos
            if not recipe_df.empty:
                sample_recipe = recipe_df.iloc[0]
                fruta_id = sample_recipe.get("fruta_id", "")
                porcentaje = sample_recipe.get("Porcentaje", 0)
                
                if fruta_id and porcentaje > 0:
                    fruit_prices = recipe_data.get("fruit_prices", {})
                    fruit_efficiency = recipe_data.get("fruit_efficiency", {})
                    fruit_names = recipe_data.get("fruit_names", {})
                    
                    if fruta_id in fruit_prices:
                        precio = fruit_prices[fruta_id]
                        eficiencia = fruit_efficiency.get(fruta_id, 0.9)
                        nombre_fruta = fruit_names.get(fruta_id, fruta_id)
                        
                        mmpp_calculado = (porcentaje / 100) * precio / eficiencia
                        
                        with st.expander("🧮 **Ejemplo de cálculo MMPP (Fruta)**", expanded=False):
                            st.write(f"**SKU de ejemplo**: {sample_recipe.get('SKU', 'N/A')}")
                            st.write(f"**Fruta**: {nombre_fruta} ({fruta_id})")
                            st.write(f"**Porcentaje en receta**: {porcentaje}%")
                            st.write(f"**Precio de la fruta**: ${precio:.3f}/kg")
                            st.write(f"**Eficiencia del proceso**: {eficiencia:.1%}")
                            st.write(f"**MMPP (Fruta) calculado**: ${mmpp_calculado:.3f}/kg")
                            st.write("")
                            st.write("**Fórmula**: (Porcentaje ÷ 100) × Precio Fruta ÷ Eficiencia")
                            st.write(f"**Cálculo**: ({porcentaje} ÷ 100) × ${precio:.3f} ÷ {eficiencia:.1%} = ${mmpp_calculado:.3f}/kg")
            
    
    # Leyenda de colores
    # with st.expander("🎨 **Leyenda de Colores**", expanded=False):
    #     col1, col2, col3, col4 = st.columns(4)
        
    #     with col1:
    #         st.markdown("""
    #         <div style="background-color: #dbeafe; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #2563eb;">
    #             <strong style="color: #2563eb;">📘 DIRECTOS</strong><br>
    #             <small>Costos Directos</small>
    #         </div>
    #         """, unsafe_allow_html=True)
        
    #     with col2:
    #         st.markdown("""
    #         <div style="background-color: #dcfce7; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #16a34a;">
    #             <strong style="color: #16a34a;">📗 INDIRECTOS</strong><br>
    #             <small>Costos Indirectos</small>
    #         </div>
    #         """, unsafe_allow_html=True)
        
    #     with col3:
    #         st.markdown("""
    #         <div style="background-color: #e5e7eb; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #6b7280;">
    #             <strong style="color: #6b7280;">📊 TOTALES</strong><br>
    #             <small>Costos/Gastos Totales</small>
    #         </div>
    #         """, unsafe_allow_html=True)
        
    #     with col4:
    #         st.markdown("""
    #         <div style="background-color: #fef9c3; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #f59e0b;">
    #             <strong style="color: #f59e0b;">💰 EBITDA</strong><br>
    #             <small>Rentabilidad</small>
    #         </div>
    #         """, unsafe_allow_html=True)

# # Información sobre subproductos en la tabla editable
# if 'subproductos' in locals() and len(subproductos) > 0:
#     st.info(f"💡 **Tabla Editable**: Solo se muestran {len(df_current)} SKUs con costos reales. {len(subproductos)} subproductos (costos = 0) fueron excluidos de la edición.")

#     # Obtener un SKU de ejemplo para mostrar los componentes
#     sample_sku = df_current["SKU"].iloc[0] if len(df_current) > 0 else None
    
#     # if sample_sku is not None:
#     #     sample_row = df_current[df_current["SKU"] == sample_sku].iloc[0]
        
#     #     st.write(f"**SKU de ejemplo:** {sample_sku}")
#     #     st.write(f"**PrecioVenta:** ${sample_row.get('PrecioVenta (USD/kg)', 'N/A'):.3f}/kg")
        
#     #     # Mostrar componentes de costos totales
#     #     costos_components = []
#     #     if "MMPP Total (USD/kg)" in df_current.columns:
#     #         costos_components.append("MMPP Total (USD/kg)")
#     #     elif all(col in df_current.columns for col in ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)"]):
#     #         costos_components.extend(["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)"])
        
#     #     other_cost_components = [
#     #         "Retail Costos Directos (USD/kg)",
#     #         "Retail Costos Indirectos (USD/kg)",
#     #         "Guarda MMPP",
#     #         "MO Directa",
#     #         "MO Indirecta",
#     #         "Materiales Cajas y Bolsas",
#     #         "Materiales Indirectos",
#     #         "Calidad",
#     #         "Mantencion",
#     #         "Servicios Generales",
#     #         "Utilities",
#     #         "Fletes",
#     #         "Comex",
#     #         "Guarda PT"
#     #     ]
        
#     #     for component in other_cost_components:
#     #         if component in df_current.columns and component not in costos_components:
#     #             costos_components.append(component)
        
#     #     st.write("**Componentes de Costos Totales:**")
#     #     total_cost = 0
#     #     for component in costos_components:
#     #         if component in sample_row:
#     #             value = sample_row[component]
#     #             total_cost += value
#     #             st.write(f"  - {component}: ${value:.3f}/kg")
        
#     #     st.write(f"**Costos Totales calculados:** ${total_cost:.3f}/kg")
        
#     #     if "Costos Totales (USD/kg)" in sample_row:
#     #         st.write(f"**Costos Totales en datos:** ${sample_row['Costos Totales (USD/kg)']:.3f}/kg")
        
#     #     if "PrecioVenta (USD/kg)" in sample_row and "Costos Totales (USD/kg)" in sample_row:
#     #         ebitda_calc = sample_row["PrecioVenta (USD/kg)"] - sample_row["Costos Totales (USD/kg)"]
#     #         st.write(f"**EBITDA calculado:** ${ebitda_calc:.3f}/kg")
            
#     #         if "EBITDA (USD/kg)" in sample_row:
#     #             st.write(f"**EBITDA en datos:** ${sample_row['EBITDA (USD/kg)']:.3f}/kg")
            
#     #         if sample_row["PrecioVenta (USD/kg)"] > 0:
#     #             ebitda_pct_calc = (ebitda_calc / sample_row["PrecioVenta (USD/kg)"]) * 100
#     #             st.write(f"**EBITDA % calculado:** {ebitda_pct_calc:.1f}%")
                
#     #             if "EBITDA Pct" in sample_row:
#     #                 st.write(f"**EBITDA % en datos:** {sample_row['EBITDA Pct']:.1f}%")
        
#     #     # Validación de cálculos
#     #     st.write("---")
#     #     st.write("**🔍 Validación de Cálculos:**")
        
#     #     # Crear DataFrame de muestra para validación
#     #     sample_df = pd.DataFrame([sample_row])
#     #     validation_result = validate_calculations(sample_df)
        
#     #     if validation_result["is_valid"]:
#     #         st.success("✅ Cálculos válidos")
#     #     else:
#     #         st.error("❌ Errores en cálculos:")
#     #         for error in validation_result["errors"]:
#     #             st.error(f"  - {error}")
        
#     #     if validation_result["warnings"]:
#     #         st.warning("⚠️ Advertencias:")
#     #         for warning in validation_result["warnings"]:
#     #             st.warning(f"  - {warning}")
        
#     #     # Mostrar valores de validación
#     #     st.write("**📊 Valores de Validación:**")
#     #     calc_info = validation_result["calculations"]
#     #     st.write(f"  - Precio: ${calc_info['precio']:.3f}/kg")
#     #     st.write(f"  - Costos Totales: ${calc_info['costos_totales']:.3f}/kg")
#     #     st.write(f"  - Gastos Totales: ${calc_info['gastos_totales']:.3f}/kg")
#     #     st.write(f"  - EBITDA: ${calc_info['ebitda']:.3f}/kg")
#     #     st.write(f"  - EBITDA %: {calc_info['ebitda_pct']:.1f}%")
        
#     #     # Verificación manual
#     #     if calc_info['precio'] > 0 and calc_info['costos_totales'] >= 0:
#     #         expected_ebitda = calc_info['precio'] - calc_info['costos_totales']
#     #         expected_pct = (expected_ebitda / calc_info['precio']) * 100
#     #         st.write(f"**🧮 Verificación Manual:**")
#     #         st.write(f"  - EBITDA esperado: ${expected_ebitda:.3f}/kg")
#     #         st.write(f"  - EBITDA % esperado: {expected_pct:.1f}%")
                
#     #         if abs(calc_info['ebitda'] - expected_ebitda) > 0.01:
#     #             st.error(f"  - ❌ Diferencia en EBITDA: {abs(calc_info['ebitda'] - expected_ebitda):.3f}")
#     #         else:
#     #             st.success(f"  - ✅ EBITDA correcto")
            
#     #         if abs(calc_info['ebitda_pct'] - expected_pct) > 0.1:
#     #             st.error(f"  - ❌ Diferencia en EBITDA %: {abs(calc_info['ebitda_pct'] - expected_pct):.1f}%")
#     #         else:
#     #             st.success(f"  - ✅ EBITDA % correcto")
                

    
    # Botón para forzar recálculo manual
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **Edita cualquier costo individual y los totales se recalcularán automáticamente**")
    with col2:
        if st.button("🔄 Recalcular Totales", type="secondary", help="Fuerza el recálculo de todos los totales"):
            # Recalcular totales en el detalle filtrado
            detalle_filtrado = recalculate_totals(detalle_filtrado)
            
            # Actualizar también en la sesión
            if "detalle" in st.session_state:
                # Obtener todos los SKUs del detalle original
                all_skus = st.session_state.detalle["SKU"].tolist()
                detalle_actualizado = st.session_state.detalle.copy()
                
                # Aplicar los cambios a los SKUs filtrados
                for sku in filtered_skus:
                    if sku in all_skus:
                        idx_original = detalle_actualizado[detalle_actualizado["SKU"] == sku].index
                        if len(idx_original) > 0:
                            idx = idx_original[0]
                            # Copiar todos los valores actualizados
                            for col in detalle_filtrado.columns:
                                if col in detalle_actualizado.columns:
                                    detalle_actualizado.loc[idx, col] = detalle_filtrado[detalle_filtrado["SKU"] == sku][col].iloc[0]
                
                # Recalcular totales en el detalle completo
                detalle_actualizado = recalculate_totals(detalle_actualizado)
                st.session_state.detalle = detalle_actualizado
            
            st.success("✅ Totales recalculados manualmente")
            st.rerun()
    
    # Preparar datos para la tabla editable usando df_current (que incluye ajustes universales)
    # Obtener datos del detalle si están disponibles en la sesión
    if "detalle" in st.session_state and st.session_state.detalle is not None:
        # IMPORTANTE: Asegurar que df_current esté sincronizado con el detalle y ajustes de frutas
        if st.session_state.get("fruit_adjustments") and "recipe_data" in st.session_state:
            # Si hay ajustes de frutas activos, recalcular MMPP desde recetas
            detalle_updated = recalculate_mmpp_from_fruit_adjustments(
                st.session_state.detalle,
                st.session_state.recipe_data,
                st.session_state.fruit_adjustments
            )
            # Actualizar df_current con los datos recalculados
            if st.session_state.get("universal_adjustments"):
                df_current_updated = apply_universal_adjustments(detalle_updated, st.session_state.universal_adjustments)
                st.session_state.df_current = df_current_updated.copy()
            else:
                st.session_state.df_current = detalle_updated.copy()
        
        # VERIFICACIÓN DE SEGURIDAD: Asegurar que df_current no sea None
        if st.session_state.get("df_current") is None:
            st.error("❌ **Error crítico**: df_current es None. Reiniciando datos...")
            # Reinicializar df_current desde detalle
            if st.session_state.get("detalle") is not None:
                st.session_state.df_current = st.session_state.detalle.copy()
                st.success("✅ df_current reinicializado desde detalle")
            else:
                st.error("❌ **Error fatal**: No hay datos de detalle disponibles")
                st.stop()
        
        # IMPORTANTE: Usar df_current que ya incluye ajustes universales y de frutas aplicados
        # VERIFICACIÓN FINAL: Asegurar que df_current sea válido
        if st.session_state.get("df_current") is None:
            st.error("❌ **Error crítico**: df_current es None en tabla editable. Reiniciando...")
            if st.session_state.get("detalle") is not None:
                st.session_state.df_current = st.session_state.detalle.copy()
                st.success("✅ df_current reinicializado desde detalle")
            else:
                st.error("❌ **Error fatal**: No hay datos disponibles para tabla editable")
                st.stop()
        
        detalle_filtrado = st.session_state.df_current.copy()
        
        # Debug: Mostrar información sobre ajustes aplicados
        if st.session_state.get("fruit_adjustments"):
            with st.expander("🔍 Debug: Ver Ajustes Aplicados en Tabla Editable", expanded=False):
                st.write("**Ajustes de frutas activos:**")
                for adjustment_key, adjustment_info in st.session_state.fruit_adjustments.items():
                    fruit_name = adjustment_info["fruit_name"]
                    old_price = adjustment_info["old_price"]
                    new_price = adjustment_info["new_price"]
                    st.write(f"- {fruit_name}: ${old_price:.3f} → ${new_price:.3f}/kg")
                
                # Mostrar algunos valores de MMPP (Fruta) para verificar
                if "MMPP (Fruta) (USD/kg)" in detalle_filtrado.columns:
                    sample_skus = detalle_filtrado[["SKU", "MMPP (Fruta) (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)"]].head(5)
                    st.write("**Valores en tabla editable (primeros 5 SKUs):**")
                    st.dataframe(sample_skus)
                    
                    # Verificar que los valores no sean todos 0
                    mmpp_values = sample_skus["MMPP (Fruta) (USD/kg)"]
                    if mmpp_values.sum() > 0:
                        st.success("✅ **MMPP (Fruta) actualizado en tabla editable**")
                    else:
                        st.warning("⚠️ **MMPP (Fruta) está en 0** - Verificar recálculo")
                else:
                    st.error("❌ **Columna MMPP (Fruta) no encontrada** en tabla editable")
        
        # Nota: SKU-Cliente se incluye en dimension_cols para el procesamiento pero se oculta en la tabla
        dimension_cols = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]
        
        # Columnas de EBITDA que estarán fijas a la izquierda después de Descripción
        ebitda_cols = ["EBITDA (USD/kg)", "EBITDA Pct"]
        
        # Columnas de totales restantes
        total_cols = ["Gastos Totales (USD/kg)", "Costos Totales (USD/kg)", "MMPP Total (USD/kg)",
        "MO Total", "Materiales Total", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)"]
        
        # Columnas de costos individuales
        editable_columns = ["PrecioVenta (USD/kg)", "MO Directa", "MO Indirecta", "Materiales Cajas y Bolsas",
                            "Materiales Indirectos", "Laboratorio", "Mantencion", "Servicios Generales",
                            "Utilities", "Fletes Internos", "Comex", "Guarda PT"]
        
        # Mantener SKU-Cliente en el DataFrame para el índice, pero no en la visualización
        display_order = dimension_cols + ebitda_cols + ["PrecioVenta (USD/kg)", 
                            "Costos Totales (USD/kg)", "MO Directa", "MO Indirecta", "MO Total", "Materiales Cajas y Bolsas",
                            "Materiales Indirectos", "Materiales Total", "Laboratorio", "Mantencion", "Servicios Generales",
                            "Utilities", "Fletes Internos", "Comex", "Guarda PT", "Retail Costos Directos (USD/kg)",
                            "Retail Costos Indirectos (USD/kg)", "Guarda MMPP", "Gastos Totales (USD/kg)", "MMPP (Fruta) (USD/kg)",
                            "Proceso Granel (USD/kg)", "MMPP Total (USD/kg)"]
        available_display_cols = [col for col in display_order if col in detalle_filtrado.columns]
        
        # Crear DataFrame para edición
        df_edit = detalle_filtrado[available_display_cols].copy()
                    
        # Configurar columnas editables (solo costos individuales, no totales)
        config_columns = {}
            
        for col in editable_columns:
            if col == "PrecioVenta (USD/kg)":
                config_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor intermedio de {col}",
                    format="%.3f",
                    step=0.01,
                    min_value=0.0,
                    max_value=10.0
                )
            else:
                config_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor de {col} (los costos se muestran como negativos)",
                    format="%.3f",
                    step=0.001
                )
        
        # Configurar columnas dimensionales (visibles pero no editables)
        for col in dimension_cols:
            if col in df_edit.columns and (col == "SKU" or col == "Descripcion"):
                config_columns[col] = st.column_config.TextColumn(
                    col,
                    disabled=True,
                    help=f"{col} (no editable)",
                    pinned="left"
                )
            else:
                config_columns[col] = st.column_config.TextColumn(
                    col,
                    disabled=True,
                    help=f"{col} (no editable)",
                )

        for col in total_cols:
            if col in df_edit.columns:
                config_columns[col] = st.column_config.NumberColumn(
                    col,
                    help=f"Valor intermedio de {col} (no editable)",
                    format="%.3f",
                    step=0.01,
                    disabled=True
                )
        # Configurar columnas de EBITDA fijas a la izquierda
        for col in ebitda_cols:
            if col in df_edit.columns:
                if col == "EBITDA Pct":
                    config_columns[col] = st.column_config.NumberColumn(
                        col,
                        help="Margen EBITDA en porcentaje (no editable)",
                        format="%.1f%%",
                        min_value=-100.0,
                        step=0.1,
                        disabled=True,
                        pinned="left"
                    )
                else:
                    config_columns[col] = st.column_config.NumberColumn(
                        col,
                        help=f"Valor total de {col} (no editable)",
                        format="%.3f",
                        step=0.01,
                        disabled=True,
                        pinned="left"
                    )
        
        # Aplicar estilos antes de mostrar la tabla editable (igual que en datos históricos)
        df_edit_styled = df_edit.copy()
        
        # ESTABLECER EL ÍNDICE ANTES de aplicar estilos
        df_edit_styled = df_edit_styled.set_index("SKU-Cliente")
        
        # IMPORTANTE: Guardar una copia del DataFrame original (con índice) ANTES de convertir a Styler
        df_edit_original = df_edit_styled.copy()
        
        # 1) SIEMPRE parte de un Styler
        styler = df_edit_styled.style

        # 2) Formato numérico/%
        fmt_cols = {}
        skip_cols = {"SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"}

        # OJO: usa SIEMPRE el DataFrame (no el Styler) aquí
        for c, dt in df_edit_styled.dtypes.items():
            if c in skip_cols:
                continue
            if "Pct" in c or "Porcentaje" in c:
                fmt_cols[c] = "{:.1%}"
            elif pd.api.types.is_numeric_dtype(dt):
                fmt_cols[c] = "{:.3f}"
        
        # Aplicar negritas a las columnas de totales
        existing_total_columns = [col for col in total_cols if col in df_edit.columns]
        
        if existing_total_columns:
            styler = styler.set_properties(
                subset=existing_total_columns,
                **{"font-weight": "bold", "background-color": "#f8f9fa"}
            )
        
        # Aplicar estilos a columnas EBITDA
        existing_ebitda_columns = [col for col in ebitda_cols if col in df_edit.columns]
        
        if existing_ebitda_columns:
            styler = styler.set_properties(
                subset=existing_ebitda_columns,
                **{"font-weight": "bold", "background-color": "#fff7ed"}
            )
        
        styler = styler.format(fmt_cols)
        
        edited_df = st.data_editor(
            styler,
            column_config=config_columns,
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
            
            # Obtener SKUs filtrados desde df_current
            filtered_skus = df_current["SKU"].tolist()
            
            # SIMPLIFICADO: Comparar directamente usando el DataFrame original sin índice
            df_edit_original_reset = df_edit_original.reset_index()
            
            # Convertir filtered_skus a strings para que coincida con los DataFrames
            filtered_skus_str = [str(sku) for sku in filtered_skus]
            
            # Buscar cambios por SKU directamente
            for sku in filtered_skus_str:
                # Buscar el SKU en ambos DataFrames
                mask_original = df_edit_original_reset["SKU"] == sku
                mask_edited = edited_df_reset["SKU"] == sku
                
                if mask_original.any() and mask_edited.any():
                    # Obtener las filas correspondientes
                    original_row = df_edit_original_reset[mask_original].iloc[0]
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
            
            if changes_detected > 0:
                st.success(f"✅ {changes_detected} cambios detectados y guardados en historial")
                
                # Validar y corregir signos antes de procesar
                edited_df_reset = validate_and_correct_signs(edited_df_reset)
                
                # IMPORTANTE: Recalcular totales directamente en edited_df_reset
                edited_df_recalculated = recalculate_totals(edited_df_reset)
                
                # ACTUALIZAR LA SESIÓN COMPLETA
                if "detalle" in st.session_state:
                    # Obtener todos los SKUs del detalle original
                    all_skus = st.session_state.detalle["SKU"].tolist()
                    detalle_actualizado = st.session_state.detalle.copy()
                    
                    # Aplicar los cambios a los SKUs filtrados
                    for sku in filtered_skus:
                        if sku in all_skus:
                            idx_original = detalle_actualizado[detalle_actualizado["SKU"] == sku].index
                            if len(idx_original) > 0:
                                idx = idx_original[0]
                                # Copiar todos los valores actualizados
                                for col in edited_df_recalculated.columns:
                                    if col in detalle_actualizado.columns:
                                        detalle_actualizado.loc[idx, col] = edited_df_recalculated[edited_df_recalculated["SKU"] == sku][col].iloc[0]
                    
                    # Recalcular totales en el detalle completo
                    detalle_actualizado = recalculate_totals(detalle_actualizado)
                    st.session_state.detalle = detalle_actualizado
                
                # ACTUALIZAR df_current para que se refleje en KPIs y gráficos
                # Aplicar ajustes universales si existen
                if st.session_state.universal_adjustments:
                    df_current_updated = apply_universal_adjustments(edited_df_recalculated, st.session_state.universal_adjustments)
                    st.session_state.df_current = df_current_updated.copy()
                else:
                    st.session_state.df_current = edited_df_recalculated.copy()
                
                st.success("✅ EBITDA recalculado automáticamente")
                
                # Forzar actualización de la vista automáticamente
                st.rerun()
            else:
                st.warning("⚠️ No se detectaron cambios específicos")
        else:
            st.info("ℹ️ No hay cambios en la tabla editable")
        
        # Mostrar historial de cambios y opciones de reversión
        if "edit_history" in st.session_state and st.session_state.edit_history:
            st.subheader("📝 Historial de Cambios Individuales")
            
            # Agrupar cambios por SKU para mejor visualización
            changes_by_sku = {}
            for change_key, change_info in st.session_state.edit_history.items():
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
                reverted_total = 0
                for change_key, change_info in list(st.session_state.edit_history.items()):
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
    if st.session_state.get("show_subproductos_main", True):
        
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
        
    # Mostrar información sobre ajustes de frutas activos
    if st.session_state.get("fruit_adjustments"):
        st.markdown("---")
        st.subheader("🍎 **Impacto de Ajustes de Frutas**")
        
        # Contar SKUs afectados por ajustes de frutas
        if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
            recipe_data = st.session_state.recipe_data
            recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
            
            if not recipe_df.empty:
                # Obtener todas las frutas con ajustes activos
                adjusted_fruits = set()
                for adjustment_info in st.session_state.fruit_adjustments.values():
                    adjusted_fruits.add(adjustment_info["fruta_id"])
                
                # Contar SKUs que usan estas frutas
                affected_skus = set()
                for fruta_id in adjusted_fruits:
                    fruta_skus = recipe_df[recipe_df["fruta_id"] == fruta_id]["SKU"].unique().tolist()
                    affected_skus.update(fruta_skus)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Frutas Modificadas",
                        len(adjusted_fruits),
                        f"de {len(st.session_state.recipe_data.get('fruit_prices', {}))} totales"
                    )
                
                with col2:
                    st.metric(
                        "SKUs Afectados",
                        len(affected_skus),
                        f"de {kpis['Total SKUs']} totales"
                    )
                
                with col3:
                    impact_percentage = (len(affected_skus) / kpis['Total SKUs'] * 100) if kpis['Total SKUs'] > 0 else 0
                    st.metric(
                        "Impacto en Simulación",
                        f"{impact_percentage:.1f}%",
                        f"{len(affected_skus)} SKUs afectados"
                    )
                
                # Mostrar lista de frutas modificadas
                with st.expander("🔍 Ver Frutas Modificadas y SKUs Afectados", expanded=False):
                    for adjustment_info in st.session_state.fruit_adjustments.values():
                        fruta_id = adjustment_info["fruta_id"]
                        fruit_name = adjustment_info["fruit_name"]
                        old_price = adjustment_info["old_price"]
                        new_price = adjustment_info["new_price"]
                        
                        # Obtener SKUs que usan esta fruta
                        fruta_skus = recipe_df[recipe_df["fruta_id"] == fruta_id]["SKU"].unique().tolist()
                        
                        st.write(f"**{fruit_name}** ({fruta_id}): ${old_price:.3f} → ${new_price:.3f}/kg")
                        st.write(f"  - SKUs afectados: {len(fruta_skus)}")
                        if len(fruta_skus) <= 10:
                            st.write(f"  - Lista: {', '.join(map(str, fruta_skus))}")
                        else:
                            st.write(f"  - Primeros 10: {', '.join(map(str, fruta_skus[:10]))}...")
                        st.write("---")
        
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
    st.info("💡 **Navegación**: Usa el menú lateral para volver a la página principal.")
    st.info("💾 **Datos persistentes**: Los cambios se mantienen durante la sesión.")
    st.info("📁 **Requisito**: Debes cargar datos en la página Home antes de usar el simulador.")

# ===================== SEGUNDA PESTAÑA: SIMULACIÓN POR RECETA =====================
with tab2:
    st.header("🍎 Simulación por Receta")
    st.markdown("Simula cambios en precios de frutas base usando fruta_id y analiza impacto en SKUs por receta.")
    
    # Verificar si hay datos de recetas disponibles
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        
        # Mostrar información de recetas cargadas
        st.success(f"✅ **Recetas cargadas**: {len(recipe_data['recipe_df'])} SKUs con recetas disponibles")
        
        # Mostrar fuente de datos
        source = recipe_data.get("source", "Desconocida")
        st.info(f"📊 **Fuente de datos**: {source}")
        
        # Mostrar precios actuales de frutas base usando fruta_id
        st.subheader("💰 Precios Actuales de Frutas Base (por fruta_id)")
        fruit_prices = recipe_data["fruit_prices"]
        fruit_efficiency = recipe_data.get("fruit_efficiency", {})
        fruit_names = recipe_data.get("fruit_names", {})
        
        # Mostrar todas las frutas disponibles en un expander para mejor organización
        with st.expander(f"🌍 Ver todas las {len(fruit_prices)} frutas disponibles (fruta_id)", expanded=False):
            # Crear columnas dinámicas según la cantidad de frutas
            if len(fruit_prices) <= 4:
                cols = st.columns(len(fruit_prices))
                for i, (fruta_id, price) in enumerate(fruit_prices.items()):
                    with cols[i]:
                        eficiencia = fruit_efficiency.get(fruta_id, 0.9)
                        fruit_name = fruit_names.get(fruta_id, fruta_id)
                        st.metric(fruit_name, f"${price:.3f}/kg", f"Eficiencia: {eficiencia:.1%}")
            else:
                # Para muchas frutas, usar un layout de grid
                cols_per_row = 4
                for i in range(0, len(fruit_prices), cols_per_row):
                    row_fruits = list(fruit_prices.items())[i:i+cols_per_row]
                    cols = st.columns(cols_per_row)
                    for j, (fruta_id, price) in enumerate(row_fruits):
                        with cols[j]:
                            eficiencia = fruit_efficiency.get(fruta_id, 0.9)
                            fruit_name = fruit_names.get(fruta_id, fruta_id)
                            st.metric(fruit_name, f"${price:.3f}/kg", f"Eficiencia: {eficiencia:.1%}")
        
        # Mostrar resumen de las frutas principales (primeras 10)
        st.subheader("🍎 Frutas Principales (primeras 10)")
        main_fruits = list(fruit_prices.keys())[:10]  # Primeras 10 frutas
        
        if main_fruits:
            cols = st.columns(min(len(main_fruits), 4))
            for i, fruta_id in enumerate(main_fruits):
                with cols[i]:
                    precio = fruit_prices.get(fruta_id, 0)
                    eficiencia = fruit_efficiency.get(fruta_id, 0.9)
                    fruit_name = fruit_names.get(fruta_id, fruta_id)
                    st.metric(fruit_name, f"${precio:.3f}/kg", f"Eficiencia: {eficiencia:.1%}")
        
        # Simulación de cambios en precios de frutas
        st.subheader("🎯 Simular Cambio en Precio de Fruta Base")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Crear opciones con nombres de frutas para mejor legibilidad
            fruit_options = []
            for fruta_id in fruit_prices.keys():
                fruit_name = fruit_names.get(fruta_id, fruta_id)
                fruit_options.append(f"{fruit_name} ({fruta_id})")
            
            selected_fruit_display = st.selectbox(
                "Seleccionar fruta a modificar:",
                options=fruit_options,
                help="Selecciona la fruta base cuyo precio quieres modificar"
            )
            
            # Extraer fruta_id del display seleccionado
            selected_fruit = selected_fruit_display.split("(")[-1].rstrip(")")
        
        with col2:
            new_price = st.number_input(
                "Nuevo precio (USD/kg):",
                min_value=0.0,
                max_value=100.0,
                value=float(fruit_prices.get(selected_fruit, 0)),
                step=0.01,
                format="%.3f",
                help="Nuevo precio de la fruta base"
            )
        
        with col3:
            if st.button("🚀 Simular Cambio", type="primary"):
                # Simular cambio en precio de fruta
                # from src.recipe_calculator import simulate_fruit_price_change
                
                # Calcular el cambio porcentual
                old_price = fruit_prices.get(selected_fruit, 0)
                if old_price > 0:
                    price_change_pct = ((new_price - old_price) / old_price) * 100
                else:
                    price_change_pct = 0
                
                # new_recipe_prices = simulate_fruit_price_change(
                #     recipe_df=recipe_data["recipe_df"],
                #     fruit_prices=fruit_prices,
                #     fruit_efficiency=recipe_data.get("fruit_efficiency", {}),
                #     fruta_id=selected_fruit,
                #     price_change_pct=price_change_pct,
                #     fruit_names=fruit_names
                # )
                
                # Guardar simulación en sesión
                st.session_state.recipe_simulation = {
                    "fruit": selected_fruit,  # Ahora es fruta_id
                    "fruit_name": fruit_names.get(selected_fruit, selected_fruit),
                    "old_price": old_price,
                    "new_price": new_price,
                    "price_change_pct": price_change_pct,
                    # "new_recipe_prices": new_recipe_prices
                }
                
                st.success(f"✅ Simulación completada: {fruit_names.get(selected_fruit, selected_fruit)} → ${new_price:.3f}/kg ({price_change_pct:+.1f}%)")
                st.rerun()
        
        # Mostrar resultados de simulación si existe
        # if "recipe_simulation" in st.session_state:
        #     sim = st.session_state.recipe_simulation
            
        #     st.subheader(f"📊 Resultados de Simulación: {sim['fruit_name']}")
        #     st.info(f"**Cambio aplicado**: ${sim['old_price']:.3f}/kg → ${sim['new_price']:.3f}/kg ({sim['price_change_pct']:+.1f}%)")
            
        #     # Mostrar tabla de precios recalculados
        #     st.write("**Impacto del cambio en MMPP (Fruta) por SKU:**")
            
        #     # Seleccionar columnas para mostrar
        #     display_columns = ["SKU", "MMPP_Fruta_Calculado_Original", "MMPP_Fruta_Calculado_Modificado", "Cambio_MMPP", "Cambio_MMPP_Pct"]
        #     if "Nombre_Fruta" in sim["new_recipe_prices"].columns:
        #         display_columns.append("Nombre_Fruta")
            
        #     available_display_cols = [col for col in display_columns if col in sim["new_recipe_prices"].columns]
            
        #     st.dataframe(
        #         sim["new_recipe_prices"][available_display_cols],
        #         use_container_width=True
        #     )
            
        #     # Mostrar resumen de impacto
        #     if not sim["new_recipe_prices"].empty:
        #         col1, col2, col3 = st.columns(3)
                
        #         with col1:
        #             skus_afectados = len(sim["new_recipe_prices"])
        #             st.metric("SKUs Afectados", skus_afectados)
                
        #         with col2:
        #             cambio_promedio = sim["new_recipe_prices"]["Cambio_MMPP"].mean()
        #             st.metric("Cambio Promedio", f"${cambio_promedio:.3f}/kg")
                
        #         with col3:
        #             max_cambio = sim["new_recipe_prices"]["Cambio_MMPP"].max()
        #             st.metric("Máximo Cambio", f"${max_cambio:.3f}/kg")
            
        #     # Botón para exportar simulación
        #     csv_sim = sim["new_recipe_prices"].to_csv(index=False)
        #     st.download_button(
        #         label="📥 Exportar Simulación por Receta (CSV)",
        #         data=csv_sim,
        #         file_name=f"simulacion_receta_{sim['fruit_name']}_{sim['price_change_pct']:+.1f}pct.csv",
        #         mime="text/csv",
        #         use_container_width=True,
        #         key="download_recipe_simulation"
        #     )
        
        # Mostrar desglose detallado de recetas
        st.subheader("📋 Desglose Detallado de Recetas (por fruta_id)")
        
        # Crear DataFrame de desglose
        recipe_breakdown = recipe_data["recipe_df"].copy()
        
        # Agregar nombres de frutas si están disponibles
        if fruit_names:
            recipe_breakdown["Nombre_Fruta"] = recipe_breakdown["fruta_id"].map(fruit_names).fillna(recipe_breakdown["fruta_id"])
        
        # Mostrar tabla de desglose
        display_columns = ["SKU", "Descripcion", "Marca", "Cliente", "fruta_id", "Porcentaje"]
        
        if "Nombre_Fruta" in recipe_breakdown.columns:
            display_columns.append("Nombre_Fruta")
        
        if "Porcentaje_Valido" in recipe_breakdown.columns:
            display_columns.append("Porcentaje_Valido")
        
        available_display_cols = [col for col in display_columns if col in recipe_breakdown.columns]
        
        st.dataframe(
            recipe_breakdown[available_display_cols],
            use_container_width=True
        )
        
        # Mostrar comparación de precios original vs receta
        st.subheader("🔍 Comparación: Precios Originales vs Receta")
        
        comparison_df = recipe_data["comparison"]
        if not comparison_df.empty:
            # Filtros para la comparación
            col1, col2 = st.columns([1, 1])
            with col1:
                filter_classification = st.selectbox(
                    "Filtrar por clasificación:",
                    options=["Todos"] + comparison_df["Clasificacion"].unique().tolist(),
                    help="Filtrar SKUs por tipo de diferencia"
                )
            
            with col2:
                if st.button("🔄 Limpiar Filtros", type="secondary"):
                    st.rerun()
            
            # Aplicar filtro
            if filter_classification != "Todos":
                filtered_comparison = comparison_df[comparison_df["Clasificacion"] == filter_classification]
            else:
                filtered_comparison = comparison_df
            
            # Seleccionar columnas para mostrar, incluyendo nombre de fruta si está disponible
            comparison_display_columns = ["SKU", "Descripcion", "fruta_id", "Porcentaje", "Precio_Original", "Precio_Receta", "Diferencia", "Diferencia_Pct", "Clasificacion"]
            
            if "Nombre_Fruta" in filtered_comparison.columns:
                comparison_display_columns.append("Nombre_Fruta")
            
            available_comparison_cols = [col for col in comparison_display_columns if col in filtered_comparison.columns]
            
            # Mostrar tabla de comparación
            st.dataframe(
                filtered_comparison[available_comparison_cols],
                use_container_width=True
            )
            
            # Estadísticas de la comparación
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SKUs Analizados", len(filtered_comparison))
            with col2:
                avg_diff = filtered_comparison["Diferencia_Pct"].mean()
                st.metric("Diferencia Promedio", f"{avg_diff:+.1f}%")
            with col3:
                max_diff = filtered_comparison["Diferencia_Pct"].abs().max()
                st.metric("Mayor Diferencia", f"{max_diff:.1f}%")
        
        # Reporte completo de análisis
        st.subheader("📋 Reporte Completo de Análisis")
        with st.expander("📊 Ver Reporte Detallado", expanded=False):
            report = recipe_data["report"]
            st.text(report)
            
            # Botón para descargar reporte
            st.download_button(
                label="📥 Descargar Reporte de Análisis (TXT)",
                data=report,
                file_name="reporte_analisis_recetas.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_recipe_report"
            )
    
    else:
        st.warning("⚠️ **No hay datos de recetas disponibles**")
        st.info("💡 **Para usar la simulación por receta, necesitas:**")
        st.info("1. 📁 Cargar un archivo Excel con la hoja 'RECETA_SKU'")
        st.info("2. 🔄 La hoja debe contener proporciones de frutas por SKU en la columna 'Porcentaje'")
        st.info("3. 📊 Los datos se procesarán automáticamente")
        
        if st.button("🏠 Ir a Inicio para Cargar Datos", type="primary", use_container_width=True):
            st.switch_page("Inicio.py")
