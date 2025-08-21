"""
Módulo del simulador para análisis de EBITDA y márgenes.
Contiene funciones para simulación de escenarios y análisis de rentabilidad.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import altair as alt
from pathlib import Path
import io
from datetime import datetime

# ===================== Funciones de Filtrado =====================
def apply_filters(df: pd.DataFrame, cliente: List[str] = None, marca: List[str] = None, 
                 especie: List[str] = None, condicion: List[str] = None) -> pd.DataFrame:
    """
    Aplica filtros a la base de datos.
    
    Args:
        df: DataFrame base
        cliente: Lista de clientes a incluir
        marca: Lista de marcas a incluir
        especie: Lista de especies a incluir
        condicion: Lista de condiciones a incluir
        
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    # Aplicar filtros solo si están especificados y no están vacíos
    if cliente and len(cliente) > 0 and "Todos" not in cliente:
        df_filtered = df_filtered[df_filtered["Cliente"].isin(cliente)]
    
    if marca and len(marca) > 0 and "Todos" not in marca:
        df_filtered = df_filtered[df_filtered["Marca"].isin(marca)]
    
    if especie and len(especie) > 0 and "Todos" not in especie:
        df_filtered = df_filtered[df_filtered["Especie"].isin(especie)]
    
    if condicion and len(condicion) > 0 and "Todos" not in condicion:
        df_filtered = df_filtered[df_filtered["Condicion"].isin(condicion)]
    
    return df_filtered

def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtiene opciones de filtro con valores únicos de cada columna.
    
    Args:
        df: DataFrame base
        
    Returns:
        Diccionario con columna -> lista de valores únicos
    """
    filter_columns = ["Cliente", "Marca", "Especie", "Condicion"]
    options = {}
    
    for col in filter_columns:
        if col in df.columns:
            values = df[col].dropna().unique().tolist()
            values = sorted([str(v) for v in values if str(v).strip()])
            options[col] = ["Todos"] + values
        else:
            options[col] = ["Todos"]
    
    return options

# ===================== Funciones de Overrides =====================
def apply_global_overrides(df: pd.DataFrame, pct_costo: float, enabled: bool) -> pd.DataFrame:
    """
    Aplica cambios porcentuales globales a los costos.
    
    Args:
        df: DataFrame base
        pct_costo: Porcentaje de cambio (-100 a +1000)
        enabled: Si se debe aplicar el override
        
    Returns:
        DataFrame con costos modificados
    """
    if not enabled or abs(pct_costo) < 0.01:
        return df.copy()
    
    df_modified = df.copy()
    
    # Aplicar cambio porcentual a costos - usar columnas que realmente existen
    cost_columns = []
    
    # Buscar columnas de costos disponibles
    if "Costos Totales (USD/kg)" in df_modified.columns:
        cost_columns.append("Costos Totales (USD/kg)")
    if "Retail Costos Directos (USD/kg)" in df_modified.columns:
        cost_columns.append("Retail Costos Directos (USD/kg)")
    if "Retail Costos Indirectos (USD/kg)" in df_modified.columns:
        cost_columns.append("Retail Costos Indirectos (USD/kg)")
    if "MMPP Total (USD/kg)" in df_modified.columns:
        cost_columns.append("MMPP Total (USD/kg)")
    if "Guarda MMPP" in df_modified.columns:
        cost_columns.append("Guarda MMPP")
    
    # Si no hay columnas de costos específicas, usar las que contengan "USD/kg"
    if not cost_columns:
        cost_columns = [col for col in df_modified.columns if "USD/kg" in col and "Precio" not in col]
    
    # Aplicar cambios a las columnas de costos encontradas
    for col in cost_columns:
        if col in df_modified.columns:
            df_modified[f"{col}_Original"] = df_modified[col]
            df_modified[col] = df_modified[col] * (1 + pct_costo / 100)
    
    # Recalcular EBITDA si es posible
    if "PrecioVenta (USD/kg)" in df_modified.columns and "Costos Totales (USD/kg)" in df_modified.columns:
        df_modified["EBITDA (USD/kg)"] = df_modified["PrecioVenta (USD/kg)"] - df_modified["Costos Totales (USD/kg)"]
        
        # Recalcular EBITDA Pct
        df_modified["EBITDA Pct"] = np.where(
            df_modified["PrecioVenta (USD/kg)"].abs() > 1e-12,
            (df_modified["EBITDA (USD/kg)"] / df_modified["PrecioVenta (USD/kg)"]) * 100,
            0.0
        )
    
    return df_modified

def apply_upload_overrides(df: pd.DataFrame, uploaded_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Aplica overrides desde archivo subido.
    
    Args:
        df: DataFrame base
        uploaded_df: DataFrame con SKU y CostoNuevo
        
    Returns:
        Tuple con (DataFrame modificado, número de SKUs actualizados)
    """
    if uploaded_df is None or uploaded_df.empty:
        return df.copy(), 0
    
    # Validar columnas requeridas
    required_cols = ["SKU", "CostoNuevo"]
    if not all(col in uploaded_df.columns for col in required_cols):
        st.error("❌ El archivo debe contener las columnas: SKU, CostoNuevo")
        return df.copy(), 0
    
    df_modified = df.copy()
    updated_count = 0
    
    # Buscar columna de costos principales para actualizar
    cost_column = None
    if "Costos Totales (USD/kg)" in df_modified.columns:
        cost_column = "Costos Totales (USD/kg)"
    elif "CostoUSD_kg" in df_modified.columns:
        cost_column = "CostoUSD_kg"
    else:
        # Buscar cualquier columna de costos
        cost_columns = [col for col in df_modified.columns if "USD/kg" in col and "Precio" not in col]
        if cost_columns:
            cost_column = cost_columns[0]
    
    if not cost_column:
        st.error("❌ No se encontró columna de costos para actualizar")
        return df.copy(), 0
    
    # Crear columna de respaldo si no existe
    if f"{cost_column}_Original" not in df_modified.columns:
        df_modified[f"{cost_column}_Original"] = df_modified[cost_column]
    
    # Aplicar overrides
    for _, row in uploaded_df.iterrows():
        sku = str(row["SKU"]).strip()
        costo_nuevo = pd.to_numeric(row["CostoNuevo"], errors="coerce")
        
        if pd.isna(costo_nuevo):
            continue
            
        # Buscar SKU en la base
        mask = df_modified["SKU"] == sku
        if mask.any():
            df_modified.loc[mask, cost_column] = costo_nuevo
            updated_count += 1
    
    if updated_count > 0:
        # Recalcular EBITDA si es posible
        if "PrecioVenta (USD/kg)" in df_modified.columns and cost_column in df_modified.columns:
            df_modified["EBITDA (USD/kg)"] = df_modified["PrecioVenta (USD/kg)"] - df_modified[cost_column]
            
            # Recalcular EBITDA Pct
            df_modified["EBITDA Pct"] = np.where(
                df_modified["PrecioVenta (USD/kg)"].abs() > 1e-12,
                (df_modified["EBITDA (USD/kg)"] / df_modified["PrecioVenta (USD/kg)"]) * 100,
                0.0
            )
        
        st.success(f"✅ Se actualizaron {updated_count} SKUs desde el archivo")
    else:
        st.warning("⚠️ No se encontraron SKUs coincidentes en la base")
    
    return df_modified, updated_count

# ===================== Cálculo de EBITDA =====================
def compute_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula EBITDA para cada SKU.
    
    Args:
        df: DataFrame con precios y costos
        
    Returns:
        DataFrame con EBITDA calculado
    """
    df_ebitda = df.copy()
    
    # Buscar columnas de precio y costos disponibles
    price_column = None
    cost_column = None
    
    if "PrecioVenta (USD/kg)" in df_ebitda.columns:
        price_column = "PrecioVenta (USD/kg)"
    elif "PrecioUSD_kg" in df_ebitda.columns:
        price_column = "PrecioUSD_kg"
    
    if "Costos Totales (USD/kg)" in df_ebitda.columns:
        cost_column = "Costos Totales (USD/kg)"
    elif "CostoUSD_kg" in df_ebitda.columns:
        cost_column = "CostoUSD_kg"
    
    # Si no hay columnas específicas, calcular costos totales
    if not cost_column:
        cost_columns = [col for col in df_ebitda.columns if "USD/kg" in col and "Precio" not in col]
        if cost_columns:
            df_ebitda["Costos Totales (USD/kg)"] = df_ebitda[cost_columns].sum(axis=1)
            cost_column = "Costos Totales (USD/kg)"
    
    # Asegurar que las columnas numéricas existen
    if not price_column:
        df_ebitda["PrecioVenta (USD/kg)"] = 0.0
        price_column = "PrecioVenta (USD/kg)"
    
    if not cost_column:
        df_ebitda["Costos Totales (USD/kg)"] = 0.0
        cost_column = "Costos Totales (USD/kg)"
    
    # Calcular EBITDA
    df_ebitda["EBITDA (USD/kg)"] = df_ebitda[price_column] - df_ebitda[cost_column]
    
    # Calcular margen porcentual
    df_ebitda["EBITDA Pct"] = np.where(
        df_ebitda[price_column].abs() > 1e-12,
        (df_ebitda["EBITDA (USD/kg)"] / df_ebitda[price_column]) * 100,
        0.0
    )
    
    return df_ebitda

# ===================== Análisis y KPIs =====================
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula KPIs principales del DataFrame.
    
    Args:
        df: DataFrame con EBITDA calculado
        
    Returns:
        Diccionario con KPIs
    """
    if df.empty:
        return {
            "EBITDA Promedio (USD/kg)": 0.0,
            "EBITDA Total (USD)": 0.0,
            "SKUs Rentables": 0,
            "EBITDA Promedio (%)": 0.0,
            "Total SKUs": 0
        }
    
    kpis = {}
    
    # Buscar columnas de EBITDA disponibles
    ebitda_column = "EBITDA (USD/kg)" if "EBITDA (USD/kg)" in df.columns else "EBITDAUSD_kg"
    ebitda_pct_column = "EBITDA Pct" if "EBITDA Pct" in df.columns else "MargenPct"
    
    if ebitda_column in df.columns:
        # EBITDA promedio
        kpis["EBITDA Promedio (USD/kg)"] = df[ebitda_column].mean()
        
        # EBITDA total (asumiendo 1 kg por SKU para el cálculo)
        kpis["EBITDA Total (USD)"] = df[ebitda_column].sum()
        
        # SKUs rentables
        kpis["SKUs Rentables"] = (df[ebitda_column] > 0).sum()
        
        # Total SKUs
        kpis["Total SKUs"] = len(df)
    else:
        kpis["EBITDA Promedio (USD/kg)"] = 0.0
        kpis["EBITDA Total (USD)"] = 0.0
        kpis["SKUs Rentables"] = 0
        kpis["Total SKUs"] = len(df)
    
    # Margen promedio
    if ebitda_pct_column in df.columns:
        kpis["EBITDA Promedio (%)"] = df[ebitda_pct_column].mean()
    else:
        kpis["EBITDA Promedio (%)"] = 0.0
    
    return kpis

def get_top_bottom_skus(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtiene los top N y bottom N SKUs por EBITDA.
    
    Args:
        df: DataFrame con EBITDA
        n: Número de SKUs a mostrar
        
    Returns:
        Tuple con (top_n_df, bottom_n_df)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Buscar columnas disponibles
    ebitda_column = "EBITDA (USD/kg)" if "EBITDA (USD/kg)" in df.columns else "EBITDAUSD_kg"
    ebitda_pct_column = "EBITDA Pct" if "EBITDA Pct" in df.columns else "MargenPct"
    
    if ebitda_column not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Top N por EBITDA
    top_n = df.nlargest(n, ebitda_column)[["SKU", "Cliente", "Marca", ebitda_column, ebitda_pct_column]].copy()
    
    # Bottom N por EBITDA
    bottom_n = df.nsmallest(n, ebitda_column)[["SKU", "Cliente", "Marca", ebitda_column, ebitda_pct_column]].copy()
    
    return top_n, bottom_n

# ===================== Gráficos =====================
def create_ebitda_chart(df: pd.DataFrame, top_n: int = 20) -> alt.Chart:
    """
    Crea gráfico de barras de EBITDA por SKU.
    
    Args:
        df: DataFrame con EBITDA
        top_n: Número de SKUs a mostrar
        
    Returns:
        Chart de Altair
    """
    if df.empty:
        return None
    
    # Buscar columnas disponibles
    ebitda_column = "EBITDA (USD/kg)" if "EBITDA (USD/kg)" in df.columns else "EBITDAUSD_kg"
    ebitda_pct_column = "EBITDA Pct" if "EBITDA Pct" in df.columns else "MargenPct"
    price_column = "PrecioVenta (USD/kg)" if "PrecioVenta (USD/kg)" in df.columns else "PrecioUSD_kg"
    cost_column = "Costos Totales (USD/kg)" if "Costos Totales (USD/kg)" in df.columns else "CostoTotalUSD_kg"
    
    if ebitda_column not in df.columns:
        return None
    
    # Obtener top N SKUs por EBITDA
    chart_data = df.nlargest(top_n, ebitda_column).copy()
    # Convertir SKU y Cliente a string antes de concatenar
    chart_data["SKU_Cliente"] = chart_data["SKU"].astype(str) + " - " + chart_data["Cliente"].astype(str)
    
    # Crear gráfico
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X(f"{ebitda_column}:Q", title="EBITDA (USD/kg)"),
        y=alt.Y("SKU_Cliente:N", title="SKU - Cliente", sort="-x"),
        color=alt.condition(
            alt.datum[ebitda_column] > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=[
            alt.Tooltip("SKU_Cliente:N", title="SKU - Cliente"),
            alt.Tooltip(f"{ebitda_column}:Q", title="EBITDA (USD/kg)", format=".3f"),
            alt.Tooltip(f"{ebitda_pct_column}:Q", title="Margen %", format=".1f"),
            alt.Tooltip(f"{price_column}:Q", title="Precio (USD/kg)", format=".3f"),
            alt.Tooltip(f"{cost_column}:Q", title="Costo Total (USD/kg)", format=".3f")
        ]
    ).properties(
        title=f"Top {top_n} SKUs por EBITDA",
        width=600,
        height=400
    )
    
    return chart

def create_margin_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Crea gráfico de distribución de márgenes.
    
    Args:
        df: DataFrame con márgenes
        
    Returns:
        Chart de Altair
    """
    if df.empty:
        return None
    
    # Buscar columna de margen disponible
    margin_column = "EBITDA Pct" if "EBITDA Pct" in df.columns else "MargenPct"
    
    if margin_column not in df.columns:
        return None
    
    # Crear bins para el histograma
    chart_data = df.copy()
    chart_data["MargenBin"] = pd.cut(
        chart_data[margin_column], 
        bins=20, 
        labels=False
    )
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X(f"{margin_column}:Q", title="Margen (%)", bin=alt.Bin(maxbins=20)),
        y=alt.Y("count():Q", title="Número de SKUs"),
        tooltip=[
            alt.Tooltip(f"{margin_column}:Q", title="Margen %", bin=True),
            alt.Tooltip("count():Q", title="Número de SKUs")
        ]
    ).properties(
        title="Distribución de Márgenes",
        width=600,
        height=300
    )
    
    return chart

# ===================== Export =====================
def export_escenario(df: pd.DataFrame, filename_prefix: str = "escenario") -> Path:
    """
    Exporta el escenario a CSV.
    
    Args:
        df: DataFrame a exportar
        filename_prefix: Prefijo del nombre del archivo
        
    Returns:
        Path al archivo exportado
    """
    # Crear directorio outputs si no existe
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = outputs_dir / filename
    
    # Exportar a CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    return filepath

# ===================== Funciones de Utilidad =====================
def format_currency(value: float) -> str:
    """Formatea un valor como moneda USD."""
    return f"${value:.3f}"

def format_percentage(value: float) -> str:
    """Formatea un valor como porcentaje."""
    return f"{value:.1f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """División segura que evita división por cero."""
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator

def validate_upload_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Valida archivo subido para overrides.
    
    Args:
        uploaded_file: Archivo subido a Streamlit
        
    Returns:
        Tuple con (es_válido, mensaje, DataFrame)
    """
    if uploaded_file is None:
        return False, "No se seleccionó ningún archivo", None
    
    try:
        # Leer archivo según extensión
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return False, "Formato de archivo no soportado. Use .csv, .xlsx o .xls", None
        
        # Validar columnas requeridas
        required_cols = ["SKU", "CostoNuevo"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Columnas faltantes: {', '.join(missing_cols)}", None
        
        # Validar que CostoNuevo sea numérico
        if not pd.api.types.is_numeric_dtype(df["CostoNuevo"]):
            return False, "La columna 'CostoNuevo' debe ser numérica", None
        
        return True, f"Archivo válido con {len(df)} filas", df
        
    except Exception as e:
        return False, f"Error leyendo archivo: {e}", None
