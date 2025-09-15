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
from src.data_io import compute_mmpp_unified

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
        # Filtro especial para especies: buscar si alguna de las especies seleccionadas
        # está contenida en la columna Especie (que ahora es una lista)
        if "Especie" in df_filtered.columns:
            def has_matching_species(sku_especies):
                if isinstance(sku_especies, list):
                    # Si es una lista, verificar si alguna especie seleccionada está en la lista
                    return any(esp in especie for esp in sku_especies)
                else:
                    # Si es string (fallback), verificar si alguna especie seleccionada está en el string
                    especies_str = str(sku_especies)
                    return any(esp in especies_str for esp in especie)
            
            especie_mask = df_filtered["Especie"].apply(has_matching_species)
            df_filtered = df_filtered[especie_mask]
    
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
            if col == "Especie":
                # Para especies, extraer todas las especies individuales de las listas
                all_species = set()
                for especies_list in df[col].dropna():
                    if isinstance(especies_list, list):
                        # Si es una lista, agregar todas las especies
                        all_species.update(especies_list)
                    else:
                        # Si es string (fallback), dividir por coma
                        especies_str = str(especies_list)
                        especies_split = [esp.strip() for esp in especies_str.split(",")]
                        all_species.update(especies_split)
                
                values = sorted([esp for esp in all_species if esp and esp.strip()])
            else:
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

def create_margin_distribution_chart(
    df: pd.DataFrame,
    *,
    col_candidates=("EBITDA Pct", "MargenPct"),
    method="iqr",           # "iqr" | "quantile"
    iqr_k=1.5,              # factor para IQR
    q_low=0.01, q_high=0.99,# percentiles si method="quantile"
    maxbins=30,             # bins del histograma
    clip_label=True         # mostrar en el título el recorte aplicado
) -> alt.Chart | None:
    """
    Histograma robusto de márgenes, recortando outliers SOLO para la visualización.
    """
    if df is None or df.empty:
        return None

    # 1) Detectar la columna de margen
    margin_column = next((c for c in col_candidates if c in df.columns), None)
    if margin_column is None:
        return None

    # 2) Serie limpia
    s = pd.to_numeric(df[margin_column], errors="coerce").dropna()
    if s.empty:
        return None

    # 3) Calcular límites robustos
    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
    else:  # "quantile"
        lo, hi = s.quantile([q_low, q_high])

    # Evitar lo > hi por datos degenerados
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = s.min(), s.max()

    # 4) Filtrar SOLO para el gráfico
    mask = (s >= lo) & (s <= hi)
    clipped = (~mask).sum()
    chart_data = pd.DataFrame({margin_column: s[mask]})

    # 5) Estadísticos (de la serie original, no recortada)
    mean_val = float(s.mean())
    med_val  = float(s.median())

    # 6) Construir histograma
    base = alt.Chart(chart_data)

    hist = base.mark_bar().encode(
        x=alt.X(f"{margin_column}:Q",
                title="Margen (%)",
                bin=alt.Bin(maxbins=maxbins)),
        y=alt.Y("count():Q", title="Número de SKUs"),
        tooltip=[
            alt.Tooltip(f"{margin_column}:Q", title="Margen %", bin=True),
            alt.Tooltip("count():Q", title="Número de SKUs")
        ]
    )

    # 7) Líneas de media y mediana (sobre la escala recortada)
    mean_rule = alt.Chart(pd.DataFrame({margin_column: [mean_val]})).mark_rule(strokeWidth=2).encode(
        x=alt.X(f"{margin_column}:Q"),
        color=alt.value("#555")
    )
    mean_text = mean_rule.mark_text(align="left", dx=5, dy=-5).encode(
        text=alt.value(f"Media: {mean_val:.1f}%")
    )

    med_rule = alt.Chart(pd.DataFrame({margin_column: [med_val]})).mark_rule(strokeDash=[4,4], strokeWidth=2).encode(
        x=alt.X(f"{margin_column}:Q"),
        color=alt.value("#999")
    )
    med_text = med_rule.mark_text(align="left", dx=5, dy=12).encode(
        text=alt.value(f"Mediana: {med_val:.1f}%")
    )

    # 8) Título con nota de recorte (opcional)
    title = "Distribución de Márgenes"
    if clip_label and clipped > 0:
        title += f" (recortado a [{lo:.1f}%, {hi:.1f}%], {clipped} outliers fuera)"

    chart = (hist + mean_rule + mean_text + med_rule + med_text).properties(
        title=title,
        width=600,
        height=320
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

# ===================== Funciones de Simulación de Granel =====================

def apply_granel_filters(df: pd.DataFrame, fruta: List[str] = None) -> pd.DataFrame:
    """
    Aplica filtros a los datos de granel.
    
    Args:
        df: DataFrame de granel
        fruta: Lista de frutas a incluir
        
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    # Aplicar filtros solo si están especificados y no están vacíos
    if fruta and len(fruta) > 0 and "Todos" not in fruta:
        df_filtered = df_filtered[df_filtered["Fruta_id"].isin(fruta)]
    
    return df_filtered

def get_granel_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtiene opciones de filtro para datos de granel.
    
    Args:
        df: DataFrame de granel
        
    Returns:
        Diccionario con opciones de filtro
    """
    options = {}
    
    if "Fruta_id" in df.columns and "Fruta" in df.columns:
        # Crear un DataFrame único con Fruta_id y Fruta para mantener la correspondencia
        unique_frutas = df[["Fruta_id", "Fruta"]].drop_duplicates().sort_values("Fruta")
        
        # Devolver listas correspondientes
        options["id"] = unique_frutas["Fruta_id"].tolist()
        options["Fruta"] = unique_frutas["Fruta"].tolist()
    
    return options

def apply_granel_global_overrides(df: pd.DataFrame, pct_change: float, enable: bool = False) -> pd.DataFrame:
    """
    Aplica overrides globales a los datos de granel.
    
    Args:
        df: DataFrame de granel
        pct_change: Porcentaje de cambio
        enable: Si aplicar el override
        
    Returns:
        DataFrame con overrides aplicados
    """
    if not enable or abs(pct_change) < 0.01:
        return df.copy()
    
    df_adjusted = df.copy()
    
    # Identificar columnas de costos (excluyendo identificadores y totales)
    cost_columns = [col for col in df_adjusted.columns 
                   if col not in ["Fruta_id", "Fruta", "Precio", "Rendimiento", "Precio Efectivo", 
                                 "Costos Directos", "Costos Indirectos"] 
                   and not col.endswith("Total")]
    
    # Aplicar cambio porcentual a costos
    for col in cost_columns:
        if col in df_adjusted.columns:
            df_adjusted[col] = df_adjusted[col] * (1 + pct_change / 100)
    
    # Recalcular totales
    df_adjusted = recalculate_granel_totals(df_adjusted)
    
    return df_adjusted

def recalculate_granel_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula los totales en los datos de granel.
    
    Args:
        df: DataFrame de granel
        
    Returns:
        DataFrame con totales recalculados
    """
    df_calc = df.copy()
    
    # Recalcular MO Total si existen las componentes
    if "MO Directa" in df_calc.columns and "MO Indirecta" in df_calc.columns:
        df_calc["MO Total"] = df_calc["MO Directa"] + df_calc["MO Indirecta"]
    
    # Recalcular Materiales Total si existen las componentes
    if "Materiales Directos" in df_calc.columns and "Materiales Indirectos" in df_calc.columns:
        df_calc["Materiales Total"] = df_calc["Materiales Directos"] + df_calc["Materiales Indirectos"]
    
    # Recalcular Costos Directos
    direct_cost_cols = ["MO Directa", "Materiales Directos", "Laboratorio", "Mantencion y Maquinaria"]
    existing_direct_cols = [col for col in direct_cost_cols if col in df_calc.columns]
    if existing_direct_cols:
        df_calc["Costos Directos"] = df_calc[existing_direct_cols].sum(axis=1)
    
    # Recalcular Costos Indirectos
    indirect_cost_cols = ["MO Indirecta", "Materiales Indirectos"]
    existing_indirect_cols = [col for col in indirect_cost_cols if col in df_calc.columns]
    if existing_indirect_cols:
        df_calc["Costos Indirectos"] = df_calc[existing_indirect_cols].sum(axis=1)
    
    # Recalcular Precio Efectivo si existen Precio y Rendimiento
    if "Precio" in df_calc.columns and "Rendimiento" in df_calc.columns:
        df_calc["Precio Efectivo"] = df_calc["Precio"] / df_calc["Rendimiento"]
    
    return df_calc

def apply_granel_universal_adjustments(df: pd.DataFrame, adjustments: dict) -> pd.DataFrame:
    """
    Aplica ajustes universales a los datos de granel.
    
    Args:
        df: DataFrame de granel
        adjustments: Diccionario con ajustes por columna
        
    Returns:
        DataFrame con ajustes aplicados
    """
    if not adjustments:
        return df

    df_adjusted = df.copy()

    for cost_column, adj in adjustments.items():
        if cost_column not in df_adjusted.columns:
            continue

        if adj["type"] == "percentage":
            # CORREGIDO: Aplicar cambio porcentual correctamente
            # +20% = multiplicar por 1.2, -20% = multiplicar por 0.8
            df_adjusted[cost_column] = df_adjusted[cost_column] * (1 + adj["value"] / 100)
        else:  # "dollars" = nuevo valor absoluto
            df_adjusted[cost_column] = adj["value"]

    # Recalcular totales
    df_adjusted = recalculate_granel_totals(df_adjusted)
    return df_adjusted

def calculate_granel_kpis(df: pd.DataFrame) -> Dict:
    """
    Calcula KPIs para datos de granel.
    
    Args:
        df: DataFrame de granel
        
    Returns:
        Diccionario con KPIs calculados
    """
    kpis = {}
    
    try:
        kpis["Total Frutas"] = len(df)
        
        if "MO Total" in df.columns:
            kpis["MO Promedio (USD/kg)"] = df["MO Total"].mean()
        
        if "Materiales Total" in df.columns:
            kpis["Materiales Promedio (USD/kg)"] = df["Materiales Total"].mean()
        
        if "Laboratorio" in df.columns:
            kpis["Laboratorio Promedio (USD/kg)"] = df["Laboratorio"].mean()
        
        if "Precio Efectivo" in df.columns:
            kpis["Precio Efectivo Promedio (USD/kg)"] = df["Precio Efectivo"].mean()
        
        if "Costos Directos" in df.columns:
            kpis["Costos Directos Promedio (USD/kg)"] = df["Costos Directos"].mean()
        
        if "Costos Indirectos" in df.columns:
            kpis["Costos Indirectos Promedio (USD/kg)"] = df["Costos Indirectos"].mean()
        
    except Exception as e:
        st.error(f"Error calculando KPIs de granel: {e}")
    
    return kpis

def get_top_bottom_granel(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtiene las frutas con mayor y menor costo total.
    
    Args:
        df: DataFrame de granel
        n: Número de frutas a retornar
        
    Returns:
        Tupla con (top_frutas, bottom_frutas)
    """
    if "Precio Efectivo" in df.columns:
        # Ordenar por Precio Efectivo (mayor a menor)
        df_sorted = df.sort_values("Precio Efectivo", ascending=False)
        top_frutas = df_sorted.head(n)
        bottom_frutas = df_sorted.tail(n)
    else:
        # Si no hay Precio Efectivo, usar Costos Directos como alternativa
        if "Costos Directos" in df.columns:
            df_sorted = df.sort_values("Costos Directos", ascending=False)
            top_frutas = df_sorted.head(n)
            bottom_frutas = df_sorted.tail(n)
        else:
            top_frutas = pd.DataFrame()
            bottom_frutas = pd.DataFrame()
    
    return top_frutas, bottom_frutas

def create_granel_cost_chart(df: pd.DataFrame, top_n: int = 20) -> Optional[alt.Chart]:
    """
    Crea un gráfico de barras con los costos de granel por fruta.
    
    Args:
        df: DataFrame de granel
        top_n: Número de frutas a mostrar
        
    Returns:
        Gráfico de Altair o None si hay error
    """
    try:
        if df.empty:
            return None
        
        # Seleccionar las top N frutas por Precio Efectivo
        if "Precio Efectivo" in df.columns:
            df_chart = df.nlargest(top_n, "Precio Efectivo")
            y_col = "Precio Efectivo"
            title = f"Top {top_n} Frutas por Precio Efectivo (USD/kg)"
        elif "Costos Directos" in df.columns:
            df_chart = df.nlargest(top_n, "Costos Directos")
            y_col = "Costos Directos"
            title = f"Top {top_n} Frutas por Costos Directos (USD/kg)"
        else:
            return None
        
        # Crear gráfico
        chart = alt.Chart(df_chart).mark_bar().add_selection(
            alt.selection_interval()
        ).encode(
            x=alt.X("Fruta:N", sort="-y", title="Fruta"),
            y=alt.Y(f"{y_col}:Q", title="Costo (USD/kg)"),
            color=alt.Color(f"{y_col}:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Fruta", "Fruta", f"{y_col}:Q"]
        ).properties(
            title=title,
            width=600,
            height=400
        )
        
        return chart
        
    except Exception as e:
        st.error(f"Error creando gráfico de granel: {e}")
        return None


def sync_granel_changes_to_retail(granel_df: pd.DataFrame, receta_df: pd.DataFrame, info_df: pd.DataFrame, retail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sincroniza los cambios de costos de granel con el simulador de retail,
    recalculando el "Proceso Granel (USD/kg)" para cada SKU.
    
    Args:
        granel_df: DataFrame de granel con costos actualizados
        receta_df: DataFrame de recetas SKU
        info_df: DataFrame de información de frutas
        retail_df: DataFrame de retail actual
        
    Returns:
        DataFrame de retail con "Proceso Granel (USD/kg)" actualizado
    """
    try:
        # Usar la función unificada
        updated_costs = compute_mmpp_unified(receta_df, info_df, granel_df)
        
        if updated_costs.empty:
            print("ERROR: No se pudieron calcular los costos actualizados")
            return retail_df
        
        # Actualizar el DataFrame de retail
        retail_updated = retail_df.copy()
        
        # Verificar que tenemos las columnas necesarias en retail
        required_cols = ["Proceso Granel (USD/kg)", "MMPP (Fruta) (USD/kg)", "Almacenaje MMPP"]
        missing_cols = [col for col in required_cols if col not in retail_updated.columns]
        if missing_cols:
            print(f"ERROR: Faltan columnas en retail_df: {missing_cols}")
            return retail_df
        
        # Actualizar usando mapeo directo
        for col in ["Proceso Granel (USD/kg)", "MMPP (Fruta) (USD/kg)", "Almacenaje MMPP"]:
            if col in updated_costs.columns:
                # Mapear columna de almacenaje
                map_col = "Almacenaje" if col == "Almacenaje MMPP" else col
                
                cost_map = updated_costs.set_index("SKU")[map_col]
                retail_updated[col] = retail_updated["SKU"].astype(str).map(
                    cost_map.reindex(cost_map.index.astype(str))
                ).fillna(retail_updated[col])
        
        # Recalcular totales en retail
        from src.data_io import recalculate_totals
        retail_updated = recalculate_totals(retail_updated)
        
        return retail_updated
        
    except Exception as e:
        print(f"ERROR sincronizando cambios de granel con retail: {e}")
        import traceback
        traceback.print_exc()
        return retail_df

def export_granel_escenario(df: pd.DataFrame, filename_prefix: str = "escenario_granel") -> Path:
    """
    Exporta un escenario de granel a CSV.
    
    Args:
        df: DataFrame de granel
        filename_prefix: Prefijo del archivo
        
    Returns:
        Path del archivo exportado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Crear directorio outputs si no existe
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    
    return filepath
