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

# ===================== Funciones de simulación =====================
def apply_simulation(df: pd.DataFrame, price_up: float = 0.0, 
                    retail_direct_up: float = 0.0, retail_indirect_up: float = 0.0,
                    mmpp_up: float = 0.0, guarda_up: float = 0.0) -> pd.DataFrame:
    """
    Aplica multiplicadores de simulación por grupo (globales).
    
    Args:
        df: DataFrame base con datos originales
        price_up: Variación porcentual en precio de venta
        retail_direct_up: Variación porcentual en costos retail directos
        retail_indirect_up: Variación porcentual en costos retail indirectos
        mmpp_up: Variación porcentual en costos MMPP
        guarda_up: Variación porcentual en costos de guarda
        
    Returns:
        DataFrame con columnas *_Sim y deltas
    """
    sim = df.copy()
    
    # Verificar qué columnas están disponibles
    available_cols = set(df.columns)
    
    # Aplicar variaciones a precios
    if "PrecioVenta (USD/kg)" in available_cols:
        sim["PrecioVenta (USD/kg)_Sim"] = df["PrecioVenta (USD/kg)"] * (1 + price_up/100.0)
    else:
        st.warning("⚠️ Columna 'PrecioVenta (USD/kg)' no encontrada")
        return df
    
    # Aplicar variaciones a costos retail
    if "Retail Costos Directos (USD/kg)" in available_cols:
        sim["Retail Costos Directos (USD/kg)_Sim"] = df["Retail Costos Directos (USD/kg)"] * (1 + retail_direct_up/100.0)
    else:
        sim["Retail Costos Directos (USD/kg)_Sim"] = 0.0
        
    if "Retail Costos Indirectos (USD/kg)" in available_cols:
        sim["Retail Costos Indirectos (USD/kg)_Sim"] = df["Retail Costos Indirectos (USD/kg)"] * (1 + retail_indirect_up/100.0)
    else:
        sim["Retail Costos Indirectos (USD/kg)_Sim"] = 0.0
    
    # Aplicar variaciones a costos MMPP
    mmpp_total = 0.0
    
    if "MMPP (Fruta) (USD/kg)" in available_cols:
        sim["MMPP (Fruta) (USD/kg)_Sim"] = df["MMPP (Fruta) (USD/kg)"] * (1 + mmpp_up/100.0)
        mmpp_total += sim["MMPP (Fruta) (USD/kg)_Sim"]
    else:
        sim["MMPP (Fruta) (USD/kg)_Sim"] = 0.0
        
    if "MMPP (Proceso Granel) (USD/kg)" in available_cols:
        sim["MMPP (Proceso Granel) (USD/kg)_Sim"] = df["MMPP (Proceso Granel) (USD/kg)"] * (1 + mmpp_up/100.0)
        mmpp_total += sim["MMPP (Proceso Granel) (USD/kg)_Sim"]
    else:
        sim["MMPP (Proceso Granel) (USD/kg)_Sim"] = 0.0
    
    # Calcular MMPP Total si no existe
    if "MMPP Total (USD/kg)" in available_cols:
        sim["MMPP Total (USD/kg)_Sim"] = df["MMPP Total (USD/kg)"] * (1 + mmpp_up/100.0)
    else:
        sim["MMPP Total (USD/kg)_Sim"] = mmpp_total
    
    # Aplicar variaciones a costos de guarda
    if "Guarda MMPP" in available_cols:
        sim["Guarda MMPP_Sim"] = df["Guarda MMPP"] * (1 + guarda_up/100.0)
    else:
        sim["Guarda MMPP_Sim"] = 0.0
    
    # Calcular costos totales simulados
    sim["Costos Totales (USD/kg)_Sim"] = (
        sim["Retail Costos Directos (USD/kg)_Sim"] + 
        sim["Retail Costos Indirectos (USD/kg)_Sim"] + 
        sim["MMPP Total (USD/kg)_Sim"] + 
        sim["Guarda MMPP_Sim"]
    )
    
    # Calcular gastos totales simulados
    sim["Gastos Totales (USD/kg)_Sim"] = (
        sim["Retail Costos Directos (USD/kg)_Sim"] + 
        sim["Retail Costos Indirectos (USD/kg)_Sim"] + 
        sim["Guarda MMPP_Sim"] + 
        sim["MMPP (Proceso Granel) (USD/kg)_Sim"]
    )
    
    # Calcular EBITDA simulado
    sim["EBITDA (USD/kg)_Sim"] = sim["PrecioVenta (USD/kg)_Sim"] - sim["Costos Totales (USD/kg)_Sim"]
    sim["EBITDA Pct_Sim"] = np.where(
        sim["PrecioVenta (USD/kg)_Sim"].abs() > 1e-12,
        sim["EBITDA (USD/kg)_Sim"] / sim["PrecioVenta (USD/kg)_Sim"],
        np.nan
    )
    
    # Calcular deltas solo si las columnas originales existen
    if "EBITDA (USD/kg)" in available_cols:
        sim["Δ_EBITDA (USD/kg)"] = sim["EBITDA (USD/kg)_Sim"] - sim["EBITDA (USD/kg)"]
    else:
        sim["Δ_EBITDA (USD/kg)"] = 0.0
        
    if "EBITDA Pct" in available_cols:
        sim["Δ_EBITDA Pct"] = (sim["EBITDA Pct_Sim"] - sim["EBITDA Pct"]) * 100
    else:
        sim["Δ_EBITDA Pct"] = 0.0
    
    return sim

def apply_row_specific_overrides(df: pd.DataFrame, overrides: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Aplica overrides específicos por fila (SKU).
    
    Args:
        df: DataFrame base
        overrides: Dict con SKU como clave y dict de columnas:valor como valor
        
    Returns:
        DataFrame con overrides aplicados
    """
    df_override = df.copy()
    
    for sku, column_overrides in overrides.items():
        if sku in df_override.index:
            for column, value in column_overrides.items():
                if column in df_override.columns:
                    df_override.loc[sku, column] = value
    
    return df_override

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula KPIs principales del DataFrame.
    
    Args:
        df: DataFrame con datos
        
    Returns:
        Dict con KPIs calculados
    """
    kpis = {}
    
    # KPIs de EBITDA
    kpis["EBITDA Promedio (USD/kg)"] = df["EBITDA (USD/kg)"].mean()
    kpis["EBITDA Total (USD)"] = df["EBITDA (USD/kg)"].sum()
    kpis["EBITDA Promedio (%)"] = df["EBITDA Pct"].mean()
    
    # KPIs de rentabilidad
    kpis["SKUs Rentables"] = len(df[df["EBITDA (USD/kg)"] > 0])
    kpis["SKUs No Rentables"] = len(df[df["EBITDA (USD/kg)"] <= 0])
    kpis["% SKUs Rentables"] = (kpis["SKUs Rentables"] / len(df)) * 100
    
    # KPIs de costos
    kpis["Costo Promedio (USD/kg)"] = df["Costos Totales (USD/kg)"].mean()
    kpis["Precio Promedio (USD/kg)"] = df["PrecioVenta (USD/kg)"].mean()
    
    # KPIs de margen
    kpis["Margen Promedio (%)"] = ((df["PrecioVenta (USD/kg)"] - df["Costos Totales (USD/kg)"]) / df["PrecioVenta (USD/kg)"]).mean() * 100
    
    return kpis

def create_ebitda_chart(df: pd.DataFrame, group_by: str = "Marca") -> alt.Chart:
    """
    Crea un gráfico de EBITDA agrupado por categoría.
    
    Args:
        df: DataFrame con datos
        group_by: Columna para agrupar (Marca, Especie, Cliente, etc.)
        
    Returns:
        Chart de Altair
    """
    if group_by not in df.columns:
        st.warning(f"Columna {group_by} no encontrada en los datos")
        return None
    
    # Agrupar datos
    chart_data = df.groupby(group_by).agg({
        "EBITDA (USD/kg)": "mean",
        "SKU": "count"
    }).reset_index()
    chart_data.columns = [group_by, "EBITDA Promedio (USD/kg)", "Cantidad SKUs"]
    
    # Crear gráfico
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X(f"{group_by}:N", title=group_by),
        y=alt.Y("EBITDA Promedio (USD/kg):Q", title="EBITDA Promedio (USD/kg)"),
        color=alt.Color("EBITDA Promedio (USD/kg):Q", scale=alt.Scale(scheme="redblue")),
        tooltip=[group_by, "EBITDA Promedio (USD/kg)", "Cantidad SKUs"]
    ).properties(
        title=f"EBITDA Promedio por {group_by}",
        width=600,
        height=400
    )
    
    return chart

def create_margin_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Crea un gráfico de distribución de márgenes.
    
    Args:
        df: DataFrame con datos
        
    Returns:
        Chart de Altair
    """
    # Crear bins para el histograma
    margin_data = df[["EBITDA Pct"]].copy()
    margin_data = margin_data.dropna()
    
    chart = alt.Chart(margin_data).mark_bar().encode(
        x=alt.X("EBITDA Pct:Q", bin=alt.Bin(step=5), title="EBITDA (%)"),
        y=alt.Y("count():Q", title="Cantidad de SKUs"),
        tooltip=["EBITDA Pct", "count()"]
    ).properties(
        title="Distribución de EBITDA por SKU",
        width=600,
        height=400
    )
    
    return chart

def export_scenario(df: pd.DataFrame, scenario_name: str = "escenario_simulado") -> bytes:
    """
    Exporta el escenario simulado a Excel.
    
    Args:
        df: DataFrame con datos del escenario
        scenario_name: Nombre del archivo
        
    Returns:
        Bytes del archivo Excel
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Hoja principal con datos simulados
        df.to_excel(writer, sheet_name="Escenario_Simulado", index=False)
        
        # Hoja con KPIs
        kpis = calculate_kpis(df)
        kpis_df = pd.DataFrame(list(kpis.items()), columns=["KPI", "Valor"])
        kpis_df.to_excel(writer, sheet_name="KPIs", index=False)
        
        # Hoja con resumen por grupo
        for group_col in ["Marca", "Especie", "Cliente"]:
            if group_col in df.columns:
                summary = df.groupby(group_col).agg({
                    "EBITDA (USD/kg)": ["mean", "min", "max", "sum"],
                    "EBITDA Pct": "mean",
                    "SKU": "count"
                }).round(3)
                summary.columns = ["EBITDA Promedio", "EBITDA Min", "EBITDA Max", "EBITDA Total", "EBITDA % Promedio", "Cantidad SKUs"]
                summary.to_excel(writer, sheet_name=f"Resumen_{group_col}")
    
    output.seek(0)
    return output.getvalue()

def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtiene opciones de filtro para cada campo dimensional.
    
    Args:
        df: DataFrame con datos
        
    Returns:
        Dict con opciones de filtro por campo
    """
    filter_options = {}
    
    dimensional_fields = ["Marca", "Especie", "Cliente", "Condicion"]
    
    for field in dimensional_fields:
        if field in df.columns:
            options = sorted(df[field].dropna().unique().tolist())
            filter_options[field] = options
    
    return filter_options

def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Aplica filtros al DataFrame.
    
    Args:
        df: DataFrame base
        filters: Dict con campo:valores_seleccionados
        
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    for field, values in filters.items():
        if field in df_filtered.columns and values:
            df_filtered = df_filtered[df_filtered[field].isin(values)]
    
    return df_filtered

def create_scenario_summary(df_original: pd.DataFrame, df_simulated: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un resumen comparativo entre escenario original y simulado.
    
    Args:
        df_original: DataFrame con datos originales
        df_simulated: DataFrame con datos simulados
        
    Returns:
        DataFrame con resumen comparativo
    """
    summary = pd.DataFrame({
        "Métrica": [
            "EBITDA Promedio (USD/kg)",
            "EBITDA Total (USD)",
            "EBITDA Promedio (%)",
            "SKUs Rentables",
            "SKUs No Rentables",
            "% SKUs Rentables"
        ],
        "Escenario Original": [
            df_original["EBITDA (USD/kg)"].mean(),
            df_original["EBITDA (USD/kg)"].sum(),
            df_original["EBITDA Pct"].mean(),
            len(df_original[df_original["EBITDA (USD/kg)"] > 0]),
            len(df_original[df_original["EBITDA (USD/kg)"] <= 0]),
            (len(df_original[df_original["EBITDA (USD/kg)"] > 0]) / len(df_original)) * 100
        ],
        "Escenario Simulado": [
            df_simulated["EBITDA (USD/kg)_Sim"].mean(),
            df_simulated["EBITDA (USD/kg)_Sim"].sum(),
            df_simulated["EBITDA Pct_Sim"].mean(),
            len(df_simulated[df_simulated["EBITDA (USD/kg)_Sim"] > 0]),
            len(df_simulated[df_simulated["EBITDA (USD/kg)_Sim"] <= 0]),
            (len(df_simulated[df_simulated["EBITDA (USD/kg)_Sim"] > 0]) / len(df_simulated)) * 100
        ]
    })
    
    # Calcular variaciones
    summary["Variación"] = summary["Escenario Simulado"] - summary["Escenario Original"]
    summary["Variación %"] = (summary["Variación"] / summary["Escenario Original"]) * 100
    
    return summary.round(3)

# ===================== Funciones de análisis avanzado =====================
def identify_critical_skus(df: pd.DataFrame, threshold: float = -0.1) -> pd.DataFrame:
    """
    Identifica SKUs críticos (con EBITDA negativo o bajo).
    
    Args:
        df: DataFrame con datos
        threshold: Umbral de EBITDA para considerar crítico
        
    Returns:
        DataFrame con SKUs críticos
    """
    critical = df[df["EBITDA Pct"] < threshold].copy()
    critical = critical.sort_values("EBITDA Pct")
    
    return critical

def analyze_cost_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la estructura de costos por SKU.
    
    Args:
        df: DataFrame con datos
        
    Returns:
        DataFrame con análisis de estructura de costos
    """
    cost_cols = [col for col in df.columns if "USD/kg" in col and "Sim" not in col and "PrecioVenta" not in col]
    
    analysis = df[["SKU"] + cost_cols].copy()
    
    # Calcular porcentajes de cada componente
    for col in cost_cols:
        analysis[f"{col}_Pct"] = (analysis[col] / analysis[cost_cols].sum(axis=1)) * 100
    
    return analysis

def calculate_break_even_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula análisis de punto de equilibrio.
    
    Args:
        df: DataFrame con datos
        
    Returns:
        DataFrame con análisis de break-even
    """
    be_analysis = df[["SKU", "PrecioVenta (USD/kg)", "Costos Totales (USD/kg)", "EBITDA (USD/kg)"]].copy()
    
    # Calcular margen de contribución
    be_analysis["Margen_Contribucion"] = be_analysis["PrecioVenta (USD/kg)"] - be_analysis["Costos Totales (USD/kg)"]
    
    # Calcular margen de contribución porcentual
    be_analysis["Margen_Contribucion_Pct"] = (
        be_analysis["Margen_Contribucion"] / be_analysis["PrecioVenta (USD/kg)"]
    ) * 100
    
    # Identificar si está por encima del break-even
    be_analysis["Por_Encima_BE"] = be_analysis["Margen_Contribucion"] > 0
    
    return be_analysis
