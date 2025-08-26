"""
Módulo de entrada/salida de datos para la aplicación de costos y márgenes.
Contiene funciones para cargar, procesar y validar datos desde archivos Excel.
"""

import io
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Optional
import streamlit as st

# ===================== Configuración =====================
REQ_SHEETS = {
    "FACT_COSTOS_POND": "Tabla con costos unitarios ponderados por SKU (Oct-Jun).",
    "FACT_PRECIOS": "Precios mensuales: SKU, Año, Mes, PrecioVentaUSD.",
    "DIM_SKU": "Dimensión de SKU (opcional, para filtrar por Marca/Especie/Cliente).",
    "RECETA_SKU": "Recetas de SKUs con proporciones de frutas base (opcional).",
    "INFO_FRUTA": "Información de frutas base con precios y eficiencias (opcional)."
}

# Hojas obligatorias (las que SIEMPRE deben estar presentes)
REQUIRED_SHEETS = ["FACT_COSTOS_POND", "FACT_PRECIOS"]

# Hojas opcionales (las que pueden estar o no)
OPTIONAL_SHEETS = ["DIM_SKU", "RECETA_SKU", "INFO_FRUTA"]

MESES_ORD = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
             "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MES2NUM = {m:i+1 for i,m in enumerate(MESES_ORD)}

# ===================== Utilidades =====================
def to_number_safe(x, comma_decimal=True):
    """Convierte '1.234,56' o '1,234.56' o '3,071' -> float. '-' o vacío -> NaN."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\xa0"," ")
    if s in {"", "-", "—"}: return np.nan
    s = s.replace(" ", "")
    if comma_decimal:
        # Si parece '1.234,56' o '3,071'
        if "," in s and (s.count(".") <= 1):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def month_to_num(m):
    """Convierte nombre de mes a número (1-12)."""
    return MES2NUM.get(str(m).strip().title(), np.nan)

def ensure_str(df, col):
    """Asegura que una columna sea string y la limpia."""
    df[col] = df[col].astype(str).str.strip()
    return df

def _norm_text(s: str) -> str:
    """Normaliza texto eliminando acentos y caracteres especiales."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.strip().split())

def bytes_key(file):
    """Genera una clave reproducible para cachear por contenido del archivo subido."""
    if file is None:
        return None
    pos = file.tell()
    data = file.read()
    file.seek(pos)
    return hash(data)

# ===================== Carga y validación =====================
@st.cache_data(show_spinner=False)
def read_workbook(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Lee el Excel completo en dict de DataFrames (todas las hojas).
    
    Args:
        uploaded_bytes: Bytes del archivo Excel subido
        
    Returns:
        Dict con nombre de hoja como clave y DataFrame como valor
    """
    bio = io.BytesIO(uploaded_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheets = {name: xls.parse(name, dtype=str) for name in xls.sheet_names}
    return sheets

def validate_required_sheets(sheets: Dict[str, pd.DataFrame]) -> list:
    """
    Valida que existan las hojas OBLIGATORIAS (no las opcionales).
    
    Args:
        sheets: Dict de hojas cargadas
        
    Returns:
        Lista de hojas obligatorias faltantes
    """
    missing = [s for s in REQUIRED_SHEETS if s not in sheets]
    return missing

# ===================== Procesamiento de datos =====================
def build_tbl_costos_pond(df_costos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa la tabla de costos ponderados y retorna resumen y detalle.
    
    Args:
        df_costos: DataFrame con los costos
        
    Returns:
        Tuple con (detalle_df, resumen_df)
    """
    df0 = df_costos.copy()
    df0.columns = [_norm_text(c) for c in df0.columns]

    # Detectar columna SKU (exacta, no SKU-Cliente)
    col_sku = None
    for c in df0.columns:
        if _norm_text(c).lower() == "sku":
            col_sku = c
            break
    if col_sku is None:
        raise ValueError("No se encontró columna SKU en la hoja de costos.")

    # Renombrar columnas conocidas
    rename_map = {}
    for c in df0.columns:
        lc = _norm_text(c).lower()
        if lc == "mo_directa":
            rename_map[c] = "MO Directa"
        elif lc == "mo_indirecta":
            rename_map[c] = "MO Indirecta"
        elif lc == "mo_total":
            rename_map[c] = "MO Total"
        elif lc == "materiales_cajas_y_bolsas":
            rename_map[c] = "Materiales Cajas y Bolsas"
        elif lc == "materiales_indirectos":
            rename_map[c] = "Materiales Indirectos"
        elif lc == "materiales_total":
            rename_map[c] = "Materiales Total"
        elif lc == "calidad":
            rename_map[c] = "Laboratorio"
        elif lc == "mantencion":
            rename_map[c] = "Mantención"
        elif lc == "sgenerales":
            rename_map[c] = "Servicios Generales"
        elif lc == "utilities":
            rename_map[c] = "Utilities"
        elif lc == "fletes":
            rename_map[c] = "Fletes Internos"
        elif lc == "comex":
            rename_map[c] = "Comex"
        elif lc == "guarda_pt":
            rename_map[c] = "Guarda Producto Terminado"
        elif lc == "guarda_mmpp":
            rename_map[c] = "Guarda MMPP"
        elif lc == "mmpp_fruta":
            rename_map[c] = "MMPP (Fruta) (USD/kg)"
        elif lc == "proceso granel":
            rename_map[c] = "Proceso Granel (USD/kg)"
        elif lc == "mmpp_total":
            rename_map[c] = "MMPP Total (USD/kg)"
        elif lc == "dir retail":
            rename_map[c] = "Retail Costos Directos (USD/kg)"
        elif lc == "ind retail":
            rename_map[c] = "Retail Costos Indirectos (USD/kg)"
        elif lc == "total":
            rename_map[c] = "Costos Totales (USD/kg)"

    df = df0.rename(columns=rename_map).copy()
    df = df.rename(columns={col_sku: "SKU"})
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Calcular gastos totales solo si todas las columnas existen
    gastos_cols = ["Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)", "Guarda MMPP", "Proceso Granel (USD/kg)"]
    if all(col in df.columns for col in gastos_cols):
        df["Gastos Totales (USD/kg)"] = (
            df["Retail Costos Directos (USD/kg)"].astype(float) + 
            df["Retail Costos Indirectos (USD/kg)"].astype(float) + 
            df["Guarda MMPP"].astype(float) + 
            df["Proceso Granel (USD/kg)"].astype(float)
        )
    else:
        df["Gastos Totales (USD/kg)"] = np.nan

    # Convertir valores a numéricos
    for c in df.columns:
        if c == "SKU":
            continue
        df[c] = df[c].apply(to_number_safe)
    # Quedarse con SKU + columnas de costos
    cost_cols = [c for c in df.columns if c != "SKU" and c not in ["Condicion", "Marca", "Descripcion"]]
    out = df[["SKU"] + cost_cols].dropna(subset=["SKU"]).reset_index(drop=True)

    return out

def build_fact_precios(df_p: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa la tabla de precios y retorna precios limpios con FechaClave.
    
    Args:
        df_p: DataFrame con precios
        
    Returns:
        DataFrame procesado con precios limpios
    """
    needed = {"SKU","SKU-Cliente","Año","Mes","PrecioVentaUSD"}
    if not needed.issubset(set(df_p.columns)):
        raise ValueError(f"FACT_PRECIOS debe contener {needed}. Columnas: {df_p.columns.tolist()}")

    p = df_p.copy()
    p.columns = [c.strip() for c in p.columns]
    p = ensure_str(p, "SKU")
    p = ensure_str(p, "SKU-Cliente")
    p["Año"] = p["Año"].apply(lambda x: int(str(x).strip()))
    p["MesNum"] = p["Mes"].apply(month_to_num).astype("Int64")
    p["PrecioVentaUSD"] = p["PrecioVentaUSD"].apply(to_number_safe)
    p = p.dropna(subset=["PrecioVentaUSD"])
    p["FechaClave"] = p["Año"]*100 + p["MesNum"].astype(int)
    return p

def build_dim_sku(df_dim: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa la dimensión SKU y retorna tabla limpia.
    
    Args:
        df_dim: DataFrame con dimensión SKU
        
    Returns:
        DataFrame procesado con dimensión limpia
    """
    dim = df_dim.copy()
    dim.columns = [c.strip() for c in dim.columns]
    if "SKU" not in dim.columns or "SKU-Cliente" not in dim.columns:
        raise ValueError("En 'DIM_SKU' no se encontró columna 'SKU' o 'SKU-Cliente")

    # Asegura columnas esperadas
    expected = ["SKU","SKU-Cliente","Condicion", "Descripcion", "Marca", "Especie", "Cliente"]
    for c in expected:
        if c not in dim.columns:
            dim[c] = np.nan

    # Limpieza básica
    for c in expected:
        dim[c] = dim[c].astype(str).str.strip()

    # Eliminar duplicados por SKU-Cliente
    dim = dim[expected].drop_duplicates(subset=["SKU-Cliente"], keep="first").reset_index(drop=True)
    return dim

def compute_latest_price(precios: pd.DataFrame, mode="global", ref_datekey=None) -> pd.DataFrame:
    """
    Calcula el último precio por SKU-Cliente.
    
    Args:
        precios: DataFrame con precios
        mode: Modo de cálculo ("global" o "to_date")
        ref_datekey: Fecha de referencia para modo "to_date"
        
    Returns:
        DataFrame con último precio por SKU-Cliente
    """
    p = precios.sort_values(["SKU-Cliente","FechaClave"]).reset_index(drop=True)
    if mode == "to_date":
        if ref_datekey is None:
            raise ValueError("ref_datekey es requerido con mode='to_date'.")
        p = p[p["FechaClave"] <= ref_datekey]
    idx = p.groupby("SKU-Cliente")["FechaClave"].idxmax()
    latest = p.loc[idx, ["SKU-Cliente","PrecioVentaUSD","FechaClave"]].rename(
        columns={"PrecioVentaUSD":"PrecioVenta (USD/kg)"})
    return latest.reset_index(drop=True)

# ===================== Construcción del mart =====================
@st.cache_data(show_spinner=True)
def build_detalle(uploaded_bytes: bytes, ultimo_precio_modo: str, ref_ym: Optional[int]) -> pd.DataFrame:
    """
    Pipeline completo para construir el detalle de datos.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel subido
        ultimo_precio_modo: Modo de cálculo de último precio
        ref_ym: Año-mes de referencia (YYYYMM)
        
    Returns:
        DataFrame con detalle de datos
    """
    sheets = read_workbook(uploaded_bytes)
    
    # Validación de hojas requeridas
    missing = validate_required_sheets(sheets)
    if missing:
        raise ValueError(f"Faltan hojas requeridas: {missing}")

    # 1) Costos ponderados
    costos_detalle = build_tbl_costos_pond(sheets["FACT_COSTOS_POND"])

    # 2) Precios + último precio por SKU
    precios = build_fact_precios(sheets["FACT_PRECIOS"])
    if ultimo_precio_modo == "global":
        latest = compute_latest_price(precios, mode="global")
    else:
        latest = compute_latest_price(precios, mode="to_date", ref_datekey=ref_ym)

    # 3) DIM_SKU
    dim = build_dim_sku(sheets["DIM_SKU"])

    # 4) Unión de tablas
    detalle = costos_detalle.merge(dim, on="SKU", how="right")
    # Si ambos tienen SKU-Cliente, unir por esa columna; si no, unir por SKU
    if "SKU-Cliente" in dim.columns:
        detalle = detalle.merge(latest, on="SKU-Cliente", how="right")
    else:
        detalle = detalle.merge(latest, on="SKU", how="right")
    detalle = detalle.drop(columns=["FechaClave"])
    
    # 4.1) Normalización de especies (consolidar variaciones y estandarizar en inglés)
    try:
        from src.species_normalizer import normalize_species_column
        if "Especie" in detalle.columns:
            # Crear columna de respaldo antes de normalizar
            detalle["Especie_Original"] = detalle["Especie"].copy()
            # Aplicar normalización
            detalle = normalize_species_column(detalle, "Especie")
    except ImportError:
        # Si no se puede importar el normalizador, continuar sin normalización
        pass
    
    # 4.2) Procesamiento de recetas (si está disponible)
    recipe_data = None
    
    try:
        from src.recipe_calculator import process_recipe_data
        if "RECETA_SKU" in sheets:
            recipe_data = process_recipe_data(sheets, detalle)
            
            if recipe_data["success"]:
                # Guardar datos de recetas en la sesión para uso posterior
                st.session_state.recipe_data = recipe_data
                
                # 4.3) Calcular MMPP (Fruta) desde recetas
                detalle = calculate_mmpp_from_recipes(detalle, recipe_data)
                
            else:
                st.warning(f"⚠️ Error en sistema de recetas: {recipe_data.get('error', 'Error desconocido')}")
        else:
            st.warning("⚠️ **RECETA_SKU NO encontrada** - no se pueden procesar recetas")
            
    except ImportError as e:
        # Si no se puede importar el calculador de recetas, continuar sin recetas
        st.warning(f"⚠️ No se pudo importar el sistema de recetas: {e}")
        pass
    except Exception as e:
        st.error(f"❌ **Error inesperado procesando recetas**: {e}")
        pass
    
    # 5) Conversión a numérico y cálculo de métricas
    numeric_columns = detalle.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        detalle[col] = pd.to_numeric(detalle[col], errors='coerce').fillna(0.0)
    
    # FORZAR SIGNOS CORRECTOS
    # Los costos siempre deben ser negativos
    
    cost_columns = [col for col in detalle.columns if "USD/kg" in col and "Precio" not in col]
    for col in cost_columns:
        if col in detalle.columns:
            # Convertir valores a negativos (costos siempre negativos)
            detalle[col] = -abs(detalle[col])
    
    # También corregir columnas de costos sin USD/kg
    other_cost_columns = ["MO Directa", "MO Indirecta", "Materiales Cajas y Bolsas", 
                         "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", 
                         "Utilities", "Fletes Internos", "Comex", "Guarda PT"]
    for col in other_cost_columns:
        if col in detalle.columns:
            # Convertir valores a negativos (costos siempre negativos)
            detalle[col] = -abs(detalle[col])
    
    # El precio de venta siempre debe ser positivo
    if "PrecioVenta (USD/kg)" in detalle.columns:
        detalle["PrecioVenta (USD/kg)"] = abs(detalle["PrecioVenta (USD/kg)"])
    
    # 1. Recalcular MMPP Total si están los componentes
    mmpp_components = [
        "MMPP (Fruta) (USD/kg)",
        "Proceso Granel (USD/kg)"
    ]
    
    if all(col in detalle.columns for col in mmpp_components):
        detalle["MMPP Total (USD/kg)"] = detalle[mmpp_components].sum(axis=1)
    
    # 2. Recalcular MO Total si están los componentes
    mo_components = [
        "MO Directa",
        "MO Indirecta"
    ]
    
    if all(col in detalle.columns for col in mo_components):
        detalle["MO Total"] = detalle[mo_components].sum(axis=1)
    
    # 3. Recalcular Materiales Total si están los componentes
    materiales_components = [
        "Materiales Cajas y Bolsas",
        "Materiales Indirectos"
    ]
    
    if all(col in detalle.columns for col in materiales_components):
        detalle["Materiales Total"] = detalle[materiales_components].sum(axis=1)

    # 4.1 Recalcular Retail Costos Directos (USD/kg) si están los componentes
    retail_costs_direct_components = [
        "MO Directa",
        "Materiales Cajas y Bolsas",
        "Laboratorio",
        "Mantención",
        "Servicios Generales",
        "Utilities",
        "Fletes Internos",
        "Comex",
        "Guarda PT",
    ]
    
    if all(col in detalle.columns for col in retail_costs_direct_components):
        detalle["Retail Costos Directos (USD/kg)"] = detalle[retail_costs_direct_components].sum(axis=1)
    
    # 4.2 Recalcular Retail Costos Indirectos (USD/kg) si están los componentes
    retail_costs_indirect_components = [
        "MO Indirecta",
        "Materiales Indirectos",
    ]
    if all(col in detalle.columns for col in retail_costs_indirect_components):
        detalle["Retail Costos Indirectos (USD/kg)"] = detalle[retail_costs_indirect_components].sum(axis=1)
    
    # 4. Recalcular Gastos Totales (costos indirectos - NO incluye MMPP)
    gastos_components = [
        "Guarda MMPP",
        "Proceso Granel (USD/kg)",
        "Retail Costos Indirectos (USD/kg)",
        "Retail Costos Directos (USD/kg)"
    ]
    
    # Solo incluir componentes que existan en el DataFrame
    available_gastos = [col for col in gastos_components if col in detalle.columns]
    if available_gastos:
        detalle["Gastos Totales (USD/kg)"] = detalle[available_gastos].sum(axis=1)
    
    # 5. Recalcular Costos Totales (MMPP + Gastos)
    costos_components = []
    
    # Agregar MMPP Total si existe
    if "MMPP (Fruta) (USD/kg)" in detalle.columns:
        costos_components.append("MMPP (Fruta) (USD/kg)")

    # Agregar Gastos Totales si existe
    if "Gastos Totales (USD/kg)" in detalle.columns:
        costos_components.append("Gastos Totales (USD/kg)")
    
    # Calcular costos totales
    if costos_components:
        detalle["Costos Totales (USD/kg)"] = detalle[costos_components].sum(axis=1)
    
    detalle["EBITDA (USD/kg)"] = detalle["PrecioVenta (USD/kg)"] - detalle["Costos Totales (USD/kg)"].abs()
    detalle["EBITDA Pct"] = np.where(
        detalle["PrecioVenta (USD/kg)"].abs() > 1e-12,
        (detalle["EBITDA (USD/kg)"] / detalle["PrecioVenta (USD/kg)"]) * 100,  # ✅ Multiplicar por 100 para porcentaje
        np.nan
    )

    # 7) Orden final
    detalle["SKU"] = detalle["SKU"].astype(int)
    # Ordenar por SKU-Cliente que es el identificador único real
    if "SKU-Cliente" in detalle.columns:
        detalle = detalle.sort_values("SKU-Cliente", ascending=True).reset_index(drop=True)
    else:
        detalle = detalle.sort_values("SKU", ascending=True).reset_index(drop=True)
    return detalle

def calculate_mmpp_from_recipes(detalle_df: pd.DataFrame, recipe_data: Dict) -> pd.DataFrame:
    """
    Calcula MMPP (Fruta) desde las recetas y lo agrega al DataFrame de detalle.
    
    Args:
        detalle_df: DataFrame con datos de detalle
        recipe_data: Diccionario con datos de recetas procesadas
        
    Returns:
        DataFrame con MMPP (Fruta) calculado desde recetas
    """
    try:
        # Crear copia del DataFrame para no modificar el original
        detalle_updated = detalle_df.copy()
        
        # Obtener datos de recetas
        recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
        fruit_prices = recipe_data.get("fruit_prices", {})
        fruit_efficiency = recipe_data.get("fruit_efficiency", {})
        
        if recipe_df.empty or not fruit_prices:
            st.warning("⚠️ No hay datos de recetas suficientes para calcular MMPP (Fruta)")
            return detalle_updated
        
        # Crear diccionario de precios por receta para cada SKU
        sku_recipe_prices = {}
        
        # Agrupar por SKU para manejar recetas con múltiples frutas
        for sku in recipe_df["SKU"].unique():
            sku_recipes = recipe_df[recipe_df["SKU"] == sku]
            
            if not sku_recipes.empty:
                total_mmpp = 0.0
                total_percentage = 0.0
                recipe_valid = True
                
                # Calcular MMPP total para este SKU sumando todas sus frutas
                for _, recipe_row in sku_recipes.iterrows():
                    fruta_id = recipe_row.get("fruta_id", "")
                    porcentaje = recipe_row.get("Porcentaje", 0)
                    
                    if fruta_id and porcentaje > 0 and fruta_id in fruit_prices:
                        # Calcular precio por receta
                        fruit_cost_per_kg = fruit_prices[fruta_id]
                        efficiency = fruit_efficiency.get(fruta_id, 0.9)
                        
                        # MMPP (Fruta) = (Porcentaje / 100) * Precio fruta / Eficiencia
                        mmpp_fruta = (porcentaje / 100) * fruit_cost_per_kg / efficiency
                        
                        total_mmpp += mmpp_fruta
                        total_percentage += porcentaje
                    
                    # Verificar si la receta es válida
                    if not recipe_row.get("Porcentaje_Valido", True):
                        recipe_valid = False
                
                # Solo incluir SKUs con recetas válidas y porcentaje total razonable
                if recipe_valid and total_percentage <= 100.1:  # Tolerancia de 0.1%
                    sku_recipe_prices[sku] = total_mmpp
        
        # Aplicar precios calculados al DataFrame de detalle
        mmpp_calculated_count = 0
        mmpp_updated_count = 0
        
        for idx, row in detalle_updated.iterrows():
            sku = str(row.get("SKU", "")).strip()
            
            if sku in sku_recipe_prices:
                mmpp_calculated = sku_recipe_prices[sku]
                
                # Verificar si ya existe la columna MMPP (Fruta) (USD/kg)
                if "MMPP (Fruta) (USD/kg)" in detalle_updated.columns:
                    # Comparar con el valor existente
                    existing_mmpp = row.get("MMPP (Fruta) (USD/kg)", 0)
                    
                    if abs(existing_mmpp - mmpp_calculated) > 0.001:  # Tolerancia de 0.001
                        detalle_updated.loc[idx, "MMPP (Fruta) (USD/kg)"] = mmpp_calculated
                        mmpp_updated_count += 1
                    else:
                        mmpp_calculated_count += 1
                else:
                    # Crear la columna si no existe
                    detalle_updated["MMPP (Fruta) (USD/kg)"] = 0.0
                    detalle_updated.loc[idx, "MMPP (Fruta) (USD/kg)"] = mmpp_calculated
                    mmpp_updated_count += 1
        
        # Mostrar estadísticas del cálculo
        if mmpp_updated_count > 0:
            pass
        if mmpp_calculated_count > 0:
            pass
        
        # Recalcular MMPP Total si existe la columna
        if "MMPP (Fruta) (USD/kg)" in detalle_updated.columns and "Proceso Granel (USD/kg)" in detalle_updated.columns:
            detalle_updated["MMPP Total (USD/kg)"] = (
                detalle_updated["MMPP (Fruta) (USD/kg)"] + 
                detalle_updated["Proceso Granel (USD/kg)"]
            )
        
        # Recalcular costos totales y EBITDA
        detalle_updated = recalculate_totals_with_mmpp(detalle_updated)
        
        return detalle_updated
        
    except Exception as e:
        st.error(f"❌ Error calculando MMPP desde recetas: {e}")
        return detalle_df

def recalculate_totals_with_mmpp(detalle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula totales incluyendo MMPP (Fruta) calculado desde recetas.
    
    Args:
        detalle_df: DataFrame con datos de detalle
        
    Returns:
        DataFrame con totales recalculados
    """
    try:
        # Crear copia para no modificar el original
        detalle_calc = detalle_df.copy()
        
        # Convertir columnas numéricas y limpiar valores inválidos
        numeric_columns = detalle_calc.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            detalle_calc[col] = pd.to_numeric(detalle_calc[col], errors='coerce').fillna(0.0)
        
        # FORZAR SIGNOS CORRECTOS
        # Los costos siempre deben ser negativos
        cost_columns = [col for col in detalle_calc.columns if "USD/kg" in col and "Precio" not in col]
        for col in cost_columns:
            if col in detalle_calc.columns:
                # Convertir valores a negativos (costos siempre negativos)
                detalle_calc[col] = -abs(detalle_calc[col])
        
        # También corregir columnas de costos sin USD/kg
        other_cost_columns = ["MO Directa", "MO Indirecta", "Materiales Cajas y Bolsas", 
                             "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", 
                             "Utilities", "Fletes Internos", "Comex", "Guarda PT"]
        for col in other_cost_columns:
            if col in detalle_calc.columns:
                # Convertir valores a negativos (costos siempre negativos)
                detalle_calc[col] = -abs(detalle_calc[col])
        
        # El precio de venta siempre debe ser positivo
        if "PrecioVenta (USD/kg)" in detalle_calc.columns:
            detalle_calc["PrecioVenta (USD/kg)"] = abs(detalle_calc["PrecioVenta (USD/kg)"])
        
        # Recalcular MMPP Total si están los componentes
        mmpp_components = [
            "MMPP (Fruta) (USD/kg)",
            "Proceso Granel (USD/kg)"
        ]
        
        if all(col in detalle_calc.columns for col in mmpp_components):
            detalle_calc["MMPP Total (USD/kg)"] = detalle_calc[mmpp_components].sum(axis=1)
        
        # Recalcular Gastos Totales (costos indirectos - NO incluye MMPP)
        gastos_components = [
            "Guarda MMPP",
            "Proceso Granel (USD/kg)",
            "Retail Costos Indirectos (USD/kg)",
            "Retail Costos Directos (USD/kg)"
        ]
        
        # Solo incluir componentes que existan en el DataFrame
        available_gastos = [col for col in gastos_components if col in detalle_calc.columns]
        if available_gastos:
            detalle_calc["Gastos Totales (USD/kg)"] = detalle_calc[available_gastos].sum(axis=1)
        
        # Recalcular Costos Totales (MMPP + Gastos)
        costos_components = []
        
        # Agregar MMPP Total si existe
        if "MMPP Total (USD/kg)" in detalle_calc.columns:
            costos_components.append("MMPP Total (USD/kg)")
        
        # Agregar Gastos Totales si existe
        if "Gastos Totales (USD/kg)" in detalle_calc.columns:
            costos_components.append("Gastos Totales (USD/kg)")
        
        # Calcular costos totales
        if costos_components:
            detalle_calc["Costos Totales (USD/kg)"] = detalle_calc[costos_components].sum(axis=1)
        
        # Recalcular EBITDA usando COSTOS TOTALES
        if "PrecioVenta (USD/kg)" in detalle_calc.columns and "Costos Totales (USD/kg)" in detalle_calc.columns:
            # Asegurar que los valores sean numéricos válidos
            precio = pd.to_numeric(detalle_calc["PrecioVenta (USD/kg)"], errors='coerce').fillna(0.0)
            costos = pd.to_numeric(detalle_calc["Costos Totales (USD/kg)"], errors='coerce').fillna(0.0)
            
            # Los costos ya están en valor absoluto (negativos), pero para el cálculo EBITDA los convertimos a positivos
            costos_abs = abs(costos)
            
            detalle_calc["EBITDA (USD/kg)"] = precio - costos_abs
            
            # Recalcular EBITDA Pct
            detalle_calc["EBITDA Pct"] = np.where(
                precio > 1e-12,
                (detalle_calc["EBITDA (USD/kg)"] / precio) * 100,
                0.0
            )
        
        return detalle_calc
        
    except Exception as e:
        st.error(f"❌ Error recalculando totales: {e}")
        return detalle_df

# ===================== Documentación para nuevas fuentes =====================
"""
INSTRUCCIONES PARA AGREGAR NUEVAS FUENTES DE DATOS:

1. FACT_COSTOS_POND:
   - Agregar nueva hoja al Excel con columnas: SKU + costos por componente
   - Modificar build_tbl_costos_pond() para incluir nuevos mapeos de columnas
   - Actualizar num_cols en build_detalle() si hay nuevas columnas numéricas

2. DIM_SKU:
   - Agregar nueva hoja con columnas: SKU + atributos dimensionales
   - Modificar build_dim_sku() para incluir nuevas columnas esperadas
   - Actualizar expected en build_dim_sku()

3. FACT_PRECIOS:
   - Agregar nueva hoja con columnas: SKU, Año, Mes, PrecioVentaUSD
   - Modificar build_fact_precios() si cambia la estructura

4. Nuevas métricas:
   - Agregar cálculos en build_detalle() después de la conversión numérica
   - Incluir en num_cols si son columnas numéricas
"""
