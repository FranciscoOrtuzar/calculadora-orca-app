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
    "RECETA_SKU": "Receta de SKU (opcional, para simular frutas).",
    "INFO_FRUTA": "Información de fruta (opcional, para simular frutas)."
}

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

def month_iter(start_y, start_m, end_y, end_m):
    y, m = start_y, start_m
    while (y < end_y) or (y == end_y and m <= end_m):
        yield y, m, y*100 + m
        m += 1
        if m == 13:
            y += 1
            m = 1

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

def carry_forward(df, key_cols, sort_col, value_col):
    df = df.sort_values(key_cols + [sort_col]).copy()
    df[value_col] = df.groupby(key_cols, dropna=False)[value_col].ffill()
    return df

def carry_forward_ignore_zeros(df, key_cols, sort_col, value_col):
    """
    Carry-forward que trata los valores 0.0 como valores faltantes.
    Reemplaza 0.0 con el último valor no-cero y no-NaN anterior.
    """
    df = df.sort_values(key_cols + [sort_col]).copy()
    
    # Convertir 0.0 a NaN para que se propaguen con carry-forward
    df[value_col] = df[value_col].replace(0.0, np.nan)
    
    # Aplicar carry-forward normal - asegurar que se agrupe correctamente
    df[value_col] = df.groupby(key_cols, dropna=False)[value_col].ffill()
    
    # Llenar los NaN restantes con 0.0 (valores que nunca tuvieron datos)
    df[value_col] = df[value_col].fillna(0.0)
    
    return df

# ---------------- Agregados de costos (defensivo) ----------------
DIRECTOS_KEYS = [
    "MO Directa",
    "Materiales Directos Cajas y Bolsas",
    "Laboratorio",
    "Mantencion y Maquinaria",
    "Utilities",
    "Fletes",
    "Comex",
    "Guarda Prodructo Terminado",
]
INDIRECTOS_KEYS = [
    "MO Indirecta",
    "Materiales Indirectos",
    "Servicios Generales",
]
# También se suelen ver así (alias simples):
ALIASES = {
    "Mantención": "Mantencion y Maquinaria",
    "Fletes Internos": "Fletes",
    "Calidad y Laboratorio": "Laboratorio",
    "Guarda PT": "Guarda Prodructo Terminado",
}

def apply_aliases(df_costs: pd.DataFrame) -> pd.DataFrame:
    df = df_costs.copy()
    rename = {k:v for k,v in ALIASES.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df

def sum_existing(df, cols):
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist].sum(axis=1) if cols_exist else 0.0

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
    Valida que existan las hojas requeridas.
    
    Args:
        sheets: Dict de hojas cargadas
        
    Returns:
        Lista de hojas faltantes
    """
    missing = [s for s in REQ_SHEETS if s not in sheets]
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
            rename_map[c] = "Materiales Directos"
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
            rename_map[c] = "Guarda PT"
        elif lc == "guarda mmpp":
            rename_map[c] = "Almacenaje MMPP"
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
    gastos_cols = ["Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)", "Almacenaje MMPP", "Proceso Granel (USD/kg)"]
    if all(col in df.columns for col in gastos_cols):
        df["Gastos Totales (USD/kg)"] = (
            df["Retail Costos Directos (USD/kg)"].astype(float) + 
            df["Retail Costos Indirectos (USD/kg)"].astype(float) + 
            df["Almacenaje MMPP"].astype(float) + 
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

def build_fact_granel_ponderado(df_granel: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa la tabla de costos de granel ponderados por fruta_id.
    
    Args:
        df_granel: DataFrame con los costos de granel por fruta
        
    Returns:
        DataFrame con costos procesados por fruta_id
    """
    df0 = df_granel.copy()
    df0.columns = [_norm_text(c) for c in df0.columns]

    # Detectar columna fruta_id
    col_fruta_id = None
    for c in df0.columns:
        if _norm_text(c).lower() in ["fruta_id", "frutaid", "fruta id"]:
            col_fruta_id = c
            break
    if col_fruta_id is None:
        raise ValueError("No se encontró columna 'fruta_id' en la hoja de granel.")

    # Renombrar columnas conocidas de granel
    rename_map = {}
    for c in df0.columns:
        lc = _norm_text(c).lower()
        if lc in ["mo dir", "mo directa", "mano de obra directa"]:
            rename_map[c] = "MO Directa"
        elif lc in ["mo ind", "mo indirecta", "mano de obra indirecta"]:
            rename_map[c] = "MO Indirecta"
        elif lc in ["mo total", "mano de obra total"]:
            rename_map[c] = "MO Total"
        elif lc in ["mat dir", "materiales directos", "materiales_directos"]:
            rename_map[c] = "Materiales Directos"
        elif lc in ["mat ind", "materiales indirectos", "materiales_indirectos"]:
            rename_map[c] = "Materiales Indirectos"
        elif lc in ["mat total", "materiales total", "materiales_total"]:
            rename_map[c] = "Materiales Total"
        elif lc in ["laboratorio", "calidad"]:
            rename_map[c] = "Laboratorio"
        elif lc in ["sgen", "servicios generales", "servicios_generales"]:
            rename_map[c] = "Servicios Generales"
        elif lc in ["utilities", "utilidades"]:
            rename_map[c] = "Utilities"
        elif lc in ["mantencion", "mantención", "mantenimiento"]:
            rename_map[c] = "Mantencion y Maquinaria"
        elif lc in ["fletes"]:
            rename_map[c] = "Fletes"
        elif lc in ["comex"]:
            rename_map[c] = "Comex"
        elif lc in ["guarda_pt", "guarda producto terminado"]:
            rename_map[c] = "Guarda Producto Terminado"
        elif lc in ["fruta", "nombre_fruta", "nombre fruta"]:
            rename_map[c] = "Fruta"

    df = df0.rename(columns=rename_map).copy()
    df = df.rename(columns={col_fruta_id: "Fruta_id"})
    df["Fruta_id"] = df["Fruta_id"].astype(str).str.strip()

    # Convertir valores a numéricos
    for c in df.columns:
        if c in ["Fruta_id", "Fruta"]:
            continue
        df[c] = df[c].apply(to_number_safe)

    # Quedarse con fruta_id + Fruta + columnas de costos
    cost_cols = [c for c in df.columns if c not in ["Fruta_id", "Fruta"]]
    out = df[["Fruta_id", "Fruta"] + cost_cols].dropna(subset=["Fruta_id"]).reset_index(drop=True)

    return out

def build_fact_costos_mensuales(df_c: pd.DataFrame, start, end, fill_before_first=False) -> pd.DataFrame:
    """
    Normaliza costos en formato largo y aplica carry-forward mensual por (SKU, tipo_costo).

    Espera columnas: ['SKU','Año','Mes','FechaClave','tipo_costo','valor_costo']
    Devuelve:        ['SKU','Año','Mes','FechaClave','tipo_costo','valor_costo'] ordenado.
    """
    need = {"SKU","Año","Mes","tipo_costo","valor_costo"}
    if not need.issubset(set(df_c.columns)):
        raise ValueError(f"FACT_COSTOS_MENSUALES debe tener columnas {need}.")
    c = df_c.copy()
    c.columns = [str(x).strip() for x in c.columns]
    c["SKU"] = c["SKU"].astype(str).str.strip()
    c["Año"] = c["Año"].apply(lambda x: int(str(x).strip()))
    c["Mes"] = c["Mes"].apply(lambda x: int(str(x).strip()))
    c["FechaClave"] = c["Año"]*100 + c["Mes"]
    c["tipo_costo"] = c["tipo_costo"].astype(str).str.strip()
    c["valor_costo"] = c["valor_costo"].apply(to_number_safe)

    # Consolidar duplicados
    c = (c.groupby(["SKU","tipo_costo","FechaClave","Año","Mes"], as_index=False)["valor_costo"]
           .sum(min_count=1))

    # Calendario por (SKU, tipo_costo, mes)
    skus  = c[["SKU"]].drop_duplicates()
    tipos = c[["tipo_costo"]].drop_duplicates()

    cal = []
    for y, m, fk in month_iter(start[0], start[1], end[0], end[1]):
        tmp = skus.merge(tipos, how="cross")        # ← cartesiano correcto
        tmp["Año"] = y; tmp["Mes"] = m; tmp["FechaClave"] = fk
        cal.append(tmp)
    base = pd.concat(cal, ignore_index=True)

    # Traer valores y CF por SKU+tipo_costo
    base = (base.merge(
                c[["SKU","tipo_costo","FechaClave","valor_costo"]],
                on=["SKU","tipo_costo","FechaClave"], how="left")
              .sort_values(["SKU","tipo_costo","FechaClave"]))
    
    base = carry_forward_ignore_zeros(base, ["SKU","tipo_costo"], "FechaClave", "valor_costo")
    
    # Llenar valores faltantes después del carry-forward
    if fill_before_first:
        # Carry backwards: tomar el primer valor disponible y propagarlo hacia atrás
        base["valor_costo"] = base.groupby(["SKU","tipo_costo"], dropna=False)["valor_costo"].bfill().ffill()
    else:
        # Para valores que no tienen carry-forward, usar 0.0 en lugar de NaN
        base["valor_costo"] = base["valor_costo"].fillna(0.0)

    # Pivot a ancho
    wide = (base.pivot_table(index=["SKU","Año","Mes","FechaClave"],
                             columns="tipo_costo",
                             values="valor_costo",
                             aggfunc="last")
                 .reset_index())
    wide.columns.name = None

    # Asegurar numéricos y signo costo negativo
    for ccol in wide.columns:
        if ccol in {"SKU","Año","Mes","FechaClave"}:
            continue
        wide[ccol] = pd.to_numeric(wide[ccol], errors="coerce")
        # Solo aplicar signo negativo si el valor no es 0
        wide[ccol] = np.where(wide[ccol] != 0, -abs(wide[ccol]), 0.0)

    return wide

def build_fact_granel(df_g: pd.DataFrame, start=(2024, 10), end=(2025, 6), fill_before_first=False) -> pd.DataFrame:
    """
    Normaliza costos de granel en formato largo y aplica carry-forward mensual por (SKU, tipo_costo).

    Espera columnas: ['SKU','Año','Mes','FechaClave','tipo_costo','valor_costo']
    Devuelve:        ['SKU','Año','Mes','FechaClave','tipo_costo','valor_costo'] ordenado.
    """
    need = {"SKU","Año","Mes","tipo_costo","valor_costo"}
    if not need.issubset(set(df_g.columns)):
        raise ValueError(f"FACT_GRANEL debe tener columnas {need}.")
    g = df_g.copy()
    g.columns = [str(x).strip() for x in g.columns]
    g["SKU"] = g["SKU"].astype(str).str.strip()
    g["Año"] = g["Año"].apply(lambda x: int(str(x).strip()))
    g["Mes"] = g["Mes"].apply(lambda x: int(str(x).strip()))
    g["FechaClave"] = g["Año"]*100 + g["Mes"]
    g["tipo_costo"] = g["tipo_costo"].astype(str).str.strip()
    g["valor_costo"] = g["valor_costo"].apply(to_number_safe)

    # Consolidar duplicados
    g = (g.groupby(["SKU","tipo_costo","FechaClave","Año","Mes"], as_index=False)["valor_costo"]
           .sum(min_count=1))

    # Calendario por (SKU, tipo_costo, mes)
    skus  = g[["SKU"]].drop_duplicates()
    tipos = g[["tipo_costo"]].drop_duplicates()

    cal = []
    for y, m, fk in month_iter(start[0], start[1], end[0], end[1]):
        tmp = skus.merge(tipos, how="cross")        # ← cartesiano correcto
        tmp["Año"] = y; tmp["Mes"] = m; tmp["FechaClave"] = fk
        cal.append(tmp)
    base = pd.concat(cal, ignore_index=True)

    # Traer valores y CF por SKU+tipo_costo
    base = (base.merge(
                g[["SKU","tipo_costo","FechaClave","valor_costo"]],
                on=["SKU","tipo_costo","FechaClave"], how="left")
              .sort_values(["SKU","tipo_costo","FechaClave"]))
    base = carry_forward_ignore_zeros(base, ["SKU","tipo_costo"], "FechaClave", "valor_costo")
    
    # Llenar valores faltantes después del carry-forward
    if fill_before_first:
        # Carry backwards: tomar el primer valor disponible y propagarlo hacia atrás
        base["valor_costo"] = base.groupby(["SKU","tipo_costo"], dropna=False)["valor_costo"].bfill().ffill()
    else:
        # Para valores que no tienen carry-forward, usar 0.0 en lugar de NaN
        base["valor_costo"] = base["valor_costo"].fillna(0.0)

    # Pivot a ancho
    wide = (base.pivot_table(index=["SKU","Año","Mes","FechaClave"],
                             columns="tipo_costo",
                             values="valor_costo",
                             aggfunc="last")
                 .reset_index())
    wide.columns.name = None

    # Asegurar numéricos y signo costo negativo
    for ccol in wide.columns:
        if ccol in {"SKU","Año","Mes","FechaClave"}:
            continue
        wide[ccol] = pd.to_numeric(wide[ccol], errors="coerce")
        # Solo aplicar signo negativo si el valor no es 0
        wide[ccol] = np.where(wide[ccol] != 0, -abs(wide[ccol]), 0.0)

    return wide


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

def build_fact_precios_cf(
    df_p: pd.DataFrame,
    start=(2024, 10),   # Oct-2024
    end=(2025, 6)       # Jun-2025
) -> pd.DataFrame:
    """
    Devuelve precios mensuales por SKU-Cliente con carry-forward en el rango [start, end].
    Requiere columnas: ['SKU-Cliente','SKU','Año','Mes','PrecioVentaUSD'] (o 'PrecioVenta').
    Salida: ['SKU-Cliente','SKU','Año','Mes','FechaClave','PrecioVentaUSD']
    """
    # Normaliza nombres
    p = df_p.copy()
    p.columns = [str(c).strip() for c in p.columns]
    if "PrecioVentaUSD" not in p.columns and "PrecioVenta" in p.columns:
        p = p.rename(columns={"PrecioVenta": "PrecioVentaUSD"})

    needed = {"SKU-Cliente","SKU","Año","Mes","PrecioVentaUSD"}
    missing = needed - set(p.columns)
    if missing:
        raise ValueError(f"Faltan columnas en FACT_PRECIOS: {sorted(missing)}")

    # Normaliza tipos
    p["SKU-Cliente"] = p["SKU-Cliente"].astype(str).str.strip()
    p["SKU"] = p["SKU"].astype(str).str.strip()
    p["Año"] = p["Año"].apply(lambda x: int(str(x).strip()))
    p["Mes"] = p["Mes"].apply(month_to_num).astype(int)
    p["FechaClave"] = p["Año"]*100 + p["Mes"]

    p["PrecioVentaUSD"] = p["PrecioVentaUSD"].apply(to_number_safe)
    p = p.dropna(subset=["PrecioVentaUSD"])

    # Si hubiese duplicados por (SKU-Cliente, FechaClave), nos quedamos con el último
    p = (p.sort_values(["SKU-Cliente","FechaClave"])
           .drop_duplicates(["SKU-Cliente","FechaClave"], keep="last"))

    # Universo de SKUs-Cliente presentes en precios
    skus = p[["SKU-Cliente","SKU"]].drop_duplicates()

    # Calendario mensual completo del periodo
    cal = []
    for y, m, fk in month_iter(start[0], start[1], end[0], end[1]):
        tmp = skus.copy()
        tmp["Año"] = y
        tmp["Mes"] = m
        tmp["FechaClave"] = fk
        cal.append(tmp)
    skeleton = pd.concat(cal, ignore_index=True)

    # Mezcla precios observados al calendario
    df = skeleton.merge(
        p[["SKU-Cliente","FechaClave","PrecioVentaUSD"]],
        on=["SKU-Cliente","FechaClave"],
        how="left"
    )

    # Carry-forward por SKU-Cliente
    df = carry_forward(df, key_cols=["SKU-Cliente"], sort_col="FechaClave", value_col="PrecioVentaUSD")

    # Orden final
    df = df.sort_values(["SKU-Cliente","FechaClave"]).reset_index(drop=True)
    return df[["SKU-Cliente","SKU","Año","Mes","FechaClave","PrecioVentaUSD"]]

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

def build_fact_volumen(df_vol: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa la tabla de volúmenes y retorna en formato limpio con FechaClave.

    Espera columnas: ['SKU','SKU-Cliente','Año','Mes','KgEmbarcados']
    """
    needed = {"SKU","SKU-Cliente","Año","Mes","KgEmbarcados"}
    if not needed.issubset(set(df_vol.columns)):
        raise ValueError(f"FACT_VOL debe contener {needed}. Columnas: {df_vol.columns.tolist()}")

    v = df_vol.copy()
    v.columns = [c.strip() for c in v.columns]

    v["SKU"] = v["SKU"].astype(str).str.strip()
    v["SKU-Cliente"] = v["SKU-Cliente"].astype(str).str.strip()
    v["Año"] = v["Año"].apply(lambda x: int(str(x).strip()))
    v["Mes"] = v["Mes"].apply(lambda x: int(str(x).strip()))
    v["KgEmbarcados"] = v["KgEmbarcados"].apply(to_number_safe)

    # Calcula FechaClave (YYYYMM)
    v["FechaClave"] = v["Año"]*100 + v["Mes"]

    # Ordenar para consistencia
    v = v.sort_values(["SKU-Cliente","FechaClave"]).reset_index(drop=True)

    return v[["SKU","SKU-Cliente","Año","Mes","FechaClave","KgEmbarcados"]]

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

def columns_config(editable: bool = True) -> dict:
    # Configurar columnas editables (solo costos individuales, no totales)
    
    # Definir tipos de columnas para la configuración
    cost_columns = ["Proceso Granel (USD/kg)", "Almacenaje MMPP", "MMPP (Fruta) (USD/kg)", "MO Directa", "MO Indirecta",
    "Materiales Directos", "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", "Utilities",
    "Fletes Internos", "Comex", "Guarda PT"]
    dimension_cols_edit = ["SKU", "SKU-Cliente", "Descripcion", "Marca", "Cliente", "Especie", "Condicion"]  # Columnas dimensionales visibles
    total_cols_edit = ["Costos Totales (USD/kg)", "Gastos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]
    intermediate_cols_edit = ["PrecioVenta (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
                                "MO Total", "Materiales Total", "MMPP Total (USD/kg)"]
    
    editable_columns = {}   
    for col in cost_columns:
        if col == "Materiales Directos":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: Cajas y Bolsas",
                format="%.3f",
                step=0.001
            )
        elif col == "Almacenaje MMPP":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Es el costo de almacenaje de la fruta granel",
                format="%.3f",
                step=0.001
            )
        else:
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Valor de {col} (los costos se muestran como negativos)",
                format="%.3f",
                step=0.001
            )
    
    # Configurar columnas dimensionales (visibles pero no editables)
    for col in dimension_cols_edit:
        if col == "SKU":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                disabled=True,
                format="%.0f",
                help=f"{col} (no editable)",
                pinned="left"
            )
        elif col == "Descripcion":
            editable_columns[col] = st.column_config.TextColumn(
                col,
                disabled=True,
                help=f"{col} (no editable)",
                width="medium",
                pinned="left"
            )
        else:
            editable_columns[col] = st.column_config.TextColumn(
                col,
                disabled=True,
                help=f"{col} (no editable)",
            )
    # Configurar columnas intermedias (no editables)
    for col in intermediate_cols_edit:
        if col == "PrecioVenta (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Último precio de venta registrado",
                format="%.3f",
                step=0.001,
                min_value=0.0,
                max_value=10.0,
                pinned="left"
            )
        elif col == "Retail Costos Directos (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: MO Directa, Materiales Cajas y Bolsas, Laboratorio, Mantención, Utilities, Fletes Internos, Comex y Guarda PT",
                format="%.3f",
                step=0.001,
                disabled=True
            )
        elif col == "Retail Costos Indirectos (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: MO Indirecta, Materiales Indirectos y Servicios Generales",
                format="%.3f",
                step=0.001,
                disabled=True
            )
        elif col == "MO Total":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: MO Directa y MO Indirecta",
                format="%.3f",
                step=0.001,
                disabled=True
            )
        elif col == "Materiales Total":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: Materiales Cajas y Bolsas y Materiales Indirectos",
                format="%.3f",
                step=0.001,
                disabled=True
            )
        elif col == "MMPP Total (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: MMPP (Fruta) (USD/kg) y Proceso Granel (USD/kg)",
                format="%.3f",
                step=0.001,
                disabled=True
            )
    
    for col in total_cols_edit:
        if col == "EBITDA Pct":
            # Formato especial para EBITDA Pct como porcentaje
            editable_columns["EBITDA Pct"] = st.column_config.NumberColumn(
                "EBITDA Pct",
                help="Margen EBITDA en porcentaje (no editable)",
                format="%.1f%%",
                min_value=-100.0,
                step=0.1,
                disabled=True,
                pinned="left"
            )
        elif col == "EBITDA (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
            col,
            help=f"Margen EBITDA: PrecioVenta (USD/kg) - Costos Totales (USD/kg)",
            format="%.3f",
            step=0.001,
            disabled=True,
            pinned="left"
            )
        elif col == "Gastos Totales (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: Retail Costos Directos (USD/kg), Retail Costos Indirectos (USD/kg), Almacenaje MMPP y Proceso Granel (USD/kg)",
                format="%.3f",
                step=0.001,
                disabled=True
            )
        elif col == "Costos Totales (USD/kg)":
            editable_columns[col] = st.column_config.NumberColumn(
                col,
                help=f"Incluye: Gastos Totales (USD/kg) y MMPP Fruta (USD/kg)",
                format="%.3f",
                step=0.001,
                disabled=True
            )
    editable_columns["KgEmbarcados"] = st.column_config.NumberColumn(
        "KgEmbarcados",
        help="Kilogramos embarcados",
        format="%.0f",
        step=1,
        disabled=True
    )
    editable_columns["EBITDA (USD)"] = st.column_config.NumberColumn(
        "EBITDA (USD)",
        help="EBITDA en dólares",
        format="%.0f",
        step=1,
        disabled=True
    )
    return editable_columns

    # ===================== Función para Recalcular Totales =====================
def recalculate_totals(detalle: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula costos totales, gastos totales y EBITDA basándose en costos individuales.
    
    Args:
        df: DataFrame con costos individuales
        
    Returns:
        DataFrame con totales recalculados
    """
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
        "Materiales Directos",
        "Materiales Indirectos"
    ]
    
    if all(col in detalle.columns for col in materiales_components):
        detalle["Materiales Total"] = detalle[materiales_components].sum(axis=1)

    # 4.1 Recalcular Retail Costos Directos (USD/kg) si están los componentes
    retail_costs_direct_components = [
        "MO Directa",
        "Materiales Directos",
        "Laboratorio",
        "Mantención",
        "Utilities",
        "Fletes Internos",
    ]
    
    if all(col in detalle.columns for col in retail_costs_direct_components):
        detalle["Retail Costos Directos (USD/kg)"] = detalle[retail_costs_direct_components].sum(axis=1)
    
    # 4.2 Recalcular Retail Costos Indirectos (USD/kg) si están los componentes
    retail_costs_indirect_components = [
        "MO Indirecta",
        "Materiales Indirectos"
    ]
    if all(col in detalle.columns for col in retail_costs_indirect_components):
        detalle["Retail Costos Indirectos (USD/kg)"] = detalle[retail_costs_indirect_components].sum(axis=1)
    
    # 4. Recalcular Gastos Totales (costos indirectos - NO incluye MMPP)
    gastos_components = [
        "Almacenaje MMPP",
        "Proceso Granel (USD/kg)",
        "Retail Costos Indirectos (USD/kg)",
        "Retail Costos Directos (USD/kg)",
        "Servicios Generales",
        "COMEX",
        "Guarda PT"
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
        (detalle["EBITDA (USD/kg)"] / detalle["PrecioVenta (USD/kg)"]) * 100,
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
# ===================== Construcción del mart =====================
@st.cache_data(show_spinner=True)
def build_detalle(uploaded_bytes: bytes, ultimo_precio_modo: str, ref_ym: Optional[int], optimo: bool = False) -> pd.DataFrame:
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
    if optimo:
        costos_detalle = build_tbl_costos_pond(sheets["FACT_COSTOS_OPT"])
    else:
        costos_detalle = build_tbl_costos_pond(sheets["FACT_COSTOS_POND"])

    # 2) Precios + último precio por SKU
    precios = build_fact_precios(sheets["FACT_PRECIOS"])
    if ultimo_precio_modo == "global":
        latest = compute_latest_price(precios, mode="global")
    else:
        latest = compute_latest_price(precios, mode="to_date", ref_datekey=ref_ym)

    # 3) DIM_SKU
    dim = build_dim_sku(sheets["DIM_SKU"])

    # 3.1) MMPP y Almacenaje por SKU
    mmpp_almacenaje = compute_mmpp_y_almacenaje_per_sku(sheets["RECETA_SKU"], sheets["INFO_FRUTA"])
    mmpp_almacenaje = mmpp_almacenaje.rename(columns={"MMPP (Fruta) (USD/kg)": "MMPP (Fruta) (USD/kg) (Calculado)", "Almacenaje": "Almacenaje (Calculado)"})    
    # 4) Unión de tablas
    costos_detalle_calculado = costos_detalle.merge(mmpp_almacenaje, on="SKU", how="right")
    costos_detalle_calculado["MMPP (Fruta) (USD/kg)"] = -abs(costos_detalle_calculado["MMPP (Fruta) (USD/kg) (Calculado)"].fillna(costos_detalle_calculado["MMPP (Fruta) (USD/kg)"]))
    costos_detalle_calculado["Almacenaje MMPP"] = -abs(costos_detalle_calculado["Almacenaje (Calculado)"].fillna(costos_detalle_calculado["Almacenaje MMPP"]))
    costos_detalle_calculado = costos_detalle_calculado.drop(columns=["MMPP (Fruta) (USD/kg) (Calculado)", "Almacenaje (Calculado)"])
    detalle = costos_detalle_calculado.merge(dim, on="SKU", how="right")
    # Si ambos tienen SKU-Cliente, unir por esa columna; si no, unir por SKU
    if "SKU-Cliente" in dim.columns:
        detalle = detalle.merge(latest, on="SKU-Cliente", how="right")
    else:
        detalle = detalle.merge(latest, on="SKU", how="right")
    detalle = detalle.drop(columns=["FechaClave"])

    # 5) Conversión a numérico y cálculo de métricas
    numeric_columns = detalle.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        detalle[col] = pd.to_numeric(detalle[col], errors='coerce').fillna(0.0)
    
    # FORZAR SIGNOS CORRECTOS
    # Los costos siempre deben ser negativos

    detalle = recalculate_totals(detalle)
    return detalle

@st.cache_data(show_spinner=True)
def build_ebitda_mensual(uploaded_bytes: bytes,
                         sheet_vol="FACT_VOL",
                         sheet_precios="FACT_PRECIOS",
                         sheet_costos="FACT_COSTOS_MENSUALES",
                         fill_costs_before_first=True):
    """
    Lee un workbook en bytes y devuelve un DF mensual por SKU-Cliente:
    Columnas: SKU, SKU-Cliente, Año, Mes, FechaClave, KgEmbarcados, PrecioVentaUSD,
              [tipo_costo...], Retail Costos Directos (USD/kg), Retail Costos Indirectos (USD/kg),
              Gastos Totales (USD/kg), Costos Totales (USD/kg),
              EBITDA (USD/kg), EBITDA Pct, Ingresos (USD), Costo Total (USD), EBITDA (USD)
    """
    sheets = read_workbook(uploaded_bytes)

    if sheet_vol not in sheets: raise ValueError(f"No está la hoja '{sheet_vol}'.")
    if sheet_precios not in sheets: raise ValueError(f"No está la hoja '{sheet_precios}'.")
    if sheet_costos not in sheets: raise ValueError(f"No está la hoja '{sheet_costos}'.")

    vol = build_fact_volumen(sheets[sheet_vol])

    # Determinar período (preferimos el rango de volúmenes)
    if len(vol):
        fc_min, fc_max = vol["FechaClave"].min(), vol["FechaClave"].max()
        start = (int(fc_min//100), int(fc_min%100))
        end   = (int(fc_max//100), int(fc_max%100))
    else:
        # Si no hay volúmenes, usa rangos de precios/costos
        p_tmp = sheets[sheet_precios].copy()
        p_tmp.columns = [c.strip() for c in p_tmp.columns]
        p_tmp["Año"] = p_tmp["Año"].apply(lambda x: int(str(x).strip()))
        p_tmp["Mes"] = p_tmp["Mes"].apply(month_to_num).astype(int)
        p_tmp["FechaClave"] = p_tmp["Año"]*100 + p_tmp["Mes"]
        c_tmp = sheets[sheet_costos].copy()
        c_tmp.columns = [c.strip() for c in c_tmp.columns]
        c_tmp["Año"] = c_tmp["Año"].apply(lambda x: int(str(x).strip()))
        c_tmp["Mes"] = c_tmp["Mes"].apply(lambda x: int(str(x).strip()))
        c_tmp["FechaClave"] = c_tmp["Año"]*100 + c_tmp["Mes"]
        fc_min = min(p_tmp["FechaClave"].min(), c_tmp["FechaClave"].min())
        fc_max = max(p_tmp["FechaClave"].max(), c_tmp["FechaClave"].max())
        start = (int(fc_min//100), int(fc_min%100))
        end   = (int(fc_max//100), int(fc_max%100))

    precios = build_fact_precios_cf(sheets[sheet_precios], start, end)
    costos_wide = build_fact_costos_mensuales(sheets[sheet_costos], start, end,
                                                  fill_before_first=fill_costs_before_first)
    costos_wide = apply_aliases(costos_wide)

    # ← columnas de costo (del pivot ya ancho)
    cost_cols_from_wide = [c for c in costos_wide.columns if c not in {"SKU","Año","Mes","FechaClave"}]
    costos_wide = costos_wide[["SKU", "FechaClave"] + cost_cols_from_wide]

    # Merge: vol + precios por SKU-Cliente + costos por SKU
    df = vol.merge(precios[["SKU-Cliente","FechaClave","PrecioVentaUSD","SKU"]],
                   on=["SKU-Cliente","FechaClave"], how="left", suffixes=("","_p"))
    # Asegura SKU desde precios si falta
    df["SKU"] = df["SKU"].fillna(df["SKU_p"]).fillna(df["SKU"].astype(str))
    df = df.drop(columns=[c for c in ["SKU_p"] if c in df.columns])

    df = df.merge(costos_wide, on=["SKU","FechaClave"], how="left")

    # Asegurar NUMÉRICO SOLO en columnas de costo
    for ccol in cost_cols_from_wide:
        if ccol in df.columns:
            df[ccol] = pd.to_numeric(df[ccol], errors="coerce")
            # Solo aplicar signo negativo si el valor no es 0
            df[ccol] = np.where(df[ccol] != 0, -abs(df[ccol]), 0.0)

    # 2) carry-forward por SKU en el tiempo - NO aplicar carry-forward aquí
    # Los datos ya vienen con carry-forward aplicado desde build_fact_costos_mensuales

    # 3) (opcional) si quieres rellenar lo anterior al primer valor con 0
    if fill_costs_before_first:
        df[cost_cols_from_wide] = df[cost_cols_from_wide].fillna(0.0)

    # ---- Agregados de costos (defensivo, solo con columnas presentes) ----
    df["Retail Costos Directos (USD/kg)"] = sum_existing(df, DIRECTOS_KEYS)
    df["Retail Costos Indirectos (USD/kg)"] = sum_existing(df, INDIRECTOS_KEYS)

    gastos_cols = ["Retail Costos Directos (USD/kg)",
                   "Retail Costos Indirectos (USD/kg)"]
    # Si existen, añade "Proceso Granel (USD/kg)" y "Almacenaje MMPP"
    if "Proceso Granel (USD/kg)" in df.columns: gastos_cols.append("Proceso Granel (USD/kg)")
    if "Almacenaje MMPP" in df.columns: gastos_cols.append("Almacenaje MMPP")

    df["Gastos Totales (USD/kg)"] = df[gastos_cols].sum(axis=1) if gastos_cols else 0.0

    # Costos Totales = Gastos + (MMPP Fruta si existe)
    comp_total = ["Gastos Totales (USD/kg)"]
    if "Costo MMPP con Granel" in df.columns:
        comp_total.append("Costo MMPP con Granel")
    df["Costos Totales (USD/kg)"] = df[comp_total].sum(axis=1)

    # ---- Resultados ----
    # Precio siempre positivo (si viene negativo lo corregimos)
    df["PrecioVentaUSD"] = df["PrecioVentaUSD"].abs()

    df["EBITDA (USD/kg)"] = df["PrecioVentaUSD"] - df["Costos Totales (USD/kg)"].abs()
    df["EBITDA Pct"] = np.where(df["PrecioVentaUSD"].abs() > 1e-12,
                                df["EBITDA (USD/kg)"] / df["PrecioVentaUSD"],
                                np.nan)

    # Magnitudes totales (ponderadas por KgEmbarcados)
    df["Ingresos (USD)"] = df["PrecioVentaUSD"] * df["KgEmbarcados"]
    df["Costo Total (USD)"] = df["Costos Totales (USD/kg)"].abs() * df["KgEmbarcados"]
    df["EBITDA (USD)"] = df["EBITDA (USD/kg)"] * df["KgEmbarcados"]

    # Orden final
    df = df.sort_values(["SKU-Cliente","FechaClave"]).reset_index(drop=True)

    # Reordenar columnas: dims, métricas base, costos, agregados y KPIs
    base_cols = ["SKU","SKU-Cliente","Año","Mes","FechaClave","KgEmbarcados","PrecioVentaUSD"]
    agg_cols = ["Retail Costos Directos (USD/kg)",
                "Retail Costos Indirectos (USD/kg)",
                "Gastos Totales (USD/kg)",
                "Costos Totales (USD/kg)",
                "EBITDA (USD/kg)","EBITDA Pct",
                "Ingresos (USD)","Costo Total (USD)","EBITDA (USD)"]
    other_costs = [c for c in df.columns if c not in base_cols + agg_cols]
    # mantén base -> costos individuales -> agregados
    ordered = base_cols + other_costs + agg_cols
    df = df[[c for c in ordered if c in df.columns]]

    return df, costos_wide, vol, precios

@st.cache_data(show_spinner=True)
def build_granel(uploaded_bytes: bytes, sheet_granel=["FACT_GRANEL", "FACT_GRANEL_POND"]) -> tuple:
    """
    Lee un workbook en bytes y devuelve un DF mensual por SKU-Cliente y un DF de granel ponderado:
    """
    sheets = read_workbook(uploaded_bytes)
    
    # Verificar que las hojas existan
    if sheet_granel[0] not in sheets: 
        raise ValueError(f"No está la hoja '{sheet_granel[0]}'.")
    if sheet_granel[1] not in sheets: 
        raise ValueError(f"No está la hoja '{sheet_granel[1]}'.")
    
    granel = build_fact_granel(sheets[sheet_granel[0]], fill_before_first=True)
    granel_ponderado = build_fact_granel_ponderado(sheets[sheet_granel[1]])
    return granel, granel_ponderado

# ===================== Carga de datos de fruta =====================
def load_receta_sku(df_excel: pd.DataFrame) -> pd.DataFrame:
    """
    Carga y normaliza la hoja RECETA_SKU del Excel.
    
    Args:
        df_excel: DataFrame de la hoja RECETA_SKU
        
    Returns:
        DataFrame normalizado con columnas [SKU, Fruta_id, Porcentaje, Óptimo]
    """
    # Verificar columnas requeridas
    required_cols = ["SKU", "Fruta_id", "Porcentaje", "Óptimo"]
    missing_cols = [col for col in required_cols if col not in df_excel.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en RECETA_SKU: {missing_cols}")
    
    # Crear copia y normalizar
    df = df_excel[required_cols].copy()
    
    # Convertir SKU a string y limpiar
    df["SKU"] = df["SKU"].astype(str).str.strip()
    
    # Convertir Fruta_id a string y limpiar
    df["Fruta_id"] = df["Fruta_id"].astype(str).str.strip()
    
    # Convertir Porcentaje a float y validar
    df["Porcentaje"] = df["Porcentaje"].apply(lambda x: to_number_safe(x, comma_decimal=True))

    # Convertir Óptimo a float y validar
    df["Óptimo"] = df["Óptimo"].apply(lambda x: to_number_safe(x, comma_decimal=True))

    # Filtrar filas válidas (con Porcentaje no NaN y ≥ 0)
    df = df[df["Porcentaje"].notna() & (df["Porcentaje"] >= 0)]
    
    # Resetear índice
    df = df.reset_index(drop=True)
    
    return df


def load_info_fruta(df_excel: pd.DataFrame) -> pd.DataFrame:
    """
    Carga y normaliza la hoja INFO_FRUTA del Excel.
    
    Args:
        df_excel: DataFrame de la hoja INFO_FRUTA
        
    Returns:
        DataFrame normalizado con columnas [Fruta_id, Precio, Rendimiento, Name, Almacenaje]
    """
    # Verificar columnas requeridas
    required_cols = ["Fruta_id", "Precio", "Rendimiento", "Name", "Almacenaje"]
    missing_cols = [col for col in required_cols if col not in df_excel.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en INFO_FRUTA: {missing_cols}")
    
    # Crear copia y normalizar
    df = df_excel[required_cols].copy()
    
    # Convertir Fruta_id a string y limpiar
    df["Fruta_id"] = df["Fruta_id"].astype(str).str.strip()
    
    # Convertir Precio a float y validar
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    
    # Convertir Rendimiento a float y validar
    df["Rendimiento"] = pd.to_numeric(df["Rendimiento"], errors="coerce")

    # Convertir Almacenaje a float y validar
    df["Almacenaje"] = pd.to_numeric(df["Almacenaje"], errors="coerce") 

    # Convertir Name a string y limpiar
    df["Name"] = df["Name"].astype(str).str.strip()
    
    # Filtrar filas válidas
    df = df[df["Precio"].notna() & df["Rendimiento"].notna()]
    
    # Validar rangos
    df = df[df["Precio"] >= 0]
    df = df[df["Rendimiento"] > 0]
    df = df[df["Rendimiento"] <= 1]
    df = df[df["Almacenaje"] <= 0]
    
    # Reemplazar rendimientos 0/NaN por mínimo 0.01 y loggear warning
    zero_efficiency_mask = (df["Rendimiento"] <= 0) | df["Rendimiento"].isna()
    if zero_efficiency_mask.any():
        zero_count = zero_efficiency_mask.sum()
        st.warning(f"⚠️ {zero_count} frutas con rendimiento ≤ 0 o NaN. Se estableció rendimiento mínimo de 0.01")
        df.loc[zero_efficiency_mask, "Rendimiento"] = 0.01

    df["Costo efectivo"] = df["Precio"] / df["Rendimiento"]
    
    # Resetear índice
    df = df.reset_index(drop=True)
    
    return df

# ===================== Cálculo de MMPP =====================
def compute_mmpp_y_almacenaje_per_sku(receta_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula MMPP (Fruta) y Almacenaje por SKU basado en recetas y precios de fruta,
    utilizando operaciones vectorizadas para optimizar el rendimiento.
    """
    # 1. Validación de columnas requeridas
    #---------------------------------------
    required_receta_cols = ["SKU", "Fruta_id", "Porcentaje"]
    missing_receta_cols = [col for col in required_receta_cols if col not in receta_df.columns]
    if missing_receta_cols:
        raise ValueError(f"Columnas faltantes en RECETA_SKU: {missing_receta_cols}")

    required_info_cols = ["Fruta_id", "Precio", "Rendimiento", "Name", "Almacenaje"]
    missing_info_cols = [col for col in required_info_cols if col not in info_df.columns]
    if missing_info_cols:
        raise ValueError(f"Columnas faltantes en INFO_FRUTA: {missing_info_cols}")

    # 2. Copia y normalización de datos
    #-----------------------------------
    # Crear copias para evitar modificar los DataFrames originales
    df_receta = receta_df[required_receta_cols].copy()
    df_info = info_df[required_info_cols].copy()

    # Normalizar 'Fruta_id' a string y limpiar espacios
    df_receta["Fruta_id"] = df_receta["Fruta_id"].astype(str).str.strip()
    df_info["Fruta_id"] = df_info["Fruta_id"].astype(str).str.strip()

    # Convertir columnas a tipo numérico, forzando NaN en errores
    df_receta["Porcentaje"] = df_receta["Porcentaje"].apply(lambda x: to_number_safe(x, comma_decimal=True))
    df_info["Precio"] = df_info["Precio"].apply(lambda x: to_number_safe(x, comma_decimal=True))
    df_info["Rendimiento"] = df_info["Rendimiento"].apply(lambda x: to_number_safe(x, comma_decimal=True))
    df_info["Almacenaje"] = df_info["Almacenaje"].apply(lambda x: to_number_safe(x, comma_decimal=True))

    # 3. Filtrado y validación de rangos
    #-----------------------------------
    # Eliminar filas con valores nulos o no válidos
    df_receta.dropna(subset=["Porcentaje"], inplace=True)
    df_info.dropna(subset=["Precio", "Rendimiento", "Almacenaje"], inplace=True)
    
    # Filtrar valores numéricos inválidos (ej. <= 0 o > 1 en Rendimiento)
    df_receta = df_receta[df_receta["Porcentaje"] > 0]
    df_info = df_info[
        (df_info["Precio"] >= 0) &
        (df_info["Rendimiento"] > 0) &
        (df_info["Rendimiento"] <= 1) &
        (df_info["Almacenaje"] <= 0)
    ]

    # 4. Cálculo optimizado
    #------------------------
    # Unir los DataFrames para combinar la información de receta y fruta
    # Únicos en cada dataset
    df_merged = pd.merge(df_receta, df_info, on="Fruta_id", how="inner")
    
    # Calcular los costos de forma vectorizada
    # MMPP: Precio de la fruta * Porcentaje de la receta / Rendimiento
    df_merged["Costo_MMPP"] = (df_merged["Precio"] * df_merged["Porcentaje"] / 100) / df_merged["Rendimiento"]
    # Almacenaje: Costo de almacenaje de la fruta * Porcentaje de la receta
    df_merged["Costo_Almacenaje"] = df_merged["Almacenaje"] * df_merged["Porcentaje"] / 100

    # Agrupar por SKU y sumar los costos para obtener el resultado final
    df_result = df_merged.groupby("SKU").agg(
        **{
            "MMPP (Fruta) (USD/kg)": pd.NamedAgg(column='Costo_MMPP', aggfunc='sum'),
            "Almacenaje": pd.NamedAgg(column='Costo_Almacenaje', aggfunc='sum')
        }
    ).reset_index()

    return df_result
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

5. DATOS DE FRUTA:
   - RECETA_SKU: Hoja con columnas [SKU, Fruta_id, Porcentaje] (formato largo)
   - INFO_FRUTA: Hoja con columnas [Fruta_id, PrecioUSD_kg, Eficiencia]
   - Usar load_receta_sku() y load_info_fruta() para cargar
"""
