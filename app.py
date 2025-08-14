# app.py
import io
import json
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

# ===================== Config básica =====================
ST_TITLE = "Calculadora de Márgenes (MVP)"
REQ_SHEETS = {
    "FACT_COSTOS_POND": "Tabla con costos unitarios ponderados por SKU (Oct-Jun).",
    "FACT_PRECIOS": "Precios mensuales: SKU, Año, Mes, PrecioVentaUSD.",
    "DIM_SKU": "Dimensión de SKU (opcional, para filtrar por Marca/Especie/Cliente)."
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
    return MES2NUM.get(str(m).strip().title(), np.nan)

def ensure_str(df, col):
    df[col] = df[col].astype(str).str.strip()
    return df

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
def read_workbook(uploaded_bytes: bytes):
    """Lee el Excel completo en dict de DataFrames (todas las hojas)"""
    bio = io.BytesIO(uploaded_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheets = {name: xls.parse(name, dtype=str) for name in xls.sheet_names}
    return sheets

def validate_required_sheets(sheets: dict):
    missing = [s for s in REQ_SHEETS if s not in sheets]
    return missing

# ===================== Procesamiento =====================
def build_tbl_costos_pond(df_costos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - resumen_df: columnas ['SKU','DirectoUSDkg','IndirectoUSDkg','TotalUSDkg']
      - detalle_df: 'SKU' + TODAS las columnas de costos originales (numéricas) para mostrar en 'Expandir'
    """
    df = df_costos.copy()
    df.columns = [c.strip() for c in df.columns]
    if "SKU" not in df.columns:
        raise ValueError("En 'FACT_COSTOS_POND' no se encontró columna 'SKU'.")
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Guardamos un duplicado para DETALLE y convertimos a numéricos (excepto SKU)
    det = df.copy()
    for c in det.columns:
        if c == "SKU": 
            continue
        det[c] = det[c].apply(to_number_safe)

    # Detecta totales si existen
    col_total_dir = next((c for c in det.columns if "costo directo" in c.lower()), None)
    col_total_ind = next((c for c in det.columns if "costo indirecto" in c.lower()), None)
    col_total_all = next((c for c in det.columns if "costo total" in c.lower()), None)

    resumen = det[["SKU"]].copy()

    if col_total_all and col_total_dir and col_total_ind:
        resumen["TotalUSDkg"]    = det[col_total_all]
        resumen["DirectoUSDkg"]  = det[col_total_dir]
        resumen["IndirectoUSDkg"]= det[col_total_ind]
    else:
        dcols = [c for c in det.columns
                 if c != "SKU" and
                 ("indirectos" not in c.lower()) and ("total" not in c.lower()) and ("indirecta" not in c.lower()) and ("totales" not in c.lower())]
        icols = [c for c in det.columns if ("indirectos" in c.lower()) or ("indirecta" in c.lower())]

        resumen["DirectoUSDkg"] = det[dcols].sum(axis=1, min_count=1) if dcols else np.nan
        resumen["IndirectoUSDkg"] = det[icols].sum(axis=1, min_count=1) if icols else 0.0
        resumen["TotalUSDkg"] = resumen["DirectoUSDkg"].fillna(0) + resumen["IndirectoUSDkg"].fillna(0)

    # Orden amigable
    resumen = resumen[["SKU","DirectoUSDkg","IndirectoUSDkg","TotalUSDkg"]]

    # DETALLE: dejamos SKU + todas las columnas de costo (numéricas) tal cual para mostrar en la UI
    detalle = det.copy()

    return resumen, detalle

def build_fact_precios(df_p: pd.DataFrame) -> pd.DataFrame:
    """
    Espera FACT_PRECIOS con: SKU, Año, Mes, PrecioVentaUSD
    Devuelve precios limpios + FechaClave (YYYYMM)
    """
    needed = {"SKU","Año","Mes","PrecioVentaUSD"}
    if not needed.issubset(set(df_p.columns)):
        raise ValueError(f"FACT_PRECIOS debe contener {needed}. Columnas: {df_p.columns.tolist()}")

    p = df_p.copy()
    p.columns = [c.strip() for c in p.columns]
    p = ensure_str(p, "SKU")
    p["Año"] = p["Año"].apply(lambda x: int(str(x).strip()))
    p["MesNum"] = p["Mes"].apply(month_to_num).astype("Int64")
    p["PrecioVentaUSD"] = p["PrecioVentaUSD"].apply(to_number_safe)
    p = p.dropna(subset=["PrecioVentaUSD"])
    p["FechaClave"] = p["Año"]*100 + p["MesNum"].astype(int)
    return p

def build_dim_sku(df_dim: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas en español:
      - SKU (obligatoria)
      - Condicion, Descripcion, Marca, Especie, Cliente ID (opcionales)
    Devuelve una tabla única por SKU con esas columnas limpias (str).
    """
    dim = df_dim.copy()
    dim.columns = [c.strip() for c in dim.columns]
    if "SKU" not in dim.columns:
        raise ValueError("En 'DIM_SKU' no se encontró columna 'SKU'.")

    # Asegura columnas esperadas (si faltan, las crea vacías)
    expected = ["SKU", "Condicion", "Descripcion", "Marca", "Especie", "Cliente"]
    for c in expected:
        if c not in dim.columns:
            dim[c] = np.nan

    # Limpieza básica
    for c in expected:
        dim[c] = dim[c].astype(str).str.strip()

    # Si hay columnas duplicadas por SKU, nos quedamos con la primera aparición
    dim = dim[expected].drop_duplicates(subset=["SKU"], keep="first").reset_index(drop=True)
    return dim

def compute_latest_price(precios: pd.DataFrame, mode="global", ref_datekey=None) -> pd.DataFrame:
    """
    Devuelve por SKU: PriceUSDkg (último precio) y LastDateKey.
    mode="global": último DateKey para cada SKU (independiente de rango).
    mode="to_date": último ≤ ref_datekey.
    """
    p = precios.sort_values(["SKU","FechaClave"]).reset_index(drop=True)
    if mode == "to_date":
        if ref_datekey is None:
            raise ValueError("ref_datekey es requerido con mode='to_date'.")
        p = p[p["FechaClave"] <= ref_datekey]
    idx = p.groupby("SKU")["FechaClave"].idxmax()
    latest = p.loc[idx, ["SKU","PrecioVentaUSD","FechaClave"]].rename(
        columns={"PrecioVentaUSD":"PrecioVentaUSDkg"})
    return latest.reset_index(drop=True)

@st.cache_data(show_spinner=True)
def build_mart(uploaded_bytes: bytes, ultimo_precio_modo: str, ref_ym: int|None):
    """Pipeline completo a partir del Excel subido."""
    sheets = read_workbook(uploaded_bytes)
    # Validación de hojas requeridas
    missing = validate_required_sheets(sheets)
    if missing:
        raise ValueError(f"Faltan hojas requeridas: {missing}")

    # 1) Costos ponderados
    costos_resumen, costos_detalle = build_tbl_costos_pond(sheets["FACT_COSTOS_POND"])

    # 2) Precios + último precio por SKU
    precios = build_fact_precios(sheets["FACT_PRECIOS"])
    if ultimo_precio_modo == "global":
        latest = compute_latest_price(precios, mode="global")
    else:
        latest = compute_latest_price(precios, mode="to_date", ref_datekey=ref_ym)

    # 3) DIM_SKU
    dim = build_dim_sku(sheets["DIM_SKU"])

    # 4) Unión (resumen + último precio + atributos DIM)
    mart = costos_resumen.merge(latest, on="SKU", how="left")
    mart = mart.merge(dim, on="SKU", how="right")

    detalle = costos_detalle.merge(latest, on="SKU", how="left")
    detalle = detalle.merge(dim, on="SKU", how="right")

    # 4) Métricas de margen unitario
    mart["MargenUSDkg"] = mart["PrecioVentaUSDkg"] - mart["TotalUSDkg"].abs() if "TotalUSDkg" != 0 else np.nan
    mart["MargenDirectoUSDkg"] = mart["PrecioVentaUSDkg"] - mart["DirectoUSDkg"].abs() if "DirectoUSDkg" != 0 else np.nan
    mart["MargenPct"] = np.where(
        mart["PrecioVentaUSDkg"].abs() > 1e-12,
        mart["MargenUSDkg"] / mart["PrecioVentaUSDkg"],
        np.nan
    )
    mart["MargenDirectoPct"] = np.where(
        mart["PrecioVentaUSDkg"].abs() > 1e-12,
        mart["MargenDirectoUSDkg"] / mart["PrecioVentaUSDkg"],
        np.nan
    )


    # Orden amigable
    mart = mart.sort_values("SKU", ascending=True).reset_index(drop=True)
    return mart, detalle

def apply_simulation(df: pd.DataFrame, price_up=0.0, direct_up=0.0, indirect_up=0.0):
    """
    Aplica multiplicadores de simulación por grupo (globales).
    Retorna un DataFrame con columnas *_Sim y deltas.
    """
    sim = df.copy()
    sim["PrecioVentaUSDkg_Sim"]   = df["PrecioVentaUSDkg"]   * (1 + price_up/100.0)
    sim["DirectoUSDkg_Sim"]  = df["DirectoUSDkg"]  * (1 + direct_up/100.0)
    sim["IndirectoUSDkg_Sim"]= df["IndirectoUSDkg"]* (1 + indirect_up/100.0)
    sim["TotalUSDkg_Sim"]   = sim["DirectoUSDkg_Sim"] + sim["IndirectoUSDkg_Sim"]
    sim["MargenUSDkg_Sim"]  = sim["PrecioVentaUSDkg_Sim"] - sim["TotalUSDkg_Sim"]
    sim["MargenPct_Sim"] = np.where(
        sim["PrecioVentaUSDkg_Sim"].abs() > 1e-12,
        sim["MargenUSDkg_Sim"] / sim["PrecioVentaUSDkg_Sim"],
        np.nan
    )
    # deltas
    sim["Δpp_MargenPct"] = (sim["MargenPct_Sim"] - sim["MargenPct"])*100
    sim["Δ_"] = sim["MargenUSDkg_Sim"] - sim["MargenUSDkg"]
    return sim

def to_excel_download(df: pd.DataFrame, filename="export.xlsx"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="data")
    st.download_button("⬇️ Descargar Excel", data=buf.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===================== UI =====================
st.set_page_config(page_title=ST_TITLE, layout="wide")
st.title(ST_TITLE)

with st.sidebar:
    st.header("1) Subir archivo maestro (.xlsx)")
    up = st.file_uploader("Selecciona tu Excel con hojas: " + ", ".join(REQ_SHEETS.keys()),
                          type=["xlsx"], accept_multiple_files=False)
    st.caption("El archivo debe contener al menos: " + " | ".join([f"**{k}** ({v})" for k,v in REQ_SHEETS.items()]))

    st.header("2) Parámetros de precio vigente")
    modo = st.radio("Último precio por SKU", ["global","to_date"], horizontal=True)
    ref_ym = None
    if modo == "to_date":
        # Selecciona una fecha (Año-Mes) para construir YYYYMM
        ref_date = st.date_input("Hasta fecha (se usa AñoMes)", value=date(2025,6,1))
        ref_ym = ref_date.year*100 + ref_date.month

    st.header("3) Simulación (multiplicadores %)")
    price_up = st.number_input("Precio: % Δ", value=0.0, step=0.5, format="%.2f")
    direct_up = st.number_input("Costos Directos: % Δ", value=0.0, step=0.5, format="%.2f")
    indirect_up = st.number_input("Costos Indirectos: % Δ", value=0.0, step=0.5, format="%.2f")

    st.markdown("---")
    st.caption("Consejo: si tus números vienen con coma decimal (3,071), este app los limpia automáticamente.")

if up is None:
    st.info("Sube tu archivo para comenzar.")
    st.stop()

# Procesamiento (cacheado por bytes del archivo + params)
file_bytes = up.read()
try:
    mart, detalle = build_mart(file_bytes, ultimo_precio_modo=modo, ref_ym=ref_ym)
except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()

# -------- Filtros sin orden (cascada dinámica) --------
# -------- Filtros sin orden (cascada dinámica) --------
st.subheader("Filtros")

# Posibles nombres (alias) por campo lógico
FIELD_ALIASES = {
    "Marca": ["Marca"],
    "Cliente": ["Cliente", "Cliente ID", "Customer", "ClienteID"],
    "Especie": ["Especie", "Species"],
    "Condicion": ["Condicion", "Condición", "Condition"],
    "SKU": ["SKU"]
}

# Resolver alias -> columna real presente en mart
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

RESOLVED = resolve_columns(mart, FIELD_ALIASES)

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

# Estado de selecciones
if "filters" not in st.session_state:
    st.session_state.filters = {k: [] for k in FILTER_FIELDS}
else:
    # Si cambió el set de filtros por alias/resolución, sincroniza
    st.session_state.filters = {k: st.session_state.filters.get(k, []) for k in FILTER_FIELDS}

cols = st.columns(len(FILTER_FIELDS) if FILTER_FIELDS else 1)

# Multiselects con opciones dependientes del resto, en cualquier orden
for i, logical in enumerate(FILTER_FIELDS):
    with cols[i]:
        real_col = RESOLVED[logical]
        df_except = _apply_filters(mart, st.session_state.filters, skip_key=logical)
        opts = sorted(_norm_series(df_except[real_col]).unique().tolist())
        current = [x for x in st.session_state.filters.get(logical, []) if x in opts]
        sel = st.multiselect(logical, options=opts, default=current, key=f"ms_{logical}")
        st.session_state.filters[logical] = sel

# Aplica todos los filtros
df_filtrado = _apply_filters(mart, st.session_state.filters).copy()

# Orden por SKU si existe y sin índice
sku_col = RESOLVED.get("SKU")
if sku_col in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values([sku_col]).reset_index(drop=True)
else:
    df_filtrado = df_filtrado.reset_index(drop=True)

# -------- Mostrar resultados --------
st.subheader("Márgenes actuales (unitarios)")
base_cols = ["SKU","Descripcion","Marca","Cliente","Especie","Condicion","PrecioVentaUSDkg","DirectoUSDkg","IndirectoUSDkg","TotalUSDkg","MargenDirectoUSDkg","MargenDirectoPct","MargenUSDkg","MargenPct"]
view_base = df_filtrado[base_cols].copy()
view_base.set_index("SKU", inplace=True)
view_base = view_base.sort_index()
st.dataframe(
    view_base.style.format({
        "SKU":"{}", "Descripcion":"{}", "Marca":"{}", "Cliente":"{}", "Especie":"{}", "Condicion":"{}",
        "PrecioVentaUSDkg":"{:.3f}",
        "DirectoUSDkg":"{:.3f}",
        "IndirectoUSDkg":"{:.3f}",
        "TotalUSDkg":"{:.3f}",
        "MargenDirectoUSDkg":"{:.3f}",
        "MargenDirectoPct":"{:.1%}",
        "MargenUSDkg":"{:.3f}",
        "MargenPct":"{:.1%}"
    }),
    use_container_width=True, height=420
)
# --- Toggle: ver detalle de costos respetando los filtros vigentes ---
expand = st.toggle("🔎 Expandir costos por SKU (temporada)", value=False)

if expand:
    # 1) Toma los SKUs actualmente visibles (ya filtrados arriba)
    skus_filtrados = df_filtrado["SKU"].astype(str).unique().tolist()

    # 2) Filtra el detalle por esos SKUs
    det = detalle[detalle["SKU"].astype(str).isin(skus_filtrados)].copy()

    # 3) Mueve atributos DIM a la derecha
    dim_candidatas = ["Descripcion","Marca","Cliente","Especie","Condicion"]
    dim_cols = [c for c in dim_candidatas if c in det.columns]
    last_cols = [c for c in det.columns if c not in dim_cols]
    det = det[dim_cols + last_cols]

    # 4) Orden y formato
    det = det.sort_values(["SKU"]).reset_index(drop=True)
    view_base_det = det.copy()
    view_base_det.set_index("SKU", inplace=True)

    fmt_cols = {
        c: "{:.3f}" for c in det.columns
        if c not in (["SKU"] + dim_cols) and np.issubdtype(det[c].dtype, np.number)
    }

    st.subheader("Detalle de costos por SKU (temporada)")
    st.dataframe(view_base_det.style.format(fmt_cols), use_container_width=True, height=700)

    # 5) Descargar
    to_excel_download(det, "costos_detalle_temporada.xlsx")