import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

# ====== Config ======
INPUT = '/Users/franciscoortuzar/Desktop/Or-Ca Consulting/Costos Granel Sucios.xlsx'  # tu archivo
SHEET_NAME = 0                                                           # o "Hoja1"
OUTPUT = "FACT_COSTOS_GRANEL.xlsx"

# Meses admitidos (nombres, abreviaturas y números)
MESES_ORD = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre",
    "Octubre","Noviembre","Diciembre",
    "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
    "1","2","3","4","5","6","7","8","9","10","11","12"
]
MES2NUM = {
    "Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,"Julio":7,"Agosto":8,"Septiembre":9,
    "Octubre":10,"Noviembre":11,"Diciembre":12,
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}

# ====== Utils ======
def norm(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip()

def looks_like_year(x) -> bool:
    x = str(x).strip()
    return x.isdigit() and 1900 <= int(x) <= 2100

def find_month_row(raw: pd.DataFrame, max_scan: int = 80) -> Optional[int]:
    for r in range(min(max_scan, len(raw))):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        hits = sum(1 for v in vals if v in MESES_ORD)
        if hits >= 4:
            return r
    return None

def find_year_row(raw: pd.DataFrame, month_row: int) -> Optional[int]:
    for r in range(month_row-1, -1, -1):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        if any(looks_like_year(v) for v in vals):
            return r
    return None

def forward_fill_row(values: List[str]) -> List[str]:
    out, last = [], ""
    for v in values:
        vv = norm(v)
        if vv == "" or vv.lower() in {"nan","none"}:
            out.append(last)
        else:
            out.append(vv); last = vv
    return out

def to_number_safe(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\xa0"," ").replace(" ", "")
    if s in {"", "-", "—"}:
        return np.nan
    # Soporta "1.234,56" (es-CL) y "1,234.56" (en-US)
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def find_sku_col(df_cols: pd.Index) -> int:
    """
    Prioriza 'SKU'. Si no, 'SKU-Cliente'. Evita substring (sku dentro de sku-cliente).
    """
    names = [str(c).strip().lower() for c in df_cols]
    for idx, name in enumerate(names):
        if name == "sku":
            return idx
    for idx, name in enumerate(names):
        if name == "sku-cliente":
            return idx
    for idx, name in enumerate(names):
        if name in {"codigo sisfrigo", "código sisfrigo", "cod.sisfrigo"}:
            return idx
    return 0

def normalize_series(s: pd.Series) -> pd.Series:
    """
    Normaliza texto en Serie: trim, NFKD (quita acentos), minúsculas, colapsa espacios.
    """
    s = s.astype(str).str.strip()
    s = s.str.normalize('NFKD').str.encode('ascii','ignore').str.decode('ascii')
    s = s.str.lower().str.replace(r"\s+", " ", regex=True)
    return s

def find_cmg_col_index(raw_df: pd.DataFrame, max_scan_rows: int = 20) -> Optional[int]:
    """
    Busca la columna 'Costo MMPP con Granel' (u otras variantes) en las primeras filas.
    Retorna índice de columna o None.
    """
    targets = {
        "costo mmpp con granel",
        "costo mmpp granel",
        "mmpp con granel",
        "costo mmpp"  # fallback
    }
    R = min(max_scan_rows, raw_df.shape[0])
    C = raw_df.shape[1]
    for r in range(R):
        for j in range(C):
            cell = raw_df.iat[r, j]
            if pd.isna(cell):
                continue
            t = str(cell)
            t_norm = normalize_series(pd.Series([t])).iloc[0]
            if t_norm in targets:
                return j
    return None

# ====== Proceso ======
# 1) Leer crudo sin header
raw = pd.read_excel(INPUT, header=None, dtype=str, engine="openpyxl", sheet_name=SHEET_NAME)

month_row = find_month_row(raw)
if month_row is None:
    raise ValueError("No pude localizar la fila de meses.")
year_row = find_year_row(raw, month_row)
if year_row is None:
    raise ValueError("No pude localizar la fila de años (encima de meses).")

# 2) filas clave
month_vals_raw = [norm(v) for v in raw.iloc[month_row].tolist()]
year_vals_raw  = [norm(v) for v in raw.iloc[year_row].tolist()]
year_vals = forward_fill_row(year_vals_raw)
month_vals = month_vals_raw  # si hay merges también en meses, usa forward_fill_row

# 3) tipo_costo está en la fila INMEDIATAMENTE SUPERIOR a la de años
tipo_row_raw = [norm(v) for v in raw.iloc[year_row - 1].tolist()]
tipo_row_ff  = forward_fill_row(tipo_row_raw)

# 4) Mapa de columnas -> (Año, MesNum, TipoCosto)
col_map: Dict[int, Tuple[int, int, str]] = {}
for j in range(raw.shape[1]):
    y = year_vals[j] if j < len(year_vals) else ""
    m = month_vals[j] if j < len(month_vals) else ""
    t = tipo_row_ff[j] if j < len(tipo_row_ff) else ""
    if looks_like_year(y) and t != "":
        # Mes a número
        if str(m).isdigit():
            mes_num = int(m)
        else:
            mes_num = MES2NUM.get(m, None)
        if mes_num is not None and 1 <= mes_num <= 12:
            col_map[j] = (int(y), mes_num, t)

if not col_map:
    raise ValueError("No encontré columnas válidas de (Año, Mes, TipoCosto).")

# 5) Leer con header en month_row para alinear posiciones
df = pd.read_excel(INPUT, dtype=str, header=month_row, engine="openpyxl", sheet_name=SHEET_NAME)

# 6) Detectar columna SKU
sku_col_idx = find_sku_col(df.columns)
sku_series = df.iloc[:, sku_col_idx].astype(str).str.strip()

# 7) Construir formato largo base
rows = []
for j, (anio, mes_num, tipo_costo) in col_map.items():
    serie = df.iloc[:, j].apply(to_number_safe)
    tmp = pd.DataFrame({
        "SKU": sku_series,
        "Año": int(anio),
        "Mes": int(mes_num),
        "FechaClave": int(anio) * 100 + int(mes_num),
        "tipo_costo": tipo_costo,
        "valor_costo": serie
    })
    rows.append(tmp)

costos_long = pd.concat(rows, ignore_index=True)
costos_long = costos_long.dropna(subset=["valor_costo"]).copy()
costos_long["SKU"] = costos_long["SKU"].astype(str).str.strip()

# --- EXCLUIR "Precio de Venta desde Comex" (vectorizado, sin warning) ---
tipo_norm = normalize_series(costos_long["tipo_costo"])
mask_excluir = (tipo_norm == "precio de venta desde comex")
costos_long = costos_long.loc[~mask_excluir].copy()

# 8) Agregar "Costo MMPP con Granel" constante por SKU a todos los meses visibles
cmg_idx = find_cmg_col_index(raw)
if cmg_idx is not None:
    serie_cmg = df.iloc[:, cmg_idx].apply(to_number_safe)
    cmg_by_sku = pd.DataFrame({
        "SKU": sku_series.astype(str).str.strip(),
        "valor_costo": serie_cmg
    })
    # meses ya presentes por SKU (si quieres calendario completo, genera uno aparte)
    base_mes = costos_long[["SKU","Año","Mes","FechaClave"]].drop_duplicates().copy()
    cmg_long = base_mes.merge(cmg_by_sku, on="SKU", how="left")
    cmg_long = cmg_long[cmg_long["valor_costo"].notna()].copy()
    cmg_long["tipo_costo"] = "Costo MMPP con Granel"

    costos_long = pd.concat(
        [costos_long, cmg_long[["SKU","Año","Mes","FechaClave","tipo_costo","valor_costo"]]],
        ignore_index=True
    )
else:
    print("⚠️ No se encontró la columna 'Costo MMPP con Granel' en las filas de encabezado. Continúo sin este costo fijo.")

# 9) Orden y export
costos_long = costos_long.sort_values(["SKU","Año","Mes","tipo_costo"]).reset_index(drop=True)
costos_long.to_excel(OUTPUT, index=False)
print(f"OK -> {OUTPUT}  (filas: {len(costos_long)})")