import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Optional, Tuple, List, Dict

# ========= Config =========
INPUT  = "/Users/franciscoortuzar/Downloads/precio tons comex.xlsx"
SHEET_NAME = "Hoja1"
OUTPUT = "FACT_VOL2.xlsx"

# Meses (es-CL)
# MESES_ORD = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
#              "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MESES_ORD = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
MES2IDX = {m:i for i,m in enumerate(MESES_ORD)}

# Si quieres limitar el período (ej. Oct-2024 a Jun-2025), pon rangos aquí;
# si no, deja None y exportará todo lo encontrado.
PERIODO_DESDE: Optional[Tuple[int,str]] = (2024, "10")
PERIODO_HASTA: Optional[Tuple[int,str]] = (2025, "6")

# ========= Utilidades =========
def norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.strip()

def norm_low(s: str) -> str:
    return norm(s).lower()

def to_number_safe(x):
    """Convierte formatos '1.234,56' (es-CL) o '1,234.56' (en-US) a float."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\xa0"," ")
    if s in {"", "-", "—"}: return np.nan
    s = s.replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def looks_like_year(x) -> bool:
    x = str(x).strip()
    return len(x) == 4 and x.isdigit() and 1900 <= int(x) <= 2100

def find_month_row(raw: pd.DataFrame) -> Optional[int]:
    """Encuentra la fila que contiene varios meses (≥4 coincidencias)."""
    for r in range(min(120, len(raw))):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        hits = sum(1 for v in vals if v in MESES_ORD)
        if hits >= 4:
            return r
    return None

def find_year_row(raw: pd.DataFrame, month_row: int) -> Optional[int]:
    """Encuentra la fila de años por encima de la fila de meses."""
    for r in range(month_row-1, -1, -1):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        hits = sum(1 for v in vals if looks_like_year(v))
        if hits >= 1:
            return r
    return None

def forward_fill_row(values: List[str]) -> List[str]:
    """Propaga a la derecha (sirve para merges en Excel)."""
    out = []
    last = ""
    for v in values:
        vv = norm(v)
        if vv == "" or vv.lower() in {"nan","none"}:
            out.append(last)
        else:
            out.append(vv)
            last = vv
    return out

def in_periodo(year: int, mes: str) -> bool:
    if mes not in MES2IDX:
        return False
    y, m = year, mes
    if PERIODO_DESDE is not None:
        y0, m0 = PERIODO_DESDE
        if (y < y0) or (y == y0 and MES2IDX[m] < MES2IDX[m0]):
            return False
    if PERIODO_HASTA is not None:
        y1, m1 = PERIODO_HASTA
        if (y > y1) or (y == y1 and MES2IDX[m] > MES2IDX[m1]):
            return False
    return True

def find_sku_col(raw: pd.DataFrame, month_row: int) -> Tuple[int, int]:
    label_row = month_row - 1
    sku_idx, sku_cliente_idx = None, None

    for j in range(raw.shape[1]):
        v = norm_low(raw.iat[label_row, j]) if label_row >= 0 else ""

        # Match exacto o por palabra
        if v.strip() in {"sku", "código sisfrigo", "cod.sisfrigo"}:
            sku_idx = j
        elif v.strip() in {"sku-cliente", "sku cliente", "sku_cliente"}:
            sku_cliente_idx = j

    # Si no encontró, fallback
    if sku_idx is None:
        sku_idx = 0
    if sku_cliente_idx is None:
        sku_cliente_idx = sku_idx

    return sku_idx, sku_cliente_idx

# ========= Proceso principal =========
# 1) Leer crudo sin header
raw = pd.read_excel(INPUT, header=None, dtype=str, engine="openpyxl", sheet_name=SHEET_NAME)

month_row = find_month_row(raw)
if month_row is None:
    raise ValueError("No pude localizar la fila de meses (Ene..Dic).")
year_row = find_year_row(raw, month_row)
if year_row is None:
    raise ValueError("No pude localizar la fila de años por encima de la fila de meses.")

# 2) Fila de años y meses (con forward fill para merges)
year_vals = forward_fill_row([norm(v) for v in raw.iloc[year_row].tolist()])
month_vals = [norm(v) for v in raw.iloc[month_row].tolist()]

# 3) Mapear columnas -> (Año, Mes) si ambos son válidos
col_map: Dict[int, Tuple[int, str]] = {}
for j in range(raw.shape[1]):
    y = year_vals[j] if j < len(year_vals) else ""
    m = month_vals[j] if j < len(month_vals) else ""
    if looks_like_year(y) and m in MESES_ORD:
        col_map[j] = (int(y), m)

if not col_map:
    raise ValueError("No encontré columnas que combinen Año + Mes.")

# 4) Encontrar columna de SKU
sku_col_idx, sku_cliente_col_idx = find_sku_col(raw, month_row)

# 5) Leer datos con header en la fila de meses (para indexar por posición con seguridad)
df = pd.read_excel(INPUT, dtype=str, header=month_row, engine="openpyxl", sheet_name=SHEET_NAME)
# Asegura que la cantidad de columnas coincida
if df.shape[1] < raw.shape[1]:
    # si por alguna razón pandas truncó, ajusta col_map a rango de df
    col_map = {j:v for j,v in col_map.items() if j < df.shape[1]}
if sku_col_idx >= df.shape[1]:
    raise ValueError("La columna de SKU detectada está fuera del rango de datos.")

# Serie de SKU
sku_series = df.iloc[:, sku_col_idx].astype(str).str.strip()
sku_cliente_series = df.iloc[:, sku_cliente_col_idx].astype(str).str.strip()

# Fallback: si quedaron iguales o SKU está vacío, derivar SKU desde SKU-Cliente
if sku_series.equals(sku_cliente_series) or (sku_series == "").all():
    s = sku_cliente_series.str.replace(r"\s+", "", regex=True)

    # Caso mixto: si es numérico -> divide por 10; si no, quita último carácter
    mask_num = s.str.fullmatch(r"\d+")
    sku_series = pd.Series(index=s.index, dtype=object)
    # numérico: 12345 -> 1234 (quita último dígito)
    sku_series[mask_num] = (s[mask_num].astype("int64") // 10).astype(str)
    # alfanumérico: ABCD1 o ABCDY -> ABCD (quita último carácter)
    sku_series[~mask_num] = s[~mask_num].str[:-1]

# 6) Construir FACT_VOL
rows = []
for j, (anio, mes) in col_map.items():
    if not in_periodo(anio, mes):
        continue
    serie = df.iloc[:, j].apply(to_number_safe)
    tmp = pd.DataFrame({
        "SKU": sku_series,
        "SKU-Cliente": sku_cliente_series,  # Asumimos que es el mismo para este caso
        "Año": anio,
        "Mes": mes,
        "KgEmbarcados": serie
    })
    rows.append(tmp)

if not rows:
    raise ValueError("No hay datos dentro del período configurado o no se pudieron leer valores numéricos.")

fact_vol = pd.concat(rows, ignore_index=True)
fact_vol = fact_vol.dropna(subset=["KgEmbarcados"])

# 7) Orden y export
fact_vol["Año"] = fact_vol["Año"].astype(int)
fact_vol["Mes"] = pd.Categorical(fact_vol["Mes"], categories=MESES_ORD, ordered=True)
fact_vol = fact_vol.sort_values(["SKU","Año","Mes"]).reset_index(drop=True)

fact_vol.to_excel(OUTPUT, index=False)
print(f"Listo -> {OUTPUT} (filas: {len(fact_vol)})")