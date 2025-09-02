import pandas as pd
import numpy as np
import unicodedata
from typing import Optional, Tuple, List, Dict

# ========= Config =========
INPUT  = "/Users/franciscoortuzar/Desktop/Or-Ca Consulting/Datos de prueba.xlsx"
SHEET_NAME = "Costos Granel"          # <-- cambia al nombre real de tu hoja
OUTPUT = "FACT_GRANEL.xlsx"

# Columnas identificadoras esperadas (ajusta a tu layout)
# Intenta encontrarlas cerca de la fila de meses; si alguna no aparece, la omite.
BASE_ID_COLS = ["Planta", "Equipo", "Proceso"]  # puedes agregar "Línea", "PlantaID", etc.

# Meses y período (Oct-2024 -> Jun-2025)
MESES_ORD = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
             "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MES2IDX = {m:i for i,m in enumerate(MESES_ORD)}
PERIODO_DESDE: Optional[Tuple[int,str]] = (2024, "Octubre")
PERIODO_HASTA: Optional[Tuple[int,str]] = (2025, "Junio")

# Mapeo de tipos (tolerante a variantes). Clave = texto del bloque, Valor = TipoCosto estándar
COSTO_MAP = {
    "mano de obra (directa)": "MO_Directa",
    "mano de obra directa": "MO_Directa",
    "mano de obra indirecta": "MO_Indirecta",
    "materiales directos": "Materiales_Directos",
    "materiales indirectos": "Materiales_Indirectos",
    "laboratorio": "Laboratorio",
    "servicios generales": "ServiciosGenerales",
    "utilities": "Utilities",
    "mantencion y maquinaria": "MantencionMaquinaria",
    "mantenimiento y maquinaria": "MantencionMaquinaria",
    # agrega más alias si los usas
}

EXCLUIR_SUB = {"margen", "%"}  # por si aparecen columnas de % o de margen

# ========= Utilidades =========
def norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.strip()

def norm_low(s: str) -> str:
    return norm(s).lower()

def to_number_safe(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\xa0"," ")
    if s in {"", "-", "—"}: return np.nan
    s = s.replace(" ", "")
    # coma decimal si no hay punto
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def looks_like_year(x) -> bool:
    x = str(x).strip()
    return len(x) == 4 and x.isdigit() and 1900 <= int(x) <= 2100

def find_month_row(raw: pd.DataFrame) -> Optional[int]:
    for r in range(min(150, len(raw))):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        if sum(1 for v in vals if v in MESES_ORD) >= 4:
            return r
    return None

def find_year_row(raw: pd.DataFrame, month_row: int) -> Optional[int]:
    for r in range(month_row-1, -1, -1):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        if sum(1 for v in vals if looks_like_year(v)) >= 1:
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

def in_periodo(year: int, mes: str) -> bool:
    if mes not in MES2IDX: return False
    if PERIODO_DESDE is not None:
        y0, m0 = PERIODO_DESDE
        if (year < y0) or (year == y0 and MES2IDX[mes] < MES2IDX[m0]): return False
    if PERIODO_HASTA is not None:
        y1, m1 = PERIODO_HASTA
        if (year > y1) or (year == y1 and MES2IDX[mes] > MES2IDX[m1]): return False
    return True

def map_tipo_costo(label: str) -> Optional[str]:
    n = norm_low(label)
    # descarta subheaders no deseados
    if any(x in n for x in EXCLUIR_SUB):
        return None
    for k, v in COSTO_MAP.items():
        if k in n:
            return v
    return None

def find_id_cols(df: pd.DataFrame, month_row: int, base_names: List[str]) -> List[str]:
    """Busca columnas identificadoras por nombre, tolerante a estar 1-2 filas cerca de la fila de meses."""
    found = []
    # nombres normalizados presentes en df.columns
    cols_norm = {norm_low(c): c for c in df.columns}
    for want in base_names:
        w = norm_low(want)
        # intenta por coincidencia exacta
        if w in cols_norm:
            found.append(cols_norm[w])
            continue
        # intenta por 'contiene' (p.ej. 'planta id', 'equipo #')
        candidates = [orig for low, orig in cols_norm.items() if w in low]
        if candidates:
            found.append(candidates[0])
    return found

# ========= Proceso principal =========
# 1) Leer crudo sin header (HOJA CORRECTA)
raw = pd.read_excel(INPUT, sheet_name=SHEET_NAME, header=None, dtype=str, engine="openpyxl")

month_row = find_month_row(raw)
if month_row is None:
    raise ValueError("No pude localizar la fila de meses (Ene..Dic).")
year_row = find_year_row(raw, month_row)
if year_row is None:
    raise ValueError("No pude localizar la fila de años por encima de la fila de meses.")
type_row = month_row - 2
if type_row < 0:
    raise ValueError("No hay fila de 'tipos de costo' dos filas por encima de la fila de meses.")

# 2) Tomar filas de tipo, año y meses (y hacer forward fill por merges)
type_vals  = forward_fill_row([norm(v) for v in raw.iloc[type_row].tolist()])
year_vals  = forward_fill_row([norm(v) for v in raw.iloc[year_row].tolist()])
month_vals = [norm(v) for v in raw.iloc[month_row].tolist()]

# 3) Mapear columnas -> (TipoCosto, Año, Mes)
col_map: Dict[int, Tuple[str, int, str]] = {}
for j in range(raw.shape[1]):
    tipo_lbl = type_vals[j] if j < len(type_vals) else ""
    year_lbl = year_vals[j] if j < len(year_vals) else ""
    mes_lbl  = month_vals[j] if j < len(month_vals) else ""
    tipo_std = map_tipo_costo(tipo_lbl)
    if tipo_std and looks_like_year(year_lbl) and mes_lbl in MESES_ORD:
        col_map[j] = (tipo_std, int(year_lbl), mes_lbl)

if not col_map:
    raise ValueError("No encontré columnas que combinen TipoCosto + Año + Mes. Revisa encabezados y COSTO_MAP.")

# 4) Leer datos con header en la fila de meses (misma hoja)
df = pd.read_excel(INPUT, sheet_name=SHEET_NAME, dtype=str, header=month_row, engine="openpyxl")

# 5) Identificadores (Planta/Equipo/Proceso, etc.)
id_cols = find_id_cols(df, month_row, BASE_ID_COLS)
# si no encuentra ninguno, toma la primera columna como ID genérico
if not id_cols:
    id_cols = [df.columns[0]]

# 6) Construir FACT_GRANEL
rows = []
for j, (tipo, anio, mes) in col_map.items():
    if not in_periodo(anio, mes):
        continue
    serie = df.iloc[:, j].apply(to_number_safe)
    # arma el bloque
    tmp = pd.DataFrame({
        **{c: df[c].astype(str).str.strip() for c in id_cols},
        "Año": anio,
        "Mes": mes,
        "TipoCosto": tipo,
        "ValorUSD/Kg": serie
    })
    rows.append(tmp)

if not rows:
    raise ValueError("No hay datos útiles dentro del período configurado o no se pudieron leer valores numéricos.")

fact_granel = pd.concat(rows, ignore_index=True)
fact_granel = fact_granel.dropna(subset=["ValorUSD/Kg"])

# 7) Orden y export
fact_granel["Año"] = fact_granel["Año"].astype(int)
fact_granel["Mes"] = pd.Categorical(fact_granel["Mes"], categories=MESES_ORD, ordered=True)

sort_cols = id_cols + ["Año","Mes","TipoCosto"]
fact_granel = fact_granel.sort_values(sort_cols).reset_index(drop=True)

fact_granel.to_excel(OUTPUT, index=False)
print(f"Listo -> {OUTPUT} (filas: {len(fact_granel)})")
print(f"[DEBUG] month_row={month_row}, year_row={year_row}, type_row={type_row}, ids={id_cols}, cols tiempo={len(col_map)}")