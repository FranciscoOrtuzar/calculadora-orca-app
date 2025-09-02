import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Dict, List, Optional, Tuple

# === Configura aquí ===
INPUT = "/Users/franciscoortuzar/Desktop/Or-Ca Consulting/Datos de prueba.xlsx"
OUTPUT = "FACT_PRECIOS.xlsx"

MESES = ["Octubre","Noviembre","Diciembre","Enero","Febrero","Marzo","Abril","Mayo","Junio"]
MAP_ANIO = {"Octubre":2024,"Noviembre":2024,"Diciembre":2024,
            "Enero":2025,"Febrero":2025,"Marzo":2025,"Abril":2025,"Mayo":2025,"Junio":2025}

# Etiquetas aceptadas para el bloque de precio (fila 5)
PRICE_LABELS = ["precio de venta", "precio de venta desde comex", "precio de venta comex"]

# (Opcional) fuerza un rango de columnas por letra Excel, p.ej. ("O","W")
MANUAL_RANGE: Optional[Tuple[str,str]] = ("Q", "Y")

# ========== utilidades ==========

def norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.strip()

def norm_lower(s: str) -> str:
    return norm(s).lower()

def to_number_safe(x):
    """Convierte '-',' 3,071 ', '1.234,56' en float robusto."""
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\xa0"," ")
    if s in {"", "-", "—"}: return np.nan
    s = s.replace(" ", "")
    # si hay coma y no hay punto -> considerar coma decimal (es-CL)
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # caso en-US: quitar miles
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def excel_col_to_idx(col_letter: str) -> int:
    """Convierte 'A'->0, 'B'->1, ..., 'AA'->26, etc."""
    col_letter = col_letter.strip().upper()
    val = 0
    for ch in col_letter:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Columna inválida: {col_letter}")
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return val - 1  # 0-based

def find_month_row(raw: pd.DataFrame, meses: List[str]) -> Optional[int]:
    """Devuelve el índice de fila donde aparecen los meses (la primera coincidencia suficiente)."""
    for r in range(min(100, len(raw))):
        vals = [norm(v) for v in raw.iloc[r].tolist()]
        hits = sum(1 for v in vals if v in meses)
        if hits >= 3:  # con 3 ya es muy probable que sea la fila de meses
            return r
    return None

def forward_fill_row(values: List[str]) -> List[str]:
    """Forward-fill a lo largo de columnas para lidiar con celdas fusionadas (NaN a la derecha)."""
    out = []
    last = ""
    for v in values:
        v2 = norm(v)
        if v2 == "" or v2.lower() in {"nan", "none"}:
            out.append(last)
        else:
            out.append(v2)
            last = v2
    return out

def find_price_block(raw: pd.DataFrame,
                     month_row: int,
                     price_labels: List[str],
                     manual_range: Optional[Tuple[str,str]] = None
) -> Dict[str, int]:
    """
    Encuentra, por posición, las columnas de meses bajo el bloque 'Precio de Venta'.
    Retorna dict {Mes -> col_idx} usando índices de columna (no nombres).
    """
    price_labels_n = [norm_lower(x) for x in price_labels]

    # fila de rótulos de secciones (tu "fila 5"):
    type_row = month_row
    if type_row < 0:
        raise ValueError("No hay fila de rótulos (fila 5) por encima de la fila de meses.")

    # Valores crudos
    type_vals = raw.iloc[type_row].tolist()
    month_vals = raw.iloc[month_row].tolist()

    # forward fill para propagar 'Precio de Venta' a la derecha
    type_vals_ff = forward_fill_row(type_vals)

    # Rango manual (opcional)
    j_start, j_end = 0, raw.shape[1]
    if manual_range is not None:
        j_start = max(0, excel_col_to_idx(manual_range[0]))
        j_end = min(raw.shape[1], excel_col_to_idx(manual_range[1]) + 1)

    # Recolectar columnas cuya etiqueta ffill == 'precio de venta' y el mes es reconocido
    candidates: Dict[str,int] = {}
    for j in range(j_start, j_end):
        label = norm_lower(type_vals_ff[j]) if j < len(type_vals_ff) else ""
        mes = norm(month_vals[j]) if j < len(month_vals) else ""
        if label in price_labels_n and mes in MESES and mes not in candidates:
            candidates[mes] = j

    return candidates

# ========== pipeline ==========

# 1) Leer crudo sin header para detectar filas/posiciones
raw = pd.read_excel(INPUT, header=None, dtype=str, engine="openpyxl", sheet_name="Hoja1")

month_row = find_month_row(raw, MESES)
if month_row is None:
    raise ValueError("No pude encontrar la fila de meses (Octubre…Junio).")

# 2) Encontrar columnas de meses bajo el bloque 'Precio de Venta'
price_cols = find_price_block(raw, 1, PRICE_LABELS, MANUAL_RANGE)

# 3) Leer datos con header en la fila de meses para extraer valores por índice
df = pd.read_excel(INPUT, dtype=str, header=1, engine="openpyxl", sheet_name="Hoja1")
df.columns = [str(c).strip() for c in df.columns]

# SKU (tolerante a cambios de posición)
sku_col = next((c for c in df.columns if norm_lower(c) == "sku"), None)
if sku_col is None:
    raise ValueError("No encuentro la columna 'SKU' en la fila de encabezados de datos.")
sku_series = df[sku_col].astype(str).str.strip()

# 4) Construir FACT_PRECIOS usando SOLO las columnas detectadas
rows = []
for mes in MESES:  # mantiene orden de meses
    if mes not in price_cols:
        continue  # si no existe ese mes en el bloque, se omite
    j = price_cols[mes]
    # tomar la columna por índice (robusto ante duplicados de nombre)
    serie = df.iloc[:, j].apply(to_number_safe)
    tmp = pd.DataFrame({
        "SKU": sku_series,
        "Año": MAP_ANIO[mes],
        "Mes": mes,
        "PrecioVentaUSD": serie
    })
    rows.append(tmp)

if not rows:
    raise ValueError("No se detectaron columnas de 'Precio de Venta'. "
                     "Revisa PRICE_LABELS o activa MANUAL_RANGE = ('O','W').")

precios = pd.concat(rows, ignore_index=True)
precios = precios.dropna(subset=["PrecioVentaUSD"])

# 5) Orden y salida
precios["Año"] = precios["Año"].astype(int)
precios["Mes"] = pd.Categorical(precios["Mes"], categories=MESES, ordered=True)
precios = precios.sort_values(["SKU","Año","Mes"]).reset_index(drop=True)

precios.to_excel(OUTPUT, index=False)
print(f"Listo -> {OUTPUT}  filas: {len(precios)}")

# Debug opcional
print(f"[DEBUG] month_row={month_row}, cols detectadas: {[(m, price_cols[m]) for m in price_cols]}")