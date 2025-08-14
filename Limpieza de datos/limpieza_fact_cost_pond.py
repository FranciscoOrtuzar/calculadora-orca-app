import pandas as pd
import numpy as np
import unicodedata

# ============== Configura aquí ==============
XLSX_PATH   = "/Users/franciscoortuzar/Desktop/Or-Ca Consulting/Datos de prueba.xlsx"
SHEET_IN    = "Costos Retail Agrupado"   # <— pon el nombre de TU hoja de entrada
SHEET_OUT   = "COSTOS_RETAIL_AGRUPADO"   # hoja limpia de salida (se reemplaza si existe)
HEADER_ROWS = [3, 4]                     # fila 4: MO/Materiales..., fila 5: Directa/Indirecta/...
# ============================================

FINAL_ORDER = [
    "MO Directa","MO Indirecta","MO Total",
    "Materiales Cajas y Bolsas","Materiales Indirectos","Materiales Total",
    "Calidad","Mantención","SGenerales","Utilities","Fletes","Comex","Guarda PT","Guarda MMPP"
]

# --- utilidades ---
def norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return " ".join(s.strip().split())

def to_number_safe(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\xa0"," ")
    if s in {"", "-", "—"}: return np.nan
    s = s.replace(" ", "")
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def flatten_cols(mi_cols):
    """
    Aplana columnas multi-nivel -> toma la última etiqueta informativa
    y la combina con la superior si ayuda a desambiguar.
    """
    flat = []
    for tup in mi_cols:
        parts = [norm(x) for x in tup if norm(x) not in {"", "nan", "None"}]
        if not parts:
            flat.append("")
        else:
            # típicamente: ["MO","Directa"] -> "MO Directa"
            flat.append(" ".join(parts))
    return flat

# --- leer hoja con encabezado multinivel (y fallback si hace falta) ---
try:
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_IN, header=HEADER_ROWS, dtype=str, engine="openpyxl")
    if not isinstance(df.columns, pd.MultiIndex):
        # por si el archivo llega con una sola fila de encabezado
        # leemos de nuevo con un solo header
        df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_IN, header=4, dtype=str, engine="openpyxl")
        df.columns = [norm(c) for c in df.columns]
        col_sku = None
        for c in df.columns:
            if norm(c).lower().startswith("sku"):
                col_sku = c
                break
        if col_sku is None:
            raise ValueError("No encontré columna SKU.")
        # renombra columnas conocidas directo
        rename_map = {}
        for c in df.columns:
            lc = norm(c).lower()
            if "directa" in lc and "mo" in lc: rename_map[c] = "MO Directa"
            elif "indirecta" in lc and "mo" in lc: rename_map[c] = "MO Indirecta"
            elif lc == "total" and "mo" in " ".join(df.columns).lower(): rename_map[c] = "MO Total"
            elif "cajas" in lc: rename_map[c] = "Materiales Cajas y Bolsas"
            elif "indirectos" in lc and "material" in lc: rename_map[c] = "Materiales Indirectos"
            elif lc == "total" and "material" in " ".join(df.columns).lower(): rename_map[c] = "Materiales Total"
            elif "calidad" in lc: rename_map[c] = "Calidad"
            elif "mantencion" in lc or "mantención" in lc: rename_map[c] = "Mantención"
            elif "sgenerales" in lc or "servicios generales" in lc: rename_map[c] = "SGenerales"
            elif "utilities" in lc: rename_map[c] = "Utilities"
            elif "fletes" in lc: rename_map[c] = "Fletes"
            elif "comex" in lc: rename_map[c] = "Comex"
            elif "guarda pt" in lc: rename_map[c] = "Guarda PT"
            elif "guarda mmpp" in lc: rename_map[c] = "Guarda MMPP"
        df = df.rename(columns=rename_map)
        # quedarnos con SKU + las columnas finales (las que existan)
        cols = ["SKU"] + [c for c in FINAL_ORDER if c in df.columns]
        df = df[cols].copy()
    else:
        # aplanar
        flat_cols = flatten_cols(df.columns)
        df.columns = flat_cols
        # detectar SKU
        col_sku = None
        for c in df.columns:
            if norm(c).lower().startswith("sku"):
                col_sku = c
                break
        if col_sku is None:
            raise ValueError("No encontré columna SKU.")
        # construir mapa de nombres -> finales
        rename_map = {}
        for c in df.columns:
            lc = norm(c).lower()
            # mapeos MO
            if lc.endswith("mo directa") or lc == "directa": rename_map[c] = "MO Directa"
            elif lc.endswith("mo indirecta") or lc == "indirecta": rename_map[c] = "MO Indirecta"
            elif lc.endswith("mo total") or (lc == "total" and "mo" in " ".join(flat_cols).lower()): rename_map[c] = "MO Total"
            # materiales
            elif "cajas y bolsas" in lc or ("cajas" in lc and "bolsas" in lc): rename_map[c] = "Materiales Cajas y Bolsas"
            elif ("indirectos" in lc and "materiales" in lc) or lc.endswith("materiales indirectos"): rename_map[c] = "Materiales Indirectos"
            elif lc.endswith("materiales total") or (lc == "total" and "materiales" in " ".join(flat_cols).lower()): rename_map[c] = "Materiales Total"
            # otros
            elif "calidad" in lc: rename_map[c] = "Calidad"
            elif "mantencion" in lc or "mantención" in lc: rename_map[c] = "Mantención"
            elif "s generales" in lc or "sgenerales" in lc or "servicios generales" in lc: rename_map[c] = "SGenerales"
            elif "utilities" in lc: rename_map[c] = "Utilities"
            elif "fletes" in lc: rename_map[c] = "Fletes"
            elif "comex" in lc: rename_map[c] = "Comex"
            elif "guarda pt" in lc: rename_map[c] = "Guarda PT"
            elif "guarda mmpp" in lc: rename_map[c] = "Guarda MMPP"

        # reducir al set que nos importa
        keep = [col_sku] + [c for c in df.columns if c in rename_map]
        df = df[keep].rename(columns=rename_map)

        # puede haber duplicados por “Total” ambiguo; colapsa dejando una sola por nombre final
        df = df.loc[:, ~df.columns.duplicated()]

        # reordenar
        df = df.rename(columns={col_sku: "SKU"})
        cols = ["SKU"] + [c for c in FINAL_ORDER if c in df.columns]
        df = df[cols]

    # limpiar SKU y convertir números
    df["SKU"] = df["SKU"].astype(str).str.strip()
    for c in df.columns:
        if c == "SKU": continue
        df[c] = df[c].apply(to_number_safe)

    # exportar a la misma planilla, hoja de salida
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        df.to_excel(w, index=False, sheet_name=SHEET_OUT)

    print(f"OK -> hoja '{SHEET_OUT}' creada/actualizada en:\n{XLSX_PATH}")
    print(f"Filas: {len(df)}  |  Columnas: {len(df.columns)}")

except Exception as e:
    raise