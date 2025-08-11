import os
import io
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st

# =============================
# 🧭 Configuración básica
# =============================
st.set_page_config(page_title="Herramienta de Pricing", page_icon="💸", layout="wide")

# ---------- Utilidades ----------
@st.cache_data
def read_uploaded_file(file) -> dict:
    # Acepta .xlsx (hojas Ingredients, Recipes) o .csv sueltos
    if file.name.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        sheets = {}
        for sheet in xls.sheet_names:
            sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
        return sheets
    elif file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        return {"Data": df}
    else:
        raise ValueError("Formato no soportado. Usa .xlsx o .csv")

@st.cache_data
def sample_data():
    # Datos de ejemplo mínimos para correr sin archivos externos
    ingredients = pd.DataFrame(
        [
            {"ingredient_id": "ING-01", "name": "Filete Pescado A", "unit": "kg", "unit_cost": 3.20, "yield_pct": 0.92,
             "packaging_cost": 0.20, "qc_cost": 0.05},
            {"ingredient_id": "ING-02", "name": "Rebozador", "unit": "kg", "unit_cost": 0.80, "yield_pct": 0.98,
             "packaging_cost": 0.03, "qc_cost": 0.01},
            {"ingredient_id": "ING-03", "name": "Aceite", "unit": "L", "unit_cost": 1.10, "yield_pct": 0.99,
             "packaging_cost": 0.00, "qc_cost": 0.00},
        ]
    )
    recipes = pd.DataFrame(
        [
            {"product_code": "PRD-001", "product_name": "Merluza Apanada 1kg", "ingredient_id": "ING-01", "qty": 0.70},
            {"product_code": "PRD-001", "product_name": "Merluza Apanada 1kg", "ingredient_id": "ING-02", "qty": 0.25},
            {"product_code": "PRD-001", "product_name": "Merluza Apanada 1kg", "ingredient_id": "ING-03", "qty": 0.05},
        ]
    )
    # Costos globales (conceptos no específicos al ingrediente)
    globals_df = pd.DataFrame([
        {"concept": "labor_processing_per_kg", "value": 0.35},
        {"concept": "overhead_direct_per_kg", "value": 0.18},
        {"concept": "overhead_indirect_alloc_per_kg", "value": 0.22},
        {"concept": "freight_to_port_per_kg", "value": 0.25},
        {"concept": "export_fees_per_kg", "value": 0.07},
        {"concept": "insurance_financing_pct", "value": 0.012},
        {"concept": "commission_rebates_pct", "value": 0.02},
        {"concept": "fx_rate_usd_clp", "value": 950.0},
        {"concept": "wastage_extra_pct", "value": 0.01},
    ])
    return ingredients, recipes, globals_df

# ---------- Estado ----------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

# =============================
# 🧱 Sidebar: carga y parámetros
# =============================
st.sidebar.header("Configuración & Datos")

mode = st.sidebar.radio(
    "Fuente de datos",
    ["Datos de ejemplo", "Subir archivo (.xlsx/.csv)"]
)

if mode == "Subir archivo (.xlsx/.csv)":
    upl = st.sidebar.file_uploader("Cargar archivo de datos", type=["xlsx", "csv"], accept_multiple_files=False)
    if upl:
        try:
            sheets = read_uploaded_file(upl)
            # Intento estándar: Ingredients y Recipes
            ingredients = sheets.get("Ingredients") or sheets.get("ingredients")
            recipes = sheets.get("Recipes") or sheets.get("recipes")
            globals_df = sheets.get("Globals") or sheets.get("globals")
            if ingredients is None or recipes is None:
                st.sidebar.warning("No se encontraron hojas 'Ingredients' y 'Recipes'. Usando datos de ejemplo.")
                ingredients, recipes, globals_df = sample_data()
            elif globals_df is None:
                st.sidebar.info("No se encontró hoja 'Globals'. Se usarán valores por defecto.")
                _, _, default_globals = sample_data()
                globals_df = default_globals
        except Exception as e:
            st.sidebar.error(f"Error leyendo archivo: {e}")
            ingredients, recipes, globals_df = sample_data()
    else:
        ingredients, recipes, globals_df = sample_data()
else:
    ingredients, recipes, globals_df = sample_data()

st.sidebar.subheader("Parámetros globales")
fx_rate = st.sidebar.number_input("Tipo de cambio (CLP por USD)", min_value=1.0, value=float(globals_df.loc[globals_df["concept"]=="fx_rate_usd_clp", "value"].iloc[0]), step=1.0)
insurance_fin_pct = st.sidebar.number_input("% Seguro/Financiamiento (sobre costo)", min_value=0.0, value=float(globals_df.loc[globals_df["concept"]=="insurance_financing_pct", "value"].iloc[0]), step=0.001, format="%.3f")
commission_pct = st.sidebar.number_input("% Comisiones/Rebates (sobre precio)", min_value=0.0, value=float(globals_df.loc[globals_df["concept"]=="commission_rebates_pct", "value"].iloc[0]), step=0.001, format="%.3f")
wastage_pct = st.sidebar.number_input("% Merma adicional (sobre insumos)", min_value=0.0, value=float(globals_df.loc[globals_df["concept"]=="wastage_extra_pct", "value"].iloc[0]), step=0.001, format="%.3f")

# =============================
# 🧪 Selección de producto / receta
# =============================
st.title("💸 Herramienta de Pricing — MVP Streamlit")
st.caption("Calcula precio sugerido a partir de costos (11 conceptos), margen y metas de ingreso.")

left, right = st.columns([1, 1])

# Lista de productos a partir de recipes
products = recipes[["product_code", "product_name"]].drop_duplicates().sort_values("product_code")
with left:
    st.subheader("1) Selección de producto")
    prod = st.selectbox(
        "Producto",
        options=[f"{r.product_code} — {r.product_name}" for _, r in products.iterrows()]
    )
    selected_code = prod.split(" — ")[0]

    st.markdown("**Composición (puedes editar cantidades):**")
    base_recipe = recipes[recipes["product_code"] == selected_code].merge(ingredients, on="ingredient_id", how="left")
    base_recipe = base_recipe[["ingredient_id", "name", "qty", "unit", "unit_cost", "yield_pct", "packaging_cost", "qc_cost"]].copy()

    edit_df = st.data_editor(
        base_recipe,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "qty": st.column_config.NumberColumn("Cantidad (unidad receta)", min_value=0.0, step=0.01),
            "unit_cost": st.column_config.NumberColumn("Costo unitario (USD/Unidad)", min_value=0.0, step=0.01),
            "yield_pct": st.column_config.NumberColumn("Rendimiento (0-1)", min_value=0.0, max_value=1.0, step=0.01),
            "packaging_cost": st.column_config.NumberColumn("Empaque (USD/kg)", min_value=0.0, step=0.01),
            "qc_cost": st.column_config.NumberColumn("Calidad/Cert (USD/kg)", min_value=0.0, step=0.01),
        },
        key="editor_recipe",
    )

with right:
    st.subheader("2) Parámetros comerciales")
    pricing_mode = st.radio(
        "Modo de pricing",
        ["Objetivo Margen (%)", "Precio objetivo (USD)"]
    )
    margin_pct = None
    target_price = None

    if pricing_mode == "Objetivo Margen (%)":
        margin_pct = st.number_input("Margen objetivo (%)", min_value=0.0, max_value=95.0, value=20.0, step=0.5)
    else:
        target_price = st.number_input("Precio de venta objetivo (USD)", min_value=0.0, value=5.00, step=0.05)

    st.divider()
    st.subheader("3) Volumen y unidad")
    units = st.number_input("Volumen (unidades)", min_value=1, value=1000, step=100)
    unit_label = st.text_input("Unidad de venta (ej: bolsa 1kg, caja 10x1kg)", value="bolsa 1kg")

# =============================
# 🧮 Cálculo de costos (11 conceptos)
# =============================
# Definición de 11 conceptos de costo (en USD por unidad de venta):
# 1) Materias primas (ajustadas por rendimiento + merma)
# 2) Mano de obra de proceso
# 3) Empaque
# 4) Control de calidad / certificaciones
# 5) Overhead directo (energía/agua/mantenimiento)
# 6) Overhead indirecto asignado
# 7) Flete interno a puerto
# 8) Tasas/fees de exportación
# 9) Seguro/financiamiento (pct sobre costo)
# 10) Merma adicional (pct sobre MP)
# 11) Comisiones/Rebates (pct sobre precio) — se aplica después sobre el precio sugerido

# Tomamos parámetros globales por kg (o por unidad de venta equivalente). En un MVP, usamos valores fijos.
# Puedes mover estos valores a la hoja Globals del Excel.
_g = {c: v for c, v in zip(globals_df["concept"], globals_df["value"])}

labor_processing = _g.get("labor_processing_per_kg", 0.35)
direct_overhead = _g.get("overhead_direct_per_kg", 0.18)
indirect_overhead = _g.get("overhead_indirect_alloc_per_kg", 0.22)
freight_to_port = _g.get("freight_to_port_per_kg", 0.25)
export_fees = _g.get("export_fees_per_kg", 0.07)

# Materias primas: sumatoria qty * unit_cost, ajustando por rendimiento
def compute_raw_material_cost(df: pd.DataFrame) -> float:
    df = df.copy()
    # Ajustamos por rendimiento efectivo (si rinde 0.92, el costo efectivo sube 1/0.92)
    df["effective_cost"] = df.apply(lambda r: (r["qty"] * r["unit_cost"]) / max(r["yield_pct"], 1e-9), axis=1)
    raw_cost = df["effective_cost"].sum()
    return float(raw_cost)

raw_material_cost = compute_raw_material_cost(edit_df)
extra_wastage = raw_material_cost * wastage_pct  # (10) Merma adicional

packaging_cost = float((edit_df["packaging_cost"] * edit_df["qty"]).sum())  # (3)
qc_cost = float((edit_df["qc_cost"] * edit_df["qty"]).sum())  # (4)

# Costos planos por unidad de venta
labor_cost = labor_processing  # (2)
direct_oh_cost = direct_overhead  # (5)
indirect_oh_cost = indirect_overhead  # (6)
freight_cost = freight_to_port  # (7)
export_cost = export_fees  # (8)

# Costo base antes de seguro/financiamiento y comisiones
cost_before_fin = (
    raw_material_cost + extra_wastage + packaging_cost + qc_cost +
    labor_cost + direct_oh_cost + indirect_oh_cost + freight_cost + export_cost
)

insurance_fin_cost = cost_before_fin * insurance_fin_pct  # (9)

# Costo total por unidad de venta (USD)
unit_total_cost_usd = cost_before_fin + insurance_fin_cost

# Precio sugerido según modo
if pricing_mode == "Objetivo Margen (%)":
    # Precio = Costo / (1 - margen - comisiones)
    # Primero estimamos precio sin comisión para calcular comisión sobre precio final
    # Iteramos una vez: P = C / (1 - m - c)
    m = (margin_pct or 0.0) / 100.0
    c = commission_pct
    if m + c >= 0.95:
        st.warning("La suma margen+comisión es muy alta. Ajusta parámetros.")
    suggested_price_usd = unit_total_cost_usd / max(1e-9, (1.0 - m - c))
    achieved_margin_pct = (1 - unit_total_cost_usd / max(1e-9, suggested_price_usd)) * 100.0
else:
    suggested_price_usd = target_price or 0.0
    achieved_margin_pct = (1 - unit_total_cost_usd / max(1e-9, suggested_price_usd)) * 100.0

commission_value = suggested_price_usd * commission_pct  # (11)

# Totales por volumen
total_cost_batch_usd = unit_total_cost_usd * units
total_revenue_batch_usd = suggested_price_usd * units

# =============================
# 📊 Salidas
# =============================
st.divider()
st.subheader("Resultados")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo unitario (USD)", f"{unit_total_cost_usd:,.2f}")
col2.metric("Precio sugerido (USD)", f"{suggested_price_usd:,.2f}")
col3.metric("Margen logrado (%)", f"{achieved_margin_pct:,.1f}")
col4.metric("Comisión (USD/unidad)", f"{commission_value:,.2f}")

# Tabla de desglose (11 conceptos)
concept_rows = [
    ("1) Materias primas (ajust. rendimiento)", raw_material_cost),
    ("2) Mano de obra de proceso", labor_cost),
    ("3) Empaque", packaging_cost),
    ("4) Calidad/Certificaciones", qc_cost),
    ("5) Overhead directo", direct_oh_cost),
    ("6) Overhead indirecto", indirect_oh_cost),
    ("7) Flete interno a puerto", freight_cost),
    ("8) Tasas/fees de exportación", export_cost),
    ("9) Seguro/financiamiento", insurance_fin_cost),
    ("10) Merma adicional", extra_wastage),
]

breakdown = pd.DataFrame(concept_rows, columns=["Concepto", "USD por unidad"]).assign(
    **{"% del costo": lambda d: 100 * d["USD por unidad"] / max(1e-9, unit_total_cost_usd)}
)

st.markdown("**Desglose de costo (USD/unidad):**")
st.dataframe(breakdown, use_container_width=True, hide_index=True)

st.markdown("**Totales del lote:**")
colA, colB = st.columns(2)
with colA:
    st.write(f"Costo total (USD): **{total_cost_batch_usd:,.0f}**")
    st.write(f"Ingresos estimados (USD): **{total_revenue_batch_usd:,.0f}**")
with colB:
    st.write(f"Tipo de cambio (CLP/USD): **{fx_rate:,.0f}**")
    st.write(f"Costo unitario (CLP): **{unit_total_cost_usd * fx_rate:,.0f}**")
    st.write(f"Precio sugerido (CLP): **{suggested_price_usd * fx_rate:,.0f}**")

# =============================
# 💾 Exportar / escenarios
# =============================
st.divider()
st.subheader("Exportar & Escenarios")

scenario_name = st.text_input("Nombre del escenario", value=f"Escenario {datetime.now().strftime('%Y-%m-%d %H:%M')}" )
if st.button("💾 Guardar escenario en memoria"):
    st.session_state.scenarios[scenario_name] = {
        "product": prod,
        "units": units,
        "unit_label": unit_label,
        "pricing_mode": pricing_mode,
        "margin_pct": margin_pct,
        "target_price": target_price,
        "fx_rate": fx_rate,
        "insurance_fin_pct": insurance_fin_pct,
        "commission_pct": commission_pct,
        "wastage_pct": wastage_pct,
        "breakdown": breakdown.to_dict(orient="records"),
        "unit_cost_usd": unit_total_cost_usd,
        "price_usd": suggested_price_usd,
        "achieved_margin_pct": achieved_margin_pct,
    }
    st.success("Escenario guardado en sesión.")

if st.session_state.scenarios:
    st.write("Escenarios guardados (sesión actual):")
    st.json(st.session_state.scenarios)

# Descargar reporte simple en CSV
out_buf = io.StringIO()
export_df = breakdown.copy()
export_df.loc[len(export_df)] = ["Precio sugerido (USD)", suggested_price_usd, None]
export_df.loc[len(export_df)] = ["Costo total por unidad (USD)", unit_total_cost_usd, None]
export_df.to_csv(out_buf, index=False)
st.download_button(
    label="⬇️ Descargar desglose CSV",
    data=out_buf.getvalue(),
    file_name=f"pricing_breakdown_{selected_code}.csv",
    mime="text/csv"
)

st.caption("MVP: parámetros globales y fórmulas simplificadas. En producción, mover a hoja 'Globals' y versiones por país/cliente.")
