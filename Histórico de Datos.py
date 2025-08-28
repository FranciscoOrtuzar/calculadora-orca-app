# Inicio.py - Página Home
"""
Página principal de la aplicación de costos y márgenes.
Muestra la vista "Datos Históricos" con análisis de EBITDA.
"""

import streamlit as st
import io
from datetime import date
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_io import build_detalle, REQ_SHEETS, load_receta_sku, load_info_fruta, columns_config
from src.state import ensure_session_state, session_state_table
import pandas as pd
import numpy as np

# ===================== Config básica =====================
ST_TITLE = "Datos Históricos de Precios y Costos Octubre 2024 - Junio 2025 (MVP)"

# ===================== UI =====================
st.set_page_config(
    page_title="Calculadora de Costos",  # Título en la pestaña
    page_icon="📊",                      # Ícono de la pestaña (emoji o ruta a imagen)
    layout="wide"
)

# Inicializar estado de navegación
if "current_page" not in st.session_state:
    st.session_state.current_page = "Histórico"

# Inicializar y migrar todas las variables de session_state
ensure_session_state()

# Mostrar página Home
st.title(ST_TITLE)

# ===================== Carga de datos (con persistencia) =====================
with st.expander("📁 **Carga de archivo maestro (.xlsx)**"):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1) Subir archivo maestro (.xlsx)")
        # Verificar si ya hay datos en la sesión
        if "hist.uploaded_file" in st.session_state and st.session_state["hist.uploaded_file"] is not None:
            st.write(f"📁 Archivo: {st.session_state['hist.uploaded_file'].name}")
            
            if st.button("🔄 Recargar archivo"):
                reset_keys_by_type = {
                    "none": [
                        "hist.df",
                        "hist.df_filtered",
                        "hist.last_loaded_at",
                        "hist.skus_excluidos",
                        "hist.uploaded_file",
                        "hist.file_bytes",
                        "sim.df",
                        "sim.df_filtered",
                        "sim.override_upload",
                        "sim.last_saved_path",
                        "sim.last_saved_at",
                        "fruta.receta_df",
                        "fruta.info_df",
                        "export.last_saved_path",
                    ],
                    "dict": [
                        "hist.filters",
                        "sim.filters",
                        "sim.overrides_row",
                        "sim.fruit_overrides",
                    ],
                    "list": [
                        "sim.undo_stack",
                        "sim.redo_stack",
                        "ui.selected_rows",
                        "ui.messages",
                    ],
                    "other": {
                        "sim.override_pct_cost": 0.0,
                        "sim.dirty": False,
                        "ui.debug": False,
                        "ui.active_tab": "Histórico",
                        "ui.top_n": 10,
                    }
                }
                # Reinicia a None
                for key in reset_keys_by_type.get("none", []):
                    st.session_state[key] = None

                # Reinicia a diccionarios vacíos
                for key in reset_keys_by_type.get("dict", []):
                    st.session_state[key] = {}

                # Reinicia a listas vacías
                for key in reset_keys_by_type.get("list", []):
                    st.session_state[key] = []
                    
                # Reinicia a otros valores específicos
                for key, value in reset_keys_by_type.get("other", {}).items():
                    st.session_state[key] = value
                st.rerun()
        else:
            up = st.file_uploader("Selecciona tu Excel con hojas: " + ", ".join(REQ_SHEETS.keys()),
                                    type=["xlsx"], accept_multiple_files=False, key="file_uploader_home")
            
            if up is not None:
                # Guardar archivo en sesión
                st.session_state["hist.uploaded_file"] = up
                st.session_state["hist.file_bytes"] = up.read()
                st.rerun()
    with col2:
        st.subheader("2) Parámetros de precio vigente")
        modo = st.radio("Último precio por SKU", ["global","to_date"], horizontal=True, key="modo_home")
        ref_ym = None
        if modo == "to_date":
            # Selecciona una fecha (Año-Mes) para construir YYYYMM
            ref_date = st.date_input("Hasta fecha (se usa AñoMes)", value=date(2025,6,1), key="ref_date_home")
            ref_ym = ref_date.year*100 + ref_date.month
    
    st.caption("El archivo debe contener al menos: " + " | ".join([f"**{k}** ({v})" for k,v in REQ_SHEETS.items()]))

    st.markdown("---")
    st.caption("Consejo: si tus números vienen con coma decimal (3,071), este app los limpia automáticamente.")

# Procesar datos solo si no están en caché o si se recargó
if st.session_state["hist.df"] is None:
    if "hist.file_bytes" in st.session_state and st.session_state["hist.file_bytes"] is not None:
        try:
            with st.spinner("Procesando archivo..."):
                detalle = build_detalle(st.session_state["hist.file_bytes"], ultimo_precio_modo=modo, ref_ym=ref_ym)
                st.session_state["hist.df"] = detalle
                
                # Cargar datos de fruta si están disponibles
                try:
                    with st.spinner("Cargando datos de fruta..."):
                        # Leer el archivo Excel completo
                        from src.data_io import read_workbook
                        sheets = read_workbook(st.session_state["hist.file_bytes"])
                        
                        # Cargar RECETA_SKU si existe
                        if "RECETA_SKU" in sheets:
                            receta_df = load_receta_sku(sheets["RECETA_SKU"])
                            st.session_state["fruta.receta_df"] = receta_df
                            st.success(f"✅ RECETA_SKU cargada: {len(receta_df)} recetas")
                        else:
                            st.info("ℹ️ Hoja RECETA_SKU no encontrada")
                        
                        # Cargar INFO_FRUTA si existe
                        if "INFO_FRUTA" in sheets:
                            info_df = load_info_fruta(sheets["INFO_FRUTA"])
                            st.session_state["fruta.info_df"] = info_df
                            st.success(f"✅ INFO_FRUTA cargada: {len(info_df)} frutas")
                        else:
                            st.info("ℹ️ Hoja INFO_FRUTA no encontrada")
                            
                except Exception as e:
                    st.warning(f"⚠️ Error cargando datos de fruta: {e}")
                    st.info("💡 Los datos de fruta no son obligatorios para el simulador básico")
                    
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            st.stop()
    else:
        st.info("Sube tu archivo para comenzar.")
        st.stop()
else:
    detalle = st.session_state["hist.df"]
    

# Verificar que detalle esté definido antes de continuar
if 'detalle' not in locals() or detalle is None:
    st.error("❌ No hay datos disponibles para procesar")
    st.info("💡 Por favor, sube tu archivo Excel primero")
    st.stop()

# ===================== Sidebar - Filtros Dinámicos (igual a Simulación) =====================
st.sidebar.header("🔍 Filtros Dinámicos")

# Base de Históricos
df_base_hist = st.session_state["hist.df"].copy()

# Aliases igual que en Simulación (puedes ampliarlos si quieres)
FIELD_ALIASES = {
    "Marca": ["Marca", "Brand"],
    "Cliente": ["Cliente", "Customer", "Cliente ID", "ClienteID"],
    "Especie": ["Especie", "Species"],
    "Condicion": ["Condicion", "Condición", "Condition"],
    "SKU": ["SKU"],
}

def resolve_columns(df, aliases_map):
    resolved = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for logical, options in aliases_map.items():
        for opt in options:
            c = cols_lower.get(opt.lower())
            if c is not None:
                resolved[logical] = c
                break
    return resolved

RESOLVED_HIST = resolve_columns(df_base_hist, FIELD_ALIASES)
FILTER_FIELDS_HIST = [k for k in ["Marca","Cliente","Especie","Condicion","SKU"] if k in RESOLVED_HIST]

def _norm_series(s: pd.Series):
    return s.fillna("(Vacío)").astype(str).str.strip()

def _apply_filters_hist(df: pd.DataFrame, selections: dict, skip_key=None):
    out = df.copy()
    for logical, sel in selections.items():
        if logical == skip_key or not sel:
            continue
        real_col = RESOLVED_HIST[logical]
        valid = [x if x != "(Vacío)" else "" for x in sel]
        out = out[out[real_col].fillna("").astype(str).str.strip().isin(valid)]
    return out

def _current_selections_hist():
    selections = {}
    for logical in FILTER_FIELDS_HIST:
        selections[logical] = st.session_state.get(f"ms_hist_{logical}", [])
    return selections

# Guarda filtros en hist.filters
st.session_state["hist.filters"] = _current_selections_hist()

# Render de filtros en filas (sidebar)
if FILTER_FIELDS_HIST:
    SELECTIONS_HIST = _current_selections_hist()
    for logical in FILTER_FIELDS_HIST:
        real_col = RESOLVED_HIST[logical]
        df_except = _apply_filters_hist(df_base_hist, SELECTIONS_HIST, skip_key=logical)
        opts = sorted(_norm_series(df_except[real_col]).unique().tolist())
        current = [x for x in SELECTIONS_HIST.get(logical, []) if x in opts]
        st.sidebar.multiselect(logical, options=opts, default=current, key=f"ms_hist_{logical}")
else:
    st.sidebar.info("No hay campos disponibles para filtrar")

# Releer selecciones actualizadas y aplicar
SELECTIONS_HIST = _current_selections_hist()
df_filtrado = _apply_filters_hist(df_base_hist, SELECTIONS_HIST).copy()

# Orden y persistencia del filtrado
if "SKU-Cliente" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values(["SKU-Cliente"]).reset_index(drop=True)
else:
    df_filtrado = df_filtrado.reset_index(drop=True)

st.session_state["hist.df_filtered"] = df_filtrado.copy()

# -------- Filtrar subproductos (SKUs con costos totales = 0) --------
# Inicializar variable subproductos
subproductos = pd.DataFrame()
sin_ventas = pd.DataFrame()

# Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
if "Costos Totales (USD/kg)" in df_filtrado.columns:
    original_count = len(df_filtrado)
    subproductos = df_filtrado[(df_filtrado["Costos Totales (USD/kg)"] == 0) | (df_filtrado["Costos Totales (USD/kg)"] is None)].copy()
    sin_ventas = df_filtrado[df_filtrado["Comex"] == 0].copy()
    skus_excluidos = pd.concat([subproductos, sin_ventas])
    skus_excluidos = skus_excluidos.drop_duplicates(subset=["SKU-Cliente"], keep="first").set_index("SKU-Cliente")
    df_filtrado = df_filtrado[(df_filtrado["Costos Totales (USD/kg)"] != 0) & (df_filtrado["Comex"] != 0)].copy()
    df_filtrado["EBITDA Pct"] = df_filtrado["EBITDA Pct"] / 100
    
    filtered_count = len(df_filtrado)
    skus_excluidos_count = len(skus_excluidos)
    
    if original_count > filtered_count:    
        # Mostrar información sobre subproductos excluidos
        with st.expander(f"📋 **SKUs excluidos** ({skus_excluidos_count} SKUs)", expanded=False):
            st.write("**¿Por qué se excluyen estos SKUs?**")
            st.write("Son SKUs sin ventas, o con costos totales = 0, que no pueden generar EBITDA real y distorsionan el análisis financiero.")
            
            # Estadísticas de subproductos
            col1, col2, col3 = st.columns(3)
            with col1:
                if "Marca" in skus_excluidos.columns:
                    marca_counts = skus_excluidos["Marca"].value_counts()
                    st.write("**Por Marca:**")
                    for marca, count in marca_counts.head(3).items():
                        st.write(f"- {marca}: {count}")
            
            with col2:
                if "Cliente" in skus_excluidos.columns:
                    cliente_counts = skus_excluidos["Cliente"].value_counts()
                    st.write("**Por Cliente:**")
                    for cliente, count in cliente_counts.head(3).items():
                        st.write(f"- {cliente}: {count}")
            
            with col3:
                if "Especie" in skus_excluidos.columns:
                    especie_counts = skus_excluidos["Especie"].value_counts()
                    st.write("**Por Especie:**")
                    for especie, count in especie_counts.head(3).items():
                        st.write(f"- {especie}: {count}")
            
            # Tabla completa de subproductos
            st.write("**Lista completa de subproductos y SKUs sin ventas excluidos:**")
            st.dataframe(
                skus_excluidos[["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Comex", "Costos Totales (USD/kg)"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Botón de exportación
            csv_subproductos = skus_excluidos.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Lista Completa de Subproductos y SKUs sin ventas (CSV)",
                data=csv_subproductos,
                file_name="subproductos_sin_ventas_excluidos_completo.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_subproductos_sin_ventas_home"
            )

# -------- Mostrar resultados --------
st.subheader("Márgenes actuales (unitarios)")
base_cols = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion","Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","Proceso Granel (USD/kg)",
    "Almacenaje MMPP","Gastos Totales (USD/kg)","MMPP (Fruta) (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
view_base = detalle[base_cols].copy()
view_base.set_index("SKU-Cliente", inplace=True)
view_base = view_base.sort_index()
styled_view_base = view_base.style
config = columns_config(editable=False)

# Aplicar negritas a las columnas de totales
total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)",
"Costos Totales (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)"]
existing_total_columns = [col for col in total_columns if col in view_base.columns]

if existing_total_columns:
    styled_view_base = styled_view_base.set_properties(
        subset=existing_total_columns,
        **{"font-weight": "bold", "background-color": "#f8f9fa"}
    )

# Aplicar estilos a columnas EBITDA
ebitda_columns = ["EBITDA (USD/kg)", "EBITDA Pct"]
existing_ebitda_columns = [col for col in ebitda_columns if col in view_base.columns]

if existing_ebitda_columns:
    styled_view_base = styled_view_base.set_properties(
        subset=existing_ebitda_columns,
        **{"font-weight": "bold", "background-color": "#fff7ed"}
    )

st.dataframe(
    styled_view_base,
    use_container_width=True, 
    height="auto",
    column_config=config,
    hide_index=True
)

# --- Toggle: ver detalle de costos respetando los filtros vigentes ---
expand = st.toggle("🔎 Expandir costos por SKU (temporada)", value=False)

if expand:
    # 1) Toma los SKUs actualmente visibles (ya filtrados arriba)
    skus_filtrados = df_filtrado["SKU-Cliente"].astype(int).unique().tolist()
    det = detalle[detalle["SKU-Cliente"].astype(int).isin(skus_filtrados)].copy()
    # 3) Mueve atributos DIM a la izquierda
    dim_candidatas = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion"]
    dim_cols = [c for c in dim_candidatas if c in det.columns]
    orden_cols = ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)", "MMPP Total (USD/kg)","MO Directa",
                    "MO Indirecta","MO Total","Materiales Directos","Materiales Indirectos","Materiales Total",
                    "Laboratorio","Mantención","Servicios Generales","Utilities","Fletes Internos","Comex","Guarda PT",
                    "Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","Almacenaje MMPP",
                    "Gastos Totales (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
    # Si falta, recalcúlala si están los componentes
    if "Gastos Totales (USD/kg)" not in det.columns:
        comp = [
            "Retail Costos Directos (USD/kg)",
            "Retail Costos Indirectos (USD/kg)",
            "Almacenaje MMPP",
            "Proceso Granel (USD/kg)",
        ]
        if all(c in det.columns for c in comp):
            det["Gastos Totales (USD/kg)"] = sum(
                pd.to_numeric(det[c], errors="coerce") for c in comp
            )
    # Filtrar solo las columnas que realmente existen en el DataFrame
    last_cols = [c for c in orden_cols if c not in dim_cols and c in det.columns]
    det = det[dim_cols + last_cols]

    # 4) Orden y formato
    det = det.sort_values(["SKU-Cliente"]).reset_index(drop=True)
    view_base_det = det.copy()
    
    # Asegurar que el índice SKU-Cliente sea único antes de aplicar estilos
    view_base_det = view_base_det.drop_duplicates(subset=["SKU-Cliente"], keep="first")
    view_base_det.set_index("SKU-Cliente", inplace=True)
    
    # Aplicar estilos de formato y negritas a columnas importantes
    view_base_det = view_base_det.style
    
    # Aplicar negritas a las columnas de totales
    total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)", "Costos Totales (USD/kg)"]
    existing_total_columns = [col for col in total_columns if col in view_base_det.data.columns]
    
    if existing_total_columns:
        view_base_det = view_base_det.set_properties(
            subset=existing_total_columns,
            **{"font-weight": "bold", "background-color": "#f8f9fa"}
        )
    
    # Aplicar estilos a columnas EBITDA
    ebitda_columns = ["EBITDA (USD/kg)", "EBITDA Pct"]
    existing_ebitda_columns = [col for col in ebitda_columns if col in view_base_det.data.columns]
    
    if existing_ebitda_columns:
        view_base_det = view_base_det.set_properties(
            subset=existing_ebitda_columns,
            **{"font-weight": "bold", "background-color": "#fff7ed"}
        )

    # Aplicar formato numérico al Styler
    fmt_cols = {}
    for c in det.columns:
        if c not in (["SKU", "SKU-Cliente"] + dim_cols):
            if "Pct" in c or "Porcentaje" in c:
                fmt_cols[c] = "{:.1%}"  # Formato de porcentaje
            elif np.issubdtype(det[c].dtype, np.number):
                fmt_cols[c] = "{:.3f}"   # Formato numérico

    # Aplicar formato al Styler existente
    view_base_det = view_base_det.format(fmt_cols)

    st.subheader("Detalle de costos por SKU (temporada)")
    st.dataframe(
        view_base_det, 
        use_container_width=True, 
        height="auto",
        column_config=config,
        hide_index=True
    )

    # 5) Descargar
    def to_excel_download(df: pd.DataFrame, filename="export.xlsx"):
        # Asegura que las columnas SKU y SKU-Cliente estén presentes y al inicio
        cols = list(df.columns)
        for col in ["SKU", "SKU-Cliente"]:
            if col in cols:
                cols.remove(col)
        export_cols = [c for c in ["SKU", "SKU-Cliente"] if c in df.columns] + cols
        df_export = df[export_cols].copy()
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df_export.to_excel(xw, index=False, sheet_name="data")
        st.download_button("⬇️ Descargar Excel", data=buf.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_detalle")

    to_excel_download(det, "costos_detalle_temporada.xlsx")

    # Descargar versión resumida
    if "hist.costos_resumen" in st.session_state:
        to_excel_download(st.session_state["hist.costos_resumen"], "costos_resumen_temporada.xlsx")

# -------- KPIs y Resumen --------
st.subheader("📊 Resumen Ejecutivo")

# Calcular KPIs básicos
total_skus = len(df_filtrado)
skus_rentables = len(df_filtrado[df_filtrado["EBITDA (USD/kg)"] > 0])
ebitda_promedio = df_filtrado["EBITDA (USD/kg)"].mean()
margen_promedio = df_filtrado["EBITDA Pct"].mean()

# Mostrar KPIs en columnas
col1, col2 = st.columns([1,1])
# col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total SKUs", total_skus, help="SKUs con costos reales (excluyendo subproductos)")
    # Información sobre subproductos excluidos en los KPIs
    if len(subproductos) > 0:
        st.caption(f"⚠️ {len(skus_excluidos)} skus excluidos (costos o ventas = 0)")

with col2:
    st.metric("SKUs Rentables", skus_rentables, f"{skus_rentables/total_skus*100:.1f}%")

# with col3:
#     st.metric("EBITDA Promedio", f"${ebitda_promedio:.3f}/kg", help="EBITDA promedio no contiene subproductos")
# with col4:
#     st.metric("Margen Promedio", f"{margen_promedio:.1%}")

# # Resumen por marca si existe
# if "Marca" in df_filtrado.columns:
#     st.subheader("📈 EBITDA por Marca")
    
#     marca_summary = df_filtrado.groupby("Marca").agg({
#         "EBITDA (USD/kg)": ["mean", "count"],
#         "EBITDA Pct": "mean"
#     }).round(3)
#     marca_summary.columns = ["EBITDA Promedio (USD/kg)", "Cantidad SKUs", "EBITDA % Promedio"]
    
#     # Formato correcto para porcentajes
#     st.dataframe(
#         marca_summary.style.format({
#             "EBITDA Promedio (USD/kg)": "{:.3f}",
#             "Cantidad SKUs": "{:.0f}",
#             "EBITDA % Promedio": "{:.1%}"  # Formato de porcentaje
#         }),
#         use_container_width=True
#     )

# # Resumen por especie si existe
# if "Especie" in df_filtrado.columns:
#     st.subheader("🌱 EBITDA por Especie")
    
#     especie_summary = df_filtrado.groupby("Especie").agg({
#         "EBITDA (USD/kg)": ["mean", "count"],
#         "EBITDA Pct": "mean"
#     }).round(3)
#     especie_summary.columns = ["EBITDA Promedio (USD/kg)", "Cantidad SKUs", "EBITDA % Promedio"]
    
#     # Formato correcto para porcentajes
#     st.dataframe(
#         especie_summary.style.format({
#             "EBITDA Promedio (USD/kg)": "{:.3f}",
#             "Cantidad SKUs": "{:.0f}",
#             "EBITDA % Promedio": "{:.1%}"  # Formato de porcentaje
#         }),
#         use_container_width=True
#     )

# -------- Información de navegación --------
st.markdown("---")

# Expander opcional para diagnóstico de session_state
with st.expander("🔎 Diagnóstico session_state", expanded=False):
    session_state_table()

st.info("💡 **Navegación**: Usa el menú lateral para acceder al Simulador EBITDA y otras funcionalidades.")
st.info("💾 **Datos persistentes**: Los archivos cargados se mantienen al cambiar de página.")