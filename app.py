# app.py - Página Home
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

from data_io import build_mart, REQ_SHEETS, MESES_ORD, MES2NUM
import pandas as pd
import numpy as np

# ===================== Config básica =====================
ST_TITLE = "Datos Históricos de Precios y Costos Octubre 2024 - Junio 2025 (MVP)"

# ===================== Navegación =====================
def show_navigation():
    """Muestra la navegación entre páginas"""
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navegación")
    
    if st.sidebar.button("🏠 Home - Datos Históricos", type="primary"):
        st.session_state.current_page = "home"
        st.rerun()
    
    if st.sidebar.button("📊 Simulador EBITDA"):
        st.session_state.current_page = "simulator"
        st.rerun()

# ===================== UI =====================
st.set_page_config(page_title=ST_TITLE, layout="wide")

# Inicializar estado de navegación
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Si estamos en la página del simulador, mostrar esa página
if st.session_state.current_page == "simulator":
    # Importar y ejecutar la página del simulador
    import importlib.util
    simulator_path = Path(__file__).parent / "pages" / "1_Simulador_EBITDA.py"
    
    if simulator_path.exists():
        spec = importlib.util.spec_from_file_location("simulator", simulator_path)
        simulator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(simulator_module)
    else:
        st.error("No se pudo encontrar la página del simulador")
        st.session_state.current_page = "home"
        st.rerun()
else:
    # Mostrar página Home
    st.title(ST_TITLE)
    
    # Mostrar navegación
    show_navigation()
    
    # ===================== Carga de datos (con persistencia) =====================
    with st.sidebar:
        st.header("1) Subir archivo maestro (.xlsx)")
        
        # Verificar si ya hay datos en la sesión
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
            st.success("✅ Archivo ya cargado")
            st.write(f"📁 Archivo: {st.session_state.uploaded_file.name}")
            
            if st.button("🔄 Recargar archivo"):
                # Limpiar datos existentes
                for key in ["mart", "detalle", "uploaded_file", "file_bytes"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            up = st.file_uploader("Selecciona tu Excel con hojas: " + ", ".join(REQ_SHEETS.keys()),
                                  type=["xlsx"], accept_multiple_files=False, key="file_uploader_home")
            
            if up is not None:
                # Guardar archivo en sesión
                st.session_state.uploaded_file = up
                st.session_state.file_bytes = up.read()
                st.rerun()
        
        st.caption("El archivo debe contener al menos: " + " | ".join([f"**{k}** ({v})" for k,v in REQ_SHEETS.items()]))

        st.header("2) Parámetros de precio vigente")
        modo = st.radio("Último precio por SKU", ["global","to_date"], horizontal=True, key="modo_home")
        ref_ym = None
        if modo == "to_date":
            # Selecciona una fecha (Año-Mes) para construir YYYYMM
            ref_date = st.date_input("Hasta fecha (se usa AñoMes)", value=date(2025,6,1), key="ref_date_home")
            ref_ym = ref_date.year*100 + ref_date.month

        st.markdown("---")
        st.caption("Consejo: si tus números vienen con coma decimal (3,071), este app los limpia automáticamente.")

    # Procesar datos solo si no están en caché o si se recargó
    if "mart" not in st.session_state or "detalle" not in st.session_state:
        if "file_bytes" in st.session_state:
            try:
                with st.spinner("Procesando archivo..."):
                    mart, detalle = build_mart(st.session_state.file_bytes, ultimo_precio_modo=modo, ref_ym=ref_ym)
                    st.session_state.mart = mart
                    st.session_state.detalle = detalle
                st.success("✅ Archivo procesado exitosamente")
            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")
                st.stop()
        else:
            st.info("Sube tu archivo para comenzar.")
            st.stop()
    else:
        # Usar datos de la sesión
        mart = st.session_state.mart
        detalle = st.session_state.detalle

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

    def _current_selections():
        selections = {}
        for logical in FILTER_FIELDS:
            selections[logical] = st.session_state.get(f"ms_{logical}", [])
        return selections

    cols = st.columns(len(FILTER_FIELDS) if FILTER_FIELDS else 1)

    # Multiselects con opciones dependientes del resto, en cualquier orden
    SELECTIONS = _current_selections()
    for i, logical in enumerate(FILTER_FIELDS):
        with cols[i]:
            real_col = RESOLVED[logical]
            df_except = _apply_filters(mart, SELECTIONS, skip_key=logical)
            opts = sorted(_norm_series(df_except[real_col]).unique().tolist())
            current = [x for x in SELECTIONS.get(logical, []) if x in opts]
            st.multiselect(logical, options=opts, default=current, key=f"ms_{logical}")

    # Releer selecciones ya actualizadas por los widgets y aplicar
    SELECTIONS = _current_selections()
    df_filtrado = _apply_filters(mart, SELECTIONS).copy()

    # Orden por SKU si existe y sin índice
    sku_col = RESOLVED.get("SKU")
    if sku_col in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values([sku_col]).reset_index(drop=True)
    else:
        df_filtrado = df_filtrado.reset_index(drop=True)

    # -------- Mostrar resultados --------
    st.subheader("Márgenes actuales (unitarios)")
    base_cols = ["SKU","Descripcion","Marca","Cliente","Especie","Condicion","Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","MMPP (Proceso Granel) (USD/kg)",
     "Guarda MMPP","Gastos Totales (USD/kg)","MMPP (Fruta) (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
    view_base = df_filtrado[base_cols].copy()
    view_base.set_index("SKU", inplace=True)
    view_base = view_base.sort_index()

    # Aplicar formato correcto para columnas de porcentaje
    st.dataframe(
        view_base.style.format({
            "SKU":"{}", "Descripcion":"{}", "Marca":"{}", "Cliente":"{}", "Especie":"{}", "Condicion":"{}",
            "PrecioVenta (USD/kg)":"{:.3f}",
            "Retail Costos Directos (USD/kg)":"{:.3f}",
            "Retail Costos Indirectos (USD/kg)":"{:.3f}",
            "MMPP (Proceso Granel) (USD/kg)":"{:.3f}",
            "Guarda MMPP":"{:.3f}",
            "Gastos Totales (USD/kg)":"{:.3f}",
            "MMPP (Fruta) (USD/kg)":"{:.3f}",
            "Costos Totales (USD/kg)":"{:.3f}",
            "EBITDA (USD/kg)":"{:.3f}",
            "EBITDA Pct":"{:.1%}"  # Formato de porcentaje
        }),
        use_container_width=True, height=420
    )

    # --- Toggle: ver detalle de costos respetando los filtros vigentes ---
    expand = st.toggle("🔎 Expandir costos por SKU (temporada)", value=False)

    if expand:
        # 1) Toma los SKUs actualmente visibles (ya filtrados arriba)
        skus_filtrados = df_filtrado["SKU"].astype(str).unique().tolist()
        det = detalle[detalle["SKU"].astype(str).isin(skus_filtrados)].copy()
        print(det.columns)

        # 3) Mueve atributos DIM a la izquierda
        dim_candidatas = ["SKU","Descripcion","Marca","Cliente","Especie","Condicion"]
        dim_cols = [c for c in dim_candidatas if c in det.columns]
        orden_cols = ["MMPP (Fruta) (USD/kg)", "MMPP (Proceso Granel) (USD/kg)", "MMPP Total (USD/kg)","MO Directa",
                      "MO Indirecta","MO Total","Materiales Cajas y Bolsas","Materiales Indirectos","Materiales Total",
                      "Calidad","Matencion","Servicios Generales","Utilities","Fletes","Comex","Guarda PT","Guarda MMPP",
                      "Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","Gastos Totales (USD/kg)",
                      "Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
        # Si falta, recalcúlala si están los componentes
        if "Gastos Totales (USD/kg)" not in det.columns:
            comp = [
                "Retail Costos Directos (USD/kg)",
                "Retail Costos Indirectos (USD/kg)",
                "Guarda MMPP",
                "MMPP (Proceso Granel) (USD/kg)",
            ]
            if all(c in det.columns for c in comp):
                det["Gastos Totales (USD/kg)"] = sum(
                    pd.to_numeric(det[c], errors="coerce") for c in comp
                )
        last_cols = [c for c in orden_cols if c not in dim_cols]
        det = det[dim_cols + last_cols]

        # 4) Orden y formato
        det = det.sort_values(["SKU"]).reset_index(drop=True)
        view_base_det = det.copy()
        view_base_det.set_index("SKU", inplace=True)

        # Formato mejorado para columnas numéricas y de porcentaje
        fmt_cols = {}
        for c in det.columns:
            if c not in (["SKU"] + dim_cols):
                if "Pct" in c or "Porcentaje" in c:
                    fmt_cols[c] = "{:.1%}"  # Formato de porcentaje
                elif np.issubdtype(det[c].dtype, np.number):
                    fmt_cols[c] = "{:.3f}"   # Formato numérico

        st.subheader("Detalle de costos por SKU (temporada)")
        st.dataframe(view_base_det.style.format(fmt_cols), use_container_width=True, height=700)

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
            st.download_button("⬇️ Descargar Excel", data=buf.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        to_excel_download(det, "costos_detalle_temporada.xlsx")

        # Descargar versión resumida
        if "costos_resumen" in st.session_state:
            to_excel_download(st.session_state["costos_resumen"], "costos_resumen_temporada.xlsx")

    # -------- KPIs y Resumen --------
    st.subheader("📊 Resumen Ejecutivo")

    # Calcular KPIs básicos
    total_skus = len(df_filtrado)
    skus_rentables = len(df_filtrado[df_filtrado["EBITDA (USD/kg)"] > 0])
    ebitda_promedio = df_filtrado["EBITDA (USD/kg)"].mean()
    ebitda_total = df_filtrado["EBITDA (USD/kg)"].sum()

    # Mostrar KPIs en columnas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", total_skus)
    with col2:
        st.metric("SKUs Rentables", skus_rentables, f"{skus_rentables/total_skus*100:.1f}%")
    with col3:
        st.metric("EBITDA Promedio", f"${ebitda_promedio:.3f}/kg")
    with col4:
        st.metric("EBITDA Total", f"${ebitda_total:.0f}")

    # Resumen por marca si existe
    if "Marca" in df_filtrado.columns:
        st.subheader("📈 EBITDA por Marca")
        marca_summary = df_filtrado.groupby("Marca").agg({
            "EBITDA (USD/kg)": ["mean", "count"],
            "EBITDA Pct": "mean"
        }).round(3)
        marca_summary.columns = ["EBITDA Promedio (USD/kg)", "Cantidad SKUs", "EBITDA % Promedio"]
        
        # Formato correcto para porcentajes
        st.dataframe(
            marca_summary.style.format({
                "EBITDA Promedio (USD/kg)": "{:.3f}",
                "Cantidad SKUs": "{:.0f}",
                "EBITDA % Promedio": "{:.1%}"  # Formato de porcentaje
            }),
            use_container_width=True
        )

    # Resumen por especie si existe
    if "Especie" in df_filtrado.columns:
        st.subheader("🌱 EBITDA por Especie")
        especie_summary = df_filtrado.groupby("Especie").agg({
            "EBITDA (USD/kg)": ["mean", "count"],
            "EBITDA Pct": "mean"
        }).round(3)
        especie_summary.columns = ["EBITDA Promedio (USD/kg)", "Cantidad SKUs", "EBITDA % Promedio"]
        
        # Formato correcto para porcentajes
        st.dataframe(
            especie_summary.style.format({
                "EBITDA Promedio (USD/kg)": "{:.3f}",
                "Cantidad SKUs": "{:.0f}",
                "EBITDA % Promedio": "{:.1%}"  # Formato de porcentaje
            }),
            use_container_width=True
        )

    # -------- Información de navegación --------
    st.markdown("---")
    st.info("💡 **Navegación**: Usa el menú lateral para acceder al Simulador EBITDA y otras funcionalidades.")
    st.info("💾 **Datos persistentes**: Los archivos cargados se mantienen al cambiar de página.")