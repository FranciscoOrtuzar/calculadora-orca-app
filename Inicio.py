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

from src.data_io import build_detalle, REQ_SHEETS, MESES_ORD, MES2NUM
import pandas as pd
import numpy as np

# ===================== Config básica =====================
ST_TITLE = "Datos Históricos de Precios y Costos Octubre 2024 - Junio 2025 (MVP)"

# ===================== Navegación =====================
# def show_navigation():
#     """Muestra la navegación entre páginas"""
#     st.sidebar.markdown("---")
#     st.sidebar.header("🧭 Navegación")
    
#     if st.sidebar.button("🏠 Home - Datos Históricos", type="primary"):
#         st.session_state.current_page = "home"
#         st.rerun()
    
#     if st.sidebar.button("📊 Simulador EBITDA"):
#         st.session_state.current_page = "simulator"
#         st.rerun()

# ===================== UI =====================
st.set_page_config(
    page_title="Calculadora de Costos",  # Título en la pestaña
    page_icon="📊",                      # Ícono de la pestaña (emoji o ruta a imagen)
    layout="wide"
)
# CSS para estilos de tabla mejorados
st.markdown("""
<style>
/* Estilos base para encabezados de tabla */
.stDataFrame th {
    font-weight: bold !important;
    text-align: center !important;
    border: 1px solid #d1d5db !important;
    background-color: #f3f4f6 !important;
    color: #374151 !important;
}

/* Estilos específicos para columnas de totales - negritas */
.stDataFrame td {
    border: 1px solid #e5e7eb !important;
    padding: 8px !important;
    background-color: white !important;
}

/* Hacer que las columnas de totales estén en negritas */
.stDataFrame tbody tr td:nth-child(n) {
    font-weight: normal;
}

/* Resaltar filas al pasar el mouse */
.stDataFrame tbody tr:hover td {
    background-color: #f9fafb !important;
}
</style>
""", unsafe_allow_html=True)



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
    # ===================== TÍTULO PRINCIPAL =====================
    st.title("📊 Calculadora de Costos y Márgenes")
    st.markdown("### Análisis de rentabilidad por SKU con sistema integrado de recetas")
    st.markdown("---")
    
    # Mostrar navegación
    # show_navigation()
    
    # ===================== CARGA DE ARCHIVOS =====================
    st.sidebar.header("📁 Carga de Archivos")
    st.sidebar.info("Sube tu archivo Excel con los datos de costos, precios y recetas para comenzar el análisis.")
    uploaded_file = st.sidebar.file_uploader("Seleccionar archivo Excel", type=["xlsx", "xls"], help="El archivo debe contener las hojas: FACT_COSTOS_POND, FACT_PRECIOS, DIM_SKU, RECETA_SKU, INFO_FRUTA)")
    if uploaded_file is not None:
        st.success("✅ Archivo seleccionado")
        st.sidebar.info(f"📄 **{uploaded_file.name}**")
        st.sidebar.info(f"📏 **{uploaded_file.size / 1024:.1f} KB**")
    else:
        st.sidebar.info("📤 Selecciona un archivo para comenzar")

    
    
    # Procesar archivo cuando se sube
    if uploaded_file is not None:
        # Guardar en sesión
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_bytes = uploaded_file.getvalue()
        
        # Configuración de parámetros
        st.subheader("⚙️ Configuración de Análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            modo = st.selectbox(
                "Modo de cálculo de último precio:",
                options=["global", "to_date"],
                help="Global: último precio disponible. To_date: precio hasta fecha específica"
            )
        
        with col2:
            if modo == "to_date":
                ref_ym = st.number_input(
                    "Año-Mes de referencia (YYYYMM):",
                    min_value=202001,
                    max_value=203012,
                    value=202412,
                    help="Formato: 202412 para Diciembre 2024"
                )
            else:
                ref_ym = None
        
        # Procesar archivo
        if st.button("🚀 Procesar Archivo", type="primary"):
            try:
                with st.spinner("Procesando archivo..."):
                    from src.data_io import read_workbook
                    sheets = read_workbook(st.session_state.file_bytes)
                    
                    # Verificar validación de hojas
                    from src.data_io import validate_required_sheets
                    missing_sheets = validate_required_sheets(sheets)
                    
                    if missing_sheets:
                        st.error(f"❌ **Hojas obligatorias faltantes**: {missing_sheets}")
                        st.stop()
                    
                    # Procesar el archivo
                    detalle = build_detalle(st.session_state.file_bytes, ultimo_precio_modo=modo, ref_ym=ref_ym)
                    st.session_state.detalle = detalle
                    
                    st.success("✅ **Archivo procesado exitosamente**")
                    st.info(f"📊 **Datos cargados**: {len(detalle)} SKUs procesados")
                    
                    # Mostrar resumen básico de datos
                    st.subheader("📊 Resumen de Datos Cargados")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total SKUs", len(detalle))
                    
                    with col2:
                        if "Marca" in detalle.columns:
                            marcas_unicas = detalle["Marca"].nunique()
                            st.metric("Marcas", marcas_unicas)
                        else:
                            st.metric("Marcas", "N/A")
                    
                    with col3:
                        if "Cliente" in detalle.columns:
                            clientes_unicos = detalle["Cliente"].nunique()
                            st.metric("Clientes", clientes_unicos)
                        else:
                            st.metric("Clientes", "N/A")
                    
                    # Mostrar primeras filas como preview
                    with st.expander("👀 Ver Preview de Datos", expanded=False):
                        st.dataframe(detalle.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ **Error procesando archivo**: {e}")
                st.info("💡 **Solución**: Verifica que el archivo tenga el formato correcto y las hojas requeridas")
    
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
    if "detalle" not in st.session_state:
        if "file_bytes" in st.session_state:
            try:
                with st.spinner("Procesando archivo..."):
                    from src.data_io import read_workbook
                    sheets = read_workbook(st.session_state.file_bytes)
                    
                    # Verificar validación de hojas
                    from src.data_io import validate_required_sheets
                    missing_sheets = validate_required_sheets(sheets)
                    
                    if missing_sheets:
                        st.error(f"❌ **Hojas obligatorias faltantes**: {missing_sheets}")
                        st.stop()
                    
                    # Procesar el archivo
                    detalle = build_detalle(st.session_state.file_bytes, ultimo_precio_modo=modo, ref_ym=ref_ym)
                    st.session_state.detalle = detalle
                    
                    st.success("✅ **Archivo procesado exitosamente**")
                    st.info(f"📊 **Datos cargados**: {len(detalle)} SKUs procesados")
                    
                    # Mostrar resumen de datos
                    show_data_summary(detalle)
                        
            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")
                st.stop()
        else:
            st.info("Sube tu archivo para comenzar.")
            st.stop()
    else:
        # Usar datos de la sesión y aplicar renombrado si es necesario
        if st.session_state.detalle is not None:
            detalle = st.session_state.detalle.copy()
            
            # Forzar renombrado de columnas para que coincidan con los nombres deseados
            column_rename_map = {
                "Calidad": "Laboratorio",
                "Matencion": "Mantención", 
                "Fletes": "Fletes Internos"
            }
            
            # Aplicar renombrado solo si las columnas existen
            for old_name, new_name in column_rename_map.items():
                if old_name in detalle.columns:
                    detalle = detalle.rename(columns={old_name: new_name})
            
            # Actualizar la sesión con los nombres corregidos
            st.session_state.detalle = detalle
            
            # Mostrar mensaje informativo sobre el renombrado
            if any(old_name in st.session_state.detalle.columns for old_name in ["Calidad", "Matencion", "Fletes"]):
                st.info("✅ **Columnas actualizadas**: Se aplicaron los nombres correctos (Laboratorio, Mantención, Fletes Internos)")
        else:
            st.warning("⚠️ Los datos de la sesión están vacíos o corruptos")
            st.info("💡 Por favor, sube tu archivo nuevamente")
            st.stop()

    # Verificar que detalle esté definido antes de continuar
    if 'detalle' not in locals() or detalle is None:
        st.error("❌ No hay datos disponibles para procesar")
    
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

    # Resolver alias -> columna real presente en detalle
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

    RESOLVED = resolve_columns(detalle, FIELD_ALIASES)

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
            df_except = _apply_filters(detalle, SELECTIONS, skip_key=logical)
            opts = sorted(_norm_series(df_except[real_col]).unique().tolist())
            current = [x for x in SELECTIONS.get(logical, []) if x in opts]
            st.multiselect(logical, options=opts, default=current, key=f"ms_{logical}")

    # Releer selecciones ya actualizadas por los widgets y aplicar
    SELECTIONS = _current_selections()
    df_filtrado = _apply_filters(detalle, SELECTIONS).copy()
    df_filtrado["EBITDA Pct"] = df_filtrado["EBITDA Pct"] / 100

    # Orden por SKU-Cliente si existe y sin índice
    sku_cliente_col = "SKU-Cliente"
    if sku_cliente_col in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values([sku_cliente_col]).reset_index(drop=True)
    else:
        df_filtrado = df_filtrado.reset_index(drop=True)

    # -------- Filtrar subproductos (SKUs con costos totales = 0) --------
    # Inicializar variable subproductos
    subproductos = pd.DataFrame()
    
    # Separar SKUs con costos totales = 0 (subproductos) de los que tienen costos reales
    if "Costos Totales (USD/kg)" in df_filtrado.columns:
        original_count = len(df_filtrado)
        subproductos = df_filtrado[df_filtrado["Costos Totales (USD/kg)"] == 0].copy()
        df_filtrado = df_filtrado[df_filtrado["Costos Totales (USD/kg)"] != 0].copy()
        
        filtered_count = len(df_filtrado)
        subproductos_count = len(subproductos)
        
        if original_count > filtered_count:    
            # Mostrar información sobre subproductos excluidos
            with st.expander(f"📋 **Subproductos excluidos** ({subproductos_count} SKUs)", expanded=False):
                st.write("**¿Por qué se excluyen estos SKUs?**")
                st.write("Los SKUs con costos totales = 0 no pueden generar EBITDA real y distorsionan el análisis financiero.")
                
                # Estadísticas de subproductos
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "Marca" in subproductos.columns:
                        marca_counts = subproductos["Marca"].value_counts()
                        st.write("**Por Marca:**")
                        for marca, count in marca_counts.head(3).items():
                            st.write(f"- {marca}: {count}")
                
                with col2:
                    if "Cliente" in subproductos.columns:
                        cliente_counts = subproductos["Cliente"].value_counts()
                        st.write("**Por Cliente:**")
                        for cliente, count in cliente_counts.head(3).items():
                            st.write(f"- {cliente}: {count}")
                
                with col3:
                    if "Especie" in subproductos.columns:
                        especie_counts = subproductos["Especie"].value_counts()
                        st.write("**Por Especie:**")
                        for especie, count in especie_counts.head(3).items():
                            st.write(f"- {especie}: {count}")
                
                # Tabla completa de subproductos
                st.write("**Lista completa de subproductos excluidos:**")
                st.dataframe(
                    subproductos[["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Costos Totales (USD/kg)"]],
                    use_container_width=True
                )
                
                # Botón de exportación
                csv_subproductos = subproductos.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Lista Completa de Subproductos (CSV)",
                    data=csv_subproductos,
                    file_name="subproductos_excluidos_completo.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_subproductos_home"
                )

    # -------- Mostrar resultados --------
    st.subheader("Márgenes actuales (unitarios)")
    
    # Mostrar información del sistema de recetas si está disponible
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
        fruit_prices = recipe_data.get("fruit_prices", {})
        
        if not recipe_df.empty and fruit_prices:
            # Contar frutas totales vs frutas con precios
            total_frutas = len(fruit_prices)
            frutas_con_precio = len([p for p in fruit_prices.values() if p > 0])
            
            st.success(f"🍎 **Sistema de Recetas Activo**: {len(recipe_df)} SKUs con recetas, {total_frutas} frutas base disponibles")
            st.info("💡 **MMPP (Fruta) se calcula automáticamente** desde recetas usando precios y eficiencias de INFO_FRUTA")
            
            if frutas_con_precio < total_frutas:
                st.warning(f"⚠️ **Atención**: Solo {frutas_con_precio} de {total_frutas} frutas tienen precios definidos. Las demás usarán precio 0.")
        else:
            st.warning("⚠️ **Sistema de Recetas**: Datos incompletos - verificar INFO_FRUTA y RECETA_SKU")
    else:
        st.info("ℹ️ **Sistema de Recetas**: No disponible - cargar archivo con hojas INFO_FRUTA y RECETA_SKU para cálculo automático de MMPP")
    
    base_cols = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion","Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","Proceso Granel (USD/kg)",
     "Guarda MMPP","Gastos Totales (USD/kg)","MMPP (Fruta) (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
    view_base = df_filtrado[base_cols].copy()
    view_base.set_index("SKU-Cliente", inplace=True)
    view_base = view_base.sort_index()

    # Aplicar formato y estilos específicos
    styled_view_base = view_base.style.format({
        "SKU":"{}", "Descripcion":"{}", "Marca":"{}", "Cliente":"{}", "Especie":"{}", "Condicion":"{}",
        "PrecioVenta (USD/kg)":"{:.3f}",
        "Retail Costos Directos (USD/kg)":"{:.3f}",
        "Retail Costos Indirectos (USD/kg)":"{:.3f}",
        "Proceso Granel (USD/kg)":"{:.3f}",
        "Guarda MMPP":"{:.3f}",
        "Gastos Totales (USD/kg)":"{:.3f}",
        "MMPP (Fruta) (USD/kg)":"{:.3f}",
        "Costos Totales (USD/kg)":"{:.3f}",
        "EBITDA (USD/kg)":"{:.3f}",
        "EBITDA Pct":"{:.1%}"  # Formato de porcentaje
    })
    
    # Aplicar negritas a las columnas de totales
    total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)", "Costos Totales (USD/kg)"]
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
        height=420,
        column_config={
            "SKU": st.column_config.TextColumn("SKU", pinned="left"),
            "Descripcion": st.column_config.TextColumn("Descripción", pinned="left"),
        },
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
                      "MO Indirecta","MO Total","Materiales Cajas y Bolsas","Materiales Indirectos","Materiales Total",
                      "Laboratorio","Mantención","Servicios Generales","Utilities","Fletes Internos","Comex","Guarda PT","Guarda MMPP",
                      "Retail Costos Directos (USD/kg)","Retail Costos Indirectos (USD/kg)","Gastos Totales (USD/kg)",
                      "Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)","EBITDA Pct"]
        # Si falta, recalcúlala si están los componentes
        if "Gastos Totales (USD/kg)" not in det.columns:
            comp = [
                "Retail Costos Directos (USD/kg)",
                "Retail Costos Indirectos (USD/kg)",
                "Guarda MMPP",
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
        
        # Agregar información sobre origen del MMPP (Fruta) si hay recetas disponibles
        if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
            recipe_data = st.session_state.recipe_data
            recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
            
            if not recipe_df.empty:
                # Crear diccionario de SKUs con recetas
                skus_with_recipes = set(recipe_df["SKU"].astype(str).str.strip())
                
                # Agregar columna de origen
                view_base_det["MMPP_Origen"] = view_base_det["SKU"].astype(str).str.strip().apply(
                    lambda x: "📊 Receta" if x in skus_with_recipes else "📋 Manual"
                )
                
                # Mover la columna MMPP_Origen al inicio (después de las dimensiones)
                cols_order = dim_cols + ["MMPP_Origen"] + [c for c in view_base_det.columns if c not in dim_cols and c != "MMPP_Origen"]
                view_base_det = view_base_det[cols_order]
        
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
            height=700,
            column_config={
                "SKU": st.column_config.TextColumn("SKU", pinned="left"),
                "Descripcion": st.column_config.TextColumn("Descripción", pinned="left"),
            },
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
        if "costos_resumen" in st.session_state:
            to_excel_download(st.session_state["costos_resumen"], "costos_resumen_temporada.xlsx")

    # -------- KPIs y Resumen --------
    st.subheader("📊 Resumen Ejecutivo")

    # Calcular KPIs básicos
    total_skus = len(df_filtrado)
    skus_rentables = len(df_filtrado[df_filtrado["EBITDA (USD/kg)"] > 0])
    ebitda_promedio = df_filtrado["EBITDA (USD/kg)"].mean()
    margen_promedio = df_filtrado["EBITDA Pct"].mean()
    
    # KPIs del sistema de recetas
    recipe_kpis = {}
    if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
        recipe_data = st.session_state.recipe_data
        recipe_df = recipe_data.get("recipe_df", pd.DataFrame())
        fruit_prices = recipe_data.get("fruit_prices", {})
        
        if not recipe_df.empty and fruit_prices:
            # Contar SKUs con MMPP calculado desde recetas
            skus_with_recipes = set(recipe_df["SKU"].astype(str).str.strip())
            skus_with_recipe_mmpp = len([sku for sku in df_filtrado["SKU"].astype(str).str.strip() if sku in skus_with_recipes])
            
            recipe_kpis = {
                "skus_con_recetas": skus_with_recipe_mmpp,
                "total_frutas": len(fruit_prices),
                "porcentaje_cobertura": (skus_with_recipe_mmpp / total_skus * 100) if total_skus > 0 else 0
            }
    
    # Mostrar KPIs en columnas
    col1, col2 = st.columns([1,1])
    # col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", total_skus, help="SKUs con costos reales (excluyendo subproductos)")
        # Información sobre subproductos excluidos en los KPIs
        if len(subproductos) > 0:
            st.caption(f"⚠️ {len(subproductos)} subproductos excluidos (costos = 0)")
    
    with col2:
        st.metric("SKUs Rentables", skus_rentables, f"{skus_rentables/total_skus*100:.1f}%")
    
    # Mostrar KPIs del sistema de recetas si está disponible
    if recipe_kpis:
        st.markdown("---")
        st.subheader("🍎 **KPIs del Sistema de Recetas**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "SKUs con Recetas", 
                recipe_kpis["skus_con_recetas"], 
                f"{recipe_kpis['porcentaje_cobertura']:.1f}% del total"
            )
        
        with col2:
            st.metric(
                "Frutas Base", 
                recipe_kpis["total_frutas"],
                "disponibles para recetas"
            )
        
        with col3:
            st.metric(
                "Cobertura", 
                f"{recipe_kpis['porcentaje_cobertura']:.1f}%",
                f"{recipe_kpis['skus_con_recetas']} de {total_skus} SKUs"
            )
        
        # Agregar métrica adicional para frutas con precios
        if "recipe_data" in st.session_state and st.session_state.recipe_data["success"]:
            fruit_prices = st.session_state.recipe_data.get("fruit_prices", {})
            frutas_con_precio = len([p for p in fruit_prices.values() if p > 0])
            total_frutas = len(fruit_prices)
            
            if frutas_con_precio < total_frutas:
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Frutas con Precios",
                        frutas_con_precio,
                        f"de {total_frutas} totales"
                    )
                
                with col2:
                    porcentaje_precios = (frutas_con_precio / total_frutas * 100) if total_frutas > 0 else 0
                    st.metric(
                        "Cobertura de Precios",
                        f"{porcentaje_precios:.1f}%",
                        f"{frutas_con_precio}/{total_frutas} frutas"
                    )
        
        st.info("💡 **Beneficio**: MMPP (Fruta) calculado automáticamente desde recetas para mayor precisión en costos")
    
    # with col3:
    #     st.metric("EBITDA Promedio", f"${ebitda_promedio:.3f}/kg", help="EBITDA promedio no contiene subproductos")
    # with col4:
    #     st.metric("Margen Promedio", f"{margen_promedio:.1%}")

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

    # Mostrar estado actual de la sesión
    if st.button("📊 Ver Estado de Sesión", type="secondary"):
        st.info("**Estado actual de la sesión:**")
        session_keys = list(st.session_state.keys())
        if session_keys:
            for key in session_keys:
                value = st.session_state[key]
                if isinstance(value, pd.DataFrame):
                    st.write(f"**{key}**: DataFrame con {len(value)} filas")
                else:
                    st.write(f"**{key}**: {type(value).__name__}")
        else:
            st.write("No hay datos en la sesión")