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
import math
import pandas as pd
from pandas import IndexSlice as idx

from src.dynamic_filters import DynamicFiltersWithList

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_io import REQ_SHEETS, columns_config, build_subtotal_row, recalculate_totals, create_subtotal_row
from src.cost_engine import build_cost_engine_pipeline
from src.state import ensure_session_state, session_state_table, sync_filters_to_shared, sync_filters_from_shared
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import pygwalker as pyg
import pandas as pd
import numpy as np
import locale

# ===================== Utilidades =====================

def create_pygwalker_chart(df: pd.DataFrame, title: str = "Análisis de Datos"):
    """
    Crea un visualizador PyGWalker para análisis exploratorio de datos.
    
    Args:
        df: DataFrame a analizar
        title: Título del visualizador
        
    Returns:
        PyGWalker chart object
    """
    # Configurar PyGWalker
    pyg_chart = pyg.walk(
        df,
        spec="./gw_config.json",  # Archivo de configuración opcional
        debug=False,
        use_kernel_calc=True,  # Usar kernel de Python para cálculos
        theme="light",  # Tema claro
        dark="light",   # Forzar tema claro
        show_cloud_tool=False,  # Deshabilitar herramientas de nube
        height=600,     # Altura del visualizador
        width="100%"    # Ancho completo
    )
    
    return pyg_chart

def format_currency_european(value, decimals=0):
    """Formatea un número como moneda con punto para miles y coma para decimales"""
    if pd.isna(value):
        return "N/A"
    
    # Usar locale si está disponible
    try:
        if decimals == 0:
            return f"${locale.format_string('%.0f', value, grouping=True)}"
        else:
            return f"${locale.format_string(f'%.{decimals}f', value, grouping=True)}"
    except:
        # Fallback: formato manual
        if decimals == 0:
            return f"${value:,.0f}".replace(",", ".")
        else:
            formatted = f"{value:,.{decimals}f}"
            parts = formatted.split(".")
            if len(parts) == 2:
                return f"${parts[0].replace(',', '.')},{parts[1]}"
            else:
                return f"${parts[0].replace(',', '.')}"

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
                        "hist.use_cost_engine",
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
    
    st.caption("El archivo debe contener al menos: " + " | ".join([f"**{k}** ({v})" for k,v in REQ_SHEETS.items()]))

# Procesar datos solo si no están en caché o si se recargó
if st.session_state["hist.df"] is None:
    if "hist.file_bytes" in st.session_state and st.session_state["hist.file_bytes"] is not None:
        # ===================== USAR COST ENGINE =====================
        # try:
            with st.spinner("🚀 Procesando archivo con Cost Engine (promedio móvil)..."):
                # Usar el pipeline del cost_engine
                pipeline_results = build_cost_engine_pipeline(
                    st.session_state["hist.file_bytes"]
                )
                
                # Extraer resultados
                detalle = pipeline_results['detalle']
                detalle_optimo = pipeline_results['detalle_optimo']
                df_granel_ponderado = pipeline_results['df_granel_ponderado']
                df_granel_optimo = pipeline_results['df_granel_optimo']
                receta_df = pipeline_results['receta_df']
                info_df = pipeline_results['info_df']
                rolling_months = pipeline_results['rolling_months']
                months_count = pipeline_results['months_count']
    
                # Guardar en session_state
                st.session_state["hist.df"] = detalle
                st.session_state["hist.granel_ponderado"] = df_granel_ponderado
                st.session_state["fruta.receta_df"] = receta_df
                st.session_state["fruta.info_df"] = info_df
                st.session_state["hist.rolling_months"] = rolling_months
                st.session_state["hist.months_count"] = months_count
                
                # Calcular métricas adicionales para compatibilidad
                if "EBITDA (USD/kg)" in detalle.columns and "KgEmbarcados" in detalle.columns:
                    detalle["EBITDA (USD)"] = detalle["EBITDA (USD/kg)"] * detalle["KgEmbarcados"]
                    detalle["EBITDA Simple (USD)"] = detalle["EBITDA (USD)"]
                    st.session_state["hist.ebitda_total"] = detalle["EBITDA (USD)"].sum()
                    st.session_state["hist.ebitda_simple_total"] = detalle["EBITDA Simple (USD)"].sum()
                
                # Crear ebitda_mensual simulado para compatibilidad
                ebitda_mensual = detalle[["SKU-Cliente", "EBITDA (USD)", "KgEmbarcados"]].copy() if "SKU-Cliente" in detalle.columns else pd.DataFrame()
                st.session_state["hist.ebitda_mensual"] = ebitda_mensual
                
                # Datos óptimos (por ahora usar los mismos)
                st.session_state["hist.df_optimo"] = detalle_optimo
                st.session_state["hist.granel_optimo"] = df_granel_optimo
                
                # Mensaje de éxito
                st.success("✅ Datos procesados con Cost Engine usando promedio móvil mes a mes")
                st.info(f"📊 {len(detalle)} SKUs procesados | {len(df_granel_ponderado)} frutas de granel | {months_count} meses")
                st.caption(f"📅 Período: {rolling_months[0]} a {rolling_months[-1]}" if rolling_months else "Sin datos de periodo")
                
        # except Exception as e:
        #     st.error(f"❌ Error procesando con Cost Engine: {str(e)}")
        #     st.stop()
    
    else:
        st.info("Sube tu archivo para comenzar.")
        st.stop()
else:
    detalle = st.session_state["hist.df"]
    
    # Mostrar información del Cost Engine
    months_count = st.session_state.get("hist.months_count", 0)
    rolling_months = st.session_state.get("hist.rolling_months", [])
    st.info(f"🚀 **Datos cargados con Cost Engine** | Promedio móvil de {months_count} meses")
    if rolling_months:
        st.caption(f"📅 Período: {rolling_months[0]} a {rolling_months[-1]}")

# Verificar que detalle esté definido antes de continuar
if 'detalle' not in locals() or detalle is None:
    st.error("❌ No hay datos disponibles para procesar")
    st.info("💡 Por favor, sube tu archivo Excel primero")
    st.stop()

#exportar excel detalle
def create_excel_download_button(df: pd.DataFrame,
    filename: str = "datos_historicos_filtrados.xlsx",
    label: str = "📥 Descargar Excel (Datos Filtrados)",
    key: str = "download_historico_excel"):
    """Crea un botón de descarga Excel para los datos filtrados"""
    from io import BytesIO
    
    # Crear buffer en memoria
    buf = BytesIO()
    
    # Escribir Excel con formato
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Hoja principal con datos mostrados
        df.to_excel(writer, index=False, sheet_name="Datos")
    
    # Crear botón de descarga
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key
    )


if detalle is not None and "Costos Totales (USD/kg)" in detalle.columns:
    # ===== Normalización y clave =====
    df = detalle.copy()
    key_col = "SKU-Cliente" if "SKU-Cliente" in df.columns else "SKU"

    # Cuenta base por clave (mejor que por filas)
    original_count = df[key_col].nunique()

    # Coercer a numérico (detecta strings, 'inf', etc.)
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    df["costos_tot"] = to_num(df.get("Costos Totales (USD/kg)"))
    df["kg"]         = to_num(df.get("KgEmbarcados"))
    df["comex"]      = to_num(df.get("Comex"))
    TOL = 1e-9

    # ===== Decisiones por CLAVE =====
    g = df.groupby(key_col, dropna=False)

    # Sin ventas si suma de kg/comex ≈ 0 o NaN
    kg_tot    = g["kg"].sum(min_count=1)
    comex_tot = g["comex"].sum(min_count=1)
    sin_ventas_key = (kg_tot.fillna(0).abs() <= TOL) | (comex_tot.fillna(0).abs() <= TOL)

    # Costos inválidos si alguna fila de la clave tiene costo NaN/inf/≈0
    cost_inval_key = g.apply(
        lambda x: (x["costos_tot"].isna() | ~np.isfinite(x["costos_tot"]) | (x["costos_tot"].abs() <= TOL)).any()
    )

    sin_ventas_keys   = sin_ventas_key.index[sin_ventas_key]
    subproductos_keys = cost_inval_key.index[cost_inval_key]
    keys_excluir      = sin_ventas_keys.union(subproductos_keys)

    # ===== DataFrames finales =====
    df_base = df[~df[key_col].isin(keys_excluir)].copy()
    filtered_count = df_base[key_col].nunique()

    # Para métricas y tabla del expander
    sin_ventas  = df[df[key_col].isin(sin_ventas_keys)].drop_duplicates(subset=[key_col])
    subproductos = df[df[key_col].isin(subproductos_keys)].drop_duplicates(subset=[key_col])

    skus_excluidos = (
        df[df[key_col].isin(keys_excluir)]
        .drop_duplicates(subset=[key_col])
        .assign(
            Subproducto=lambda x: x[key_col].isin(subproductos_keys),
            **{"Sin Ventas": lambda x: x[key_col].isin(sin_ventas_keys)}
        )
    )

    # Limpieza de columnas auxiliares
    for aux in ("costos_tot", "kg", "comex"):
        skus_excluidos.drop(columns=[aux], inplace=True, errors="ignore")
        df_base.drop(columns=[aux], inplace=True, errors="ignore")

    skus_excluidos_count = skus_excluidos[key_col].nunique()
    
    if original_count > filtered_count:        
        # IMPORTANTE: Recalcular totales en los datos cargados para asegurar que EBITDA Pct esté correcto
        if "EBITDA Pct" in df_base.columns:
            df_base = recalculate_totals(df_base)
        # Mostrar información sobre subproductos excluidos
        with st.expander(f"📋 **SKUs excluidos** ({skus_excluidos_count} SKUs)", expanded=False):
            st.write("**¿Por qué se excluyen estos SKUs?**")
            st.write("Son SKUs sin ventas, con costos totales = 0, que no pueden generar EBITDA real y distorsionan el análisis financiero.")
            
            # Estadísticas de subproductos
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("**Sin ventas:**", len(sin_ventas))
                if "Marca" in skus_excluidos.columns:
                    marca_counts = skus_excluidos["Marca"].value_counts()
                    st.write("**Por Marca:**")
                    for marca, count in marca_counts.head(3).items():
                        st.write(f"- {marca}: {count}")
                else:
                    st.write("**Por Marca:** No disponible")
            
            with col2:
                st.metric("**Con costos totales = 0:**", len(subproductos))
                if "Cliente" in skus_excluidos.columns:
                    cliente_counts = skus_excluidos["Cliente"].value_counts()
                    st.write("**Por Cliente:**")
                    for cliente, count in cliente_counts.head(3).items():
                        st.write(f"- {cliente}: {count}")
                else:
                    st.write("**Por Cliente:** No disponible")
            
            # Tabla completa de subproductos
            st.write("**Lista completa de subproductos excluidos:**")
            
            # Mostrar solo las columnas que existen
            desired_cols = ["SKU", "Descripcion", "Marca", "Cliente", "Especie", "Condicion", "Subproducto", "Sin Ventas"]
            available_cols = [col for col in desired_cols if col in skus_excluidos.columns]
            
            if available_cols:
                st.dataframe(
                    skus_excluidos[available_cols],
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("No hay columnas dimensionales disponibles para mostrar")
            
            # Botón de exportación
            csv_skus_excluidos = skus_excluidos.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Lista Completa de SKUs excluidos (CSV)",
                data=csv_skus_excluidos,
                file_name="subproductos_excluidos_completo.csv",
                mime="text/csv",
                width='stretch',
                key="download_skus_excluidos_sim_1"
            )

# Filtros Dinamicos de la libreria streamlit-dynamic-filters
st.sidebar.header("🔍 Filtros Dinámicos")
with st.sidebar.container():
    if "hist.filters" in st.session_state:
        st.session_state["hist.filters"] = sync_filters_from_shared(page="hist")
        active_filters = st.session_state["hist.filters"]
        active_count = sum(len(v) for v in active_filters.values() if v)
        if active_count > 0:
            if active_count == 1:
                st.sidebar.info(f"🔍 **{active_count} filtro activo**")
            else:
                st.sidebar.info(f"🔍 **{active_count} filtros activos**")
    # Configurar filtros dinámicos solo con columnas que existen
    available_filter_cols = ['Marca', 'Cliente', 'Especie', 'Condicion', 'SKU']
    existing_filter_cols = [col for col in available_filter_cols if col in df_base.columns]
    
    if existing_filter_cols:
        dynamic_filters = DynamicFiltersWithList(df=df_base, filters=existing_filter_cols, filters_name='hist.filters')
    else:
        # Fallback: usar solo SKU si no hay otras columnas
        fallback_cols = [col for col in ['SKU', 'SKU-Cliente'] if col in df_base.columns]
        dynamic_filters = DynamicFiltersWithList(df=df_base, filters=fallback_cols[:1], filters_name='hist.filters')
    dynamic_filters.check_state()
    dynamic_filters.display_filters(location='sidebar')
    df_filtrado = dynamic_filters.filter_df()
    st.session_state["hist.df_filtered"] = df_filtrado

with st.sidebar.container():
    st.button("Resetear Filtros", on_click=dynamic_filters.reset_filters)

sync_filters_to_shared(page="hist", filters=st.session_state["hist.filters"])

# ===================== Pestañas del Histórico =====================
tab_granel, tab_retail = st.tabs(["🏭 Granel (Fruta)", "📊 Retail (SKU)"])

with tab_retail:
    # -------- Mostrar resultados Retail --------
    st.subheader("Resumen por SKU: Márgenes actuales")
    
    # Definir columnas base una sola vez
    base_cols = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion",
                "MMPP (Fruta) (USD/kg)","Proceso Granel (USD/kg)","Retail Costos Directos (USD/kg)",
                "Retail Costos Indirectos (USD/kg)","Almacenaje MMPP","Servicios Generales","Comex",
                "Guarda PT","Gastos Totales (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)",
                "EBITDA (USD/kg)","EBITDA Pct","KgEmbarcados","KgProducidos"]
    
    # Determinar columna clave para filtrado una sola vez
    key_col = "SKU-Cliente" if "SKU-Cliente" in df_filtrado.columns else "SKU"
    
    # Obtener SKUs filtrados una sola vez
    skus_filtrados = df_filtrado[key_col].unique().tolist()
    
    # Filtrar detalle usando la columna clave disponible
    if key_col in detalle.columns:
        try:
            view_base = detalle[detalle[key_col].astype(int).isin(skus_filtrados)].copy()
        except:
            view_base = detalle[detalle[key_col].isin(skus_filtrados)].copy()
    else:
        view_base = detalle.copy()
    
    # Filtrar columnas base que existen
    available_base_cols = [col for col in base_cols if col in view_base.columns]
    view_base = view_base[available_base_cols].copy()
    
    # Establecer índice si la columna clave existe
    if key_col in view_base.columns:
        view_base.set_index(key_col, inplace=True)
    view_base = view_base.sort_index()
     
    show_subtotals_at_top = st.checkbox(
        "Subtotales al inicio",
        value=st.session_state.get("hist.show_subtotals_at_top", False),
        help="Mostrar fila de subtotales al inicio de la tabla",
        key="hist_show_subtotals_at_top"
    )
    
    if show_subtotals_at_top != st.session_state.get("hist.show_subtotals_at_top", False):
        st.session_state["hist.show_subtotals_at_top"] = show_subtotals_at_top
        st.rerun()
    
    config = columns_config(editable=False)
    styled_view_base = view_base.style
    # Aplicar negritas a las columnas de totales
    total_columns = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)",
    "Costos Totales (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
    "KgEmbarcados", "KgProducidos"]
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

    # Streamlit nativo: DataFrame + subtotal separado (arriba o abajo)
    view_base_noidx = view_base.reset_index()
    subtotal_df = build_subtotal_row(view_base_noidx)

    # Mantener el mismo orden de columnas de la vista
    subtotal_df = subtotal_df.reindex(columns=[c for c in view_base_noidx.columns if c in subtotal_df.columns], fill_value="")
    # Dejar sólo métricas: limpiar columnas no numéricas para mayor claridad visual
    try:
        numeric_cols = set(view_base_noidx.select_dtypes(include=[np.number]).columns.tolist())
    except Exception:
        numeric_cols = set()
    for col in subtotal_df.columns:
        if col not in numeric_cols:
            subtotal_df[col] = ""

    col_config = columns_config(editable=False)

    # Construir versión sólo con métricas (oculta dimensiones)
    try:
        numeric_cols_hist = [c for c in view_base_noidx.columns if pd.api.types.is_numeric_dtype(view_base_noidx[c])]
    except Exception:
        numeric_cols_hist = []
    subtotal_display = subtotal_df[[c for c in numeric_cols_hist if c in subtotal_df.columns]].copy()
    sty_sub = subtotal_display.style.set_properties(**{"font-weight":"bold","background-color":"#e8f4fd"})

    if show_subtotals_at_top:
        st.caption("Subtotal (ponderado por KgEmbarcados)")
        st.dataframe(sty_sub, column_config=col_config, width="stretch", hide_index=True)

    # Aplicar formato y estilos similares al simulador
    df_disp = view_base_noidx.copy()
    dims = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion"]
    fmt = {}
    for c in df_disp.columns:
        if c not in dims:
            if ("Pct" in c) or ("Porcentaje" in c):
                fmt[c] = "{:.1%}"
            else:
                try:
                    if pd.api.types.is_numeric_dtype(df_disp[c]):
                        fmt[c] = "{:.3f}"
                except Exception:
                    pass
    sty = df_disp.style
    if fmt:
        sty = sty.format(fmt)
    tot_cols = ["MMPP Total (USD/kg)", "MO Total", "Materiales Total", "Gastos Totales (USD/kg)",
                "Costos Totales (USD/kg)", "Retail Costos Directos (USD/kg)", "Retail Costos Indirectos (USD/kg)",
                "KgEmbarcados", "KgProducidos"]
    ex_tot = [c for c in tot_cols if c in df_disp.columns]
    if ex_tot:
        sty = sty.set_properties(subset=ex_tot, **{"font-weight":"bold","background-color":"#f8f9fa"})
    e_cols = ["EBITDA (USD/kg)", "EBITDA Pct"]
    ex_e = [c for c in e_cols if c in df_disp.columns]
    if ex_e:
        sty = sty.set_properties(subset=ex_e, **{"font-weight":"bold","background-color":"#fff7ed"})

    st.dataframe(sty, column_config=col_config, width="stretch", hide_index=True)

    if not show_subtotals_at_top:
        st.caption("Subtotal (ponderado por KgEmbarcados)")
        st.dataframe(sty_sub, column_config=col_config, width="stretch", hide_index=True)

    # 6. Botón de descarga Excel externo
    if not view_base_noidx.empty:
        # Crear botón de descarga Excel
        def create_excel_download_button(df: pd.DataFrame, filename: str = "datos_historicos_filtrados.xlsx"):
            """Crea un botón de descarga Excel para los datos filtrados"""
            from io import BytesIO
            
            # Crear buffer en memoria
            buf = BytesIO()
            
            # Escribir Excel con formato
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                # Hoja principal con datos mostrados
                df.to_excel(writer, index=False, sheet_name="Datos")
            
            # Crear botón de descarga
            st.download_button(
                label="📥 Descargar Excel (Datos Filtrados)",
                data=buf.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_historico_excel"
            )
        
        # Mostrar botón de descarga
        create_excel_download_button(view_base_noidx)
    
    # # Mostrar métricas de subtotales
    # col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)

    # with col_metrics1:
    #     if "EBITDA (USD/kg)" in view_base_noidx.columns and "KgEmbarcados" in view_base_noidx.columns:
    #         # Convertir a numérico antes de hacer operaciones
    #         ebitda_kg = pd.to_numeric(view_base_noidx["EBITDA (USD/kg)"], errors='coerce')
    #         kg_emb = pd.to_numeric(view_base_noidx["KgEmbarcados"], errors='coerce')
    #         total_ebitda_activo = (ebitda_kg * kg_emb).sum()
    #         st.metric("EBITDA Total Filtrados (USD)", f"{total_ebitda_activo:,.2f}")

    # with col_metrics2:
    #     total_ebitda = st.session_state["hist.ebitda_simple_total"]
    #     st.metric("EBITDA Total (USD)", f"{total_ebitda:,.2f}")
    
    # with col_metrics3:
    #     if "KgEmbarcados" in view_base_noidx.columns:
    #         # Convertir a numérico antes de sumar
    #         kg_emb = pd.to_numeric(view_base_noidx["KgEmbarcados"], errors='coerce')
    #         total_kg = kg_emb.sum()
    #         st.metric("Kg Embarcados Filtrados", f"{total_kg:,.0f}")
            
    # with col_metrics4:
    #     total_rows = len(view_base_noidx)
    #     st.metric("SKUs Filtrados", f"{total_rows:,}")


    # --- Toggle: ver detalle de costos respetando los filtros vigentes ---
    expand = st.toggle("🔎 Expandir costos por SKU (temporada)", value=False)

    if expand:
        # 1) Toma los SKUs actualmente visibles (ya filtrados arriba)
        key_col = "SKU-Cliente" if "SKU-Cliente" in df_filtrado.columns else "SKU"
        
        # Convertir a strings para que coincida con detalle[key_col]
        skus_filtrados = df_filtrado[key_col].astype(str).unique().tolist()
        
        if key_col in detalle.columns:
            # Usar filtrado directo con strings
            det = detalle[detalle[key_col].isin(skus_filtrados)].copy()
        else:
            det = detalle.copy()
        # 3) Mueve atributos DIM a la izquierda
        dim_candidatas = ["SKU","SKU-Cliente","Descripcion","Marca","Cliente","Especie","Condicion"]
        dim_cols = [c for c in dim_candidatas if c in det.columns]
        orden_cols = ["MMPP (Fruta) (USD/kg)", "Proceso Granel (USD/kg)", "MMPP Total (USD/kg)","MO Directa",
                        "MO Indirecta","MO Total","Materiales Directos","Materiales Indirectos","Materiales Total",
                        "Laboratorio","Mantención","Utilities","Fletes Internos","Retail Costos Directos (USD/kg)",
                        "Retail Costos Indirectos (USD/kg)","Almacenaje MMPP","Servicios Generales","Comex","Guarda PT",
                        "Gastos Totales (USD/kg)","Costos Totales (USD/kg)","PrecioVenta (USD/kg)","EBITDA (USD/kg)",
                        "EBITDA Pct","KgEmbarcados","KgProducidos"]
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
        if key_col in det.columns:
            det = det.sort_values([key_col]).reset_index(drop=True)
        
        view_base_det = det.copy()
        
        # Asegurar que el índice sea único antes de aplicar estilos
        if key_col in view_base_det.columns:
            view_base_det = view_base_det.drop_duplicates(subset=[key_col], keep="first")
            view_base_det.set_index(key_col, inplace=True)
        
        # Aplicar estilos de formato y negritas a columnas importantes
        view_base_det = view_base_det.style
        
        # Aplicar negritas a las columnas de totales
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
            width="stretch", 
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
    ebitda_total = st.session_state["hist.ebitda_total"]
    ebitda_actual = df_filtrado["EBITDA (USD)"].sum()
    ebitda_simple_total = st.session_state["hist.ebitda_simple_total"]
    ebitda_simple_actual = df_filtrado["EBITDA Simple (USD)"].sum()

    # Mostrar KPIs en columnas
    # col1, col2 = st.columns([1,1])
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total SKUs", total_skus, help="SKUs con costos reales (excluyendo subproductos)")
        # Información sobre subproductos excluidos en los KPIs
        if len(subproductos) > 0:
            st.caption(f"⚠️ {len(skus_excluidos)} skus excluidos (costos o ventas = 0)")

    with col2:
        if total_skus > 0:
            st.metric("SKUs Rentables", skus_rentables, f"{skus_rentables/total_skus*100:.1f}%")
        else:
            st.metric("SKUs Rentables", skus_rentables, "0.0%")

    # with col3:
    #     st.metric("EBITDA Compañia", format_currency_european(ebitda_total, 0), help="EBITDA total de la compañia (no contiene subproductos)")
    # with col4:
    #     st.metric("EBITDA Activo", format_currency_european(ebitda_actual, 0), help="EBITDA de SKUs visibles (no contiene subproductos)")
    # with col5:
    #     st.metric("EBITDA Simple Total", format_currency_european(ebitda_simple_total, 0), help="EBITDA simple total de la compañia (no contiene subproductos)")
    # with col6:
    #     st.metric("EBITDA Simple Actual", format_currency_european(ebitda_simple_actual, 0), help="EBITDA simple de SKUs visibles (no contiene subproductos)")


with tab_granel:
    st.subheader("🏭 Análisis de Costos de Granel por Fruta")
    
    # Verificar que los datos de granel estén disponibles
    granel_ponderado = st.session_state.get("hist.granel_ponderado")
    info_fruta = st.session_state.get("fruta.info_df")
    precio_fruta = info_fruta[["Precio","Fruta_id"]]
    rendimiento = info_fruta[["Rendimiento", "Fruta_id"]]
    # Agregar columnas de precio y rendimiento
    granel_ponderado = granel_ponderado.merge(precio_fruta, how="left", on="Fruta_id")
    granel_ponderado = granel_ponderado.merge(rendimiento, how="left", on="Fruta_id")
    # Calcular Costo Efectivo
    granel_ponderado["Precio Efectivo"] = granel_ponderado["Precio"] / granel_ponderado["Rendimiento"]
    # Calcular Costos Directos y Costos Indirectos
    granel_ponderado["Costos Directos"] = granel_ponderado["MO Directa"] + granel_ponderado["Materiales Directos"] + granel_ponderado["Laboratorio"] + granel_ponderado["Mantención"]
    granel_ponderado["Costos Indirectos"] = granel_ponderado["MO Indirecta"] + granel_ponderado["Materiales Indirectos"]
    if granel_ponderado is None or granel_ponderado.empty:
        st.error("❌ **No hay datos de granel disponibles**")
        st.info("💡 **Para ver los datos de granel, asegúrate de que tu archivo Excel contenga la hoja 'FACT_GRANEL_POND'**")
    else:
        # Mostrar tabla de granel
        st.subheader("📊 Costos de Granel por Fruta")
        
        # Aplicar formato a la tabla
        granel_display = granel_ponderado.copy()

        # Eliminar las especies sin proceso
        granel_display = granel_display[granel_display["Proceso Granel (USD/kg)"] < 0]

        # Formatear columnas numéricas
        numeric_cols = [col for col in granel_display.columns if col not in ["Fruta_id", "Fruta", "Name", "Nombre"]]
        for col in numeric_cols:
            granel_display[col] = pd.to_numeric(granel_display[col], errors='coerce')
        
        order_cols = ["Fruta_id", "Name", "Precio Efectivo", "Proceso Granel (USD/kg)", "MO Directa", "MO Indirecta",
        "MO Total", "Materiales Directos", "Materiales Indirectos", "Materiales Total", "Laboratorio", "Mantención",
        "Costos Directos", "Costos Indirectos", "Servicios Generales", "Utilities"]
        granel_display = granel_display[order_cols]
        granel_display = granel_display.set_index("Fruta_id").sort_index()
        granel_display = granel_display.sort_values(by="Proceso Granel (USD/kg)")
        # Crear subtotal ponderado por KgProducidos
        subtotal_row = create_subtotal_row(granel_display, weight_col="KgProducidos")
        # etiqueta fija y campos visibles
        subtotal_row.update({
            "Fruta_id": "__TOTAL__",
            "Name": "TOTAL",
        })
        subtotal_df = pd.DataFrame([subtotal_row]).set_index("Fruta_id", drop=True)

        # Concatenar subtotal al final SIN reordenar más
        granel_display = pd.concat([granel_display, subtotal_df.reindex(granel_display.columns, axis=1)], axis=0)
        
        #Formato
        fmt_num = {col: "{:.3f}" for col in numeric_cols}
        fmt_pct = {"Rendimiento": "{:.1%}"} if "Rendimiento" in granel_display.columns else {}

        fmt = fmt_num | fmt_pct
        total_columns = ["MO Total", "Materiales Total", "Costos Directos", "Costos Indirectos"]
        granel_display = granel_display.style

        if total_columns:
                granel_display = granel_display.set_properties(
                    subset=total_columns,
                    **{"font-weight": "bold", "background-color": "#f8f9fa"}
                )
        if "Proceso Granel (USD/kg)" in granel_display.columns:
            granel_display = granel_display.set_properties(
                subset=["Proceso Granel (USD/kg)"],
                    **{"font-weight": "bold", "background-color": "#fff7ed"}
                )
        # Aplicar estilo especial a la fila de subtotales    
        from pandas import IndexSlice as idx
        if "__TOTAL__" in granel_display.index:
            sty = sty.set_properties(
                subset=idx["__TOTAL__", :],
                **{"font-weight":"bold","background-color":"#e8f4fd","border-top":"2px solid #1f77b4"}
            )
        # Mostrar tabla con formato
        st.dataframe(
            granel_display.format(fmt),
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn(
                    "Fruta",
                    disabled=True,
                    pinned="left"
                ),
                "Precio Efectivo": st.column_config.NumberColumn(
                    "Precio Efectivo",
                    disabled=True,
                    pinned="left"
                ),
                "Proceso Granel (USD/kg)": st.column_config.NumberColumn(
                    "Proceso Granel (USD/kg)",
                    disabled=True,
                    pinned="left"
                ),
            },
            hide_index=True
        )
        
        # # KPIs de granel
        # st.subheader("📈 KPIs de Granel")
        
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     st.metric("Total Frutas", len(granel_ponderado))
        
        # with col2:
        #     if "MO Total" in granel_ponderado.columns:
        #         mo_promedio = granel_ponderado["MO Total"].mean()
        #         st.metric("MO Promedio", f"${mo_promedio:.3f}/kg")
        #     else:
        #         st.metric("MO Promedio", "N/A")
        
        # with col3:
        #     if "Materiales Total" in granel_ponderado.columns:
        #         mat_promedio = granel_ponderado["Materiales Total"].mean()
        #         st.metric("Materiales Promedio", f"${mat_promedio:.3f}/kg")
        #     else:
        #         st.metric("Materiales Promedio", "N/A")
        
        # with col4:
        #     if "Laboratorio" in granel_ponderado.columns:
        #         lab_promedio = granel_ponderado["Laboratorio"].mean()
        #         st.metric("Laboratorio Promedio", f"${lab_promedio:.3f}/kg")
        #     else:
        #         st.metric("Laboratorio Promedio", "N/A")
        
        # Análisis por tipo de costo
        st.subheader("📊 Análisis por Tipo de Costo")
        
        # Crear gráfico de barras si hay datos
        try:
            import plotly.express as px
            
            # Preparar datos para el gráfico
            cost_types = [col for col in granel_ponderado.columns if col not in ["Fruta_id", "Fruta", "Name", "Nombre"]]
            
            if cost_types:
                # Calcular promedios por tipo de costo
                cost_averages = []
                for cost_type in cost_types:
                    avg_cost = granel_ponderado[cost_type].mean()
                    cost_averages.append({
                        "Tipo de Costo": cost_type,
                        "Costo Promedio (USD/kg)": abs(avg_cost)  # Valor absoluto para visualización
                    })
                
                cost_df = pd.DataFrame(cost_averages)
                
                # Crear gráfico de barras
                fig = px.bar(
                    cost_df,
                    x="Tipo de Costo",
                    y="Costo Promedio (USD/kg)",
                    title="Costos Promedio por Tipo (USD/kg)",
                    color="Costo Promedio (USD/kg)",
                    color_continuous_scale="Blues"
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
            
        except ImportError:
            st.info("📊 Para ver gráficos, instala plotly: `pip install plotly`")
        
        # Descarga de datos de granel
        st.subheader("📥 Descarga de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Descargar datos de granel
            csv_granel = granel_ponderado.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Datos de Granel (CSV)",
                data=csv_granel,
                file_name="costos_granel_por_fruta.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col2:
            # Descargar Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                granel_ponderado.to_excel(xw, index=False, sheet_name="Granel")
            st.download_button(
                label="📥 Descargar Excel",
                data=buf.getvalue(),
                file_name="costos_granel_por_fruta.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )


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
#         width="stretch"
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
#         width="stretch"
#     )

# -------- Información de navegación --------
st.markdown("---")

# Expander opcional para diagnóstico de session_state
with st.expander("🔎 Diagnóstico session_state", expanded=False):
    session_state_table()

st.info("💡 **Navegación**: Usa el menú lateral para acceder al Simulador EBITDA y otras funcionalidades.")
st.info("💾 **Datos persistentes**: Los archivos cargados se mantienen al cambiar de página.")

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

def render_grid_with_modal(df: pd.DataFrame):
    """
    Renderiza un AgGrid con una columna de acción '🔍'.
    Al hacer click, selecciona la fila y abre un modal con el detalle.
    """

    # --- DataFrame base para el grid (no dependemos del índice para abrir el modal) ---
    grid_df = df.reset_index(drop=True).copy()
    if "__open" not in grid_df.columns:
        grid_df.insert(0, "__open", "")  # columna vacía que renderizamos como botón

    # --- Renderer de botón (selecciona la fila) ---
    OPEN_RENDERER = JsCode("""
    class BtnRenderer {
      init(params){
        this.params = params;
        const b = document.createElement('button');
        b.textContent = '🔍';
        b.style.cursor = 'pointer';
        b.style.padding = '4px 8px';
        b.style.border = '1px solid #ccc';
        b.style.borderRadius = '6px';
        b.style.background = '#f8f9fa';
        b.addEventListener('click', () => {
          // selecciona la fila (esto dispara SELECTION_CHANGED en Python)
          params.api.deselectAll();
          params.node.setSelected(true);
        });
        this.eGui = b;
      }
      getGui(){ return this.eGui; }
      destroy(){ this.eGui = null; }
    }
    """)

    # --- Opciones del grid ---
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    # Columna acción (sin filtro/orden y ancha fija)
    gb.configure_column(
        "__open",
        headerName="",
        width=60,
        pinned="left",
        filter=False,
        sortable=False,
        editable=False,
        cellRenderer=OPEN_RENDERER,
    )

    # Selección de fila (single) — importante para que podamos detectar el click
    gb.configure_selection(selection_mode="single", use_checkbox=False)

    # Altura dinámica (compacta para pocos registros)
    row_h = 32
    header_h = 40
    n = max(len(grid_df), 5)  # mínimo 5 filas de alto para que no quede demasiado chico
    dynamic_height = min(600, header_h + n * row_h + 8)

    grid_options = gb.build()

    resp = AgGrid(
        grid_df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.FILTERING_CHANGED,
        theme="balham",
        height=dynamic_height,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
    )

    # --- Abrir modal cuando hay fila seleccionada ---
    sel_raw = resp.get("selected_rows", None)

    # Normalizar a lista de dicts
    if sel_raw is None:
        sel_records = []
    elif isinstance(sel_raw, pd.DataFrame):
        sel_records = sel_raw.to_dict("records")
    elif isinstance(sel_raw, list):
        sel_records = sel_raw
    else:
        # fallback: intenta convertir a una fila/record
        try:
            sel_records = pd.DataFrame(sel_raw).to_dict("records")
        except Exception:
            sel_records = []

    if len(sel_records) > 0:
        row = pd.Series(sel_records[0])

        @st.dialog(f"Detalle SKU {row.get('SKU', '')}")
        def detalle_sku(row: pd.Series):
            st.title(f"Detalle SKU {row.get('SKU', '')}")
            st.write("**SKU-Cliente:**", row.get("SKU-Cliente", ""))
            st.write("**Descripción:**", row.get("Descripcion", ""))
            st.write("**Marca:**", row.get("Marca", ""))
            st.write("**Cliente:**", row.get("Cliente", ""))
            st.write("**Especie:**", row.get("Especie", ""))
            st.write("**Condición:**", row.get("Condicion", ""))
            st.write("**PrecioVenta (USD/kg):**", row.get("PrecioVenta (USD/kg)", ""))
            st.write("**EBITDA (USD/kg):**", row.get("EBITDA (USD/kg)", ""))
            st.write("**EBITDA Pct:**", row.get("EBITDA Pct", ""))
        detalle_sku(row)

    return resp

# Con Streamlit nativo, no hay click-to-open por fila.
# Como alternativa simple, ofrece un selector para ver detalle.
with st.expander("🔎 Ver detalle de un SKU", expanded=False):
    # Determinar columnas disponibles para el selector
    key_col = "SKU-Cliente" if "SKU-Cliente" in view_base_noidx.columns else "SKU"
    desc_col = "Descripcion" if "Descripcion" in view_base_noidx.columns else key_col
    
    if key_col in view_base_noidx.columns:
        try:
            opciones = view_base_noidx[[key_col, desc_col]].astype(str)
            if desc_col != key_col:
                opciones["label"] = opciones[key_col] + " — " + opciones[desc_col].str.slice(0, 60)
            else:
                opciones["label"] = opciones[key_col]
            
            mapa = dict(zip(opciones["label"], opciones[key_col]))
            elegido = st.selectbox(f"Elige un {key_col}", ["(ninguno)"] + list(mapa.keys()))
            
            if elegido != "(ninguno)":
                sku_sel = mapa[elegido]
                try:
                    fila = view_base_noidx[view_base_noidx[key_col].astype(str) == str(sku_sel)].head(1)
                except:
                    fila = view_base_noidx[view_base_noidx[key_col] == sku_sel].head(1)
                st.dataframe(fila, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Error mostrando selector: {str(e)}")
    else:
        st.info("No hay datos disponibles para mostrar detalles")
