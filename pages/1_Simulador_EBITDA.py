# Simulador EBITDA
"""
Página del simulador de EBITDA para análisis de escenarios y rentabilidad.
Permite simular variaciones en precios y costos, con filtros y análisis detallado.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import io

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_io import build_mart, REQ_SHEETS
from simulator import (
    apply_simulation, calculate_kpis, create_ebitda_chart, 
    create_margin_distribution_chart, export_scenario, get_filter_options,
    apply_filters, create_scenario_summary, identify_critical_skus
)

# ===================== Configuración de la página =====================
st.set_page_config(
    page_title="Simulador EBITDA",
    page_icon="📊",
    layout="wide"
)

# ===================== Navegación =====================
def show_navigation():
    """Muestra la navegación entre páginas"""
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navegación")
    
    if st.sidebar.button("🏠 Home - Datos Históricos"):
        st.session_state.current_page = "home"
        st.rerun()
    
    if st.sidebar.button("📊 Simulador EBITDA", type="primary"):
        st.rerun()

st.title("📊 Simulador EBITDA")
st.markdown("Simula escenarios de variación en precios y costos para analizar impacto en rentabilidad.")

# Mostrar navegación
show_navigation()

# ===================== Carga de datos (con persistencia) =====================
def load_data_from_session():
    """Carga datos desde la sesión de Streamlit"""
    if "mart" in st.session_state and "detalle" in st.session_state:
        st.success("✅ Datos cargados desde sesión (archivo ya procesado)")
        return st.session_state.mart
    else:
        st.warning("⚠️ No hay datos cargados en la sesión")
        return None

def load_data_from_local():
    """Intenta cargar datos desde archivo local."""
    try:
        from data_io import _build_costos_from_local_file
        df_costos = _build_costos_from_local_file()
        if df_costos is not None and not df_costos.empty:
            st.success("✅ Datos cargados desde archivo local 'Costos ponderados.xlsx'")
            return df_costos
    except Exception as e:
        st.warning(f"No se pudo cargar archivo local: {e}")
    return None

def load_data_from_upload():
    """Carga datos desde archivo subido."""
    uploaded_file = st.file_uploader(
        "Archivo Excel con hojas: " + ", ".join(REQ_SHEETS.keys()),
        type=["xlsx"],
        help="El archivo debe contener las hojas: " + " | ".join([f"**{k}** ({v})" for k,v in REQ_SHEETS.items()]),
        key="file_uploader_simulator"
    )
    
    if uploaded_file is None:
        return None
    
    try:
        file_bytes = uploaded_file.read()
        mart, detalle = build_mart(file_bytes, ultimo_precio_modo="global", ref_ym=None)
        
        # Guardar en sesión para persistencia
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_bytes = file_bytes
        st.session_state.mart = mart
        st.session_state.detalle = detalle
        
        st.success("✅ Archivo cargado exitosamente")
        return mart
    except Exception as e:
        st.error(f"❌ Error procesando el archivo: {e}")
        return None

# Cargar datos con prioridad: sesión > local > upload
df_original = load_data_from_session()

if df_original is None:
    df_original = load_data_from_local()

if df_original is None:
    st.info("📁 Sube tu archivo Excel con las hojas requeridas para continuar.")
    df_original = load_data_from_upload()

if df_original is None or df_original.empty:
    st.error("No se pudieron cargar los datos. Verifica el archivo e intenta nuevamente.")
    st.stop()

# ===================== Información de Datos =====================
st.header("📊 Información de Datos Cargados")

# Mostrar columnas disponibles
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Columnas Disponibles")
    st.write(f"**Total de columnas:** {len(df_original.columns)}")
    st.write(f"**Total de SKUs:** {len(df_original)}")
    
    # Mostrar columnas de costos disponibles
    cost_columns = [col for col in df_original.columns if any(keyword in col.lower() for keyword in ['costo', 'mmpp', 'guarda', 'retail'])]
    if cost_columns:
        st.write("**Columnas de costos:**")
        for col in sorted(cost_columns):
            st.write(f"• {col}")
    
    # Mostrar columnas de precios disponibles
    price_columns = [col for col in df_original.columns if 'precio' in col.lower()]
    if price_columns:
        st.write("**Columnas de precios:**")
        for col in sorted(price_columns):
            st.write(f"• {col}")

with col2:
    st.subheader("🔍 Columnas para Simulación")
    
    # Verificar columnas requeridas para simulación
    required_cols = [
        "PrecioVenta (USD/kg)",
        "Retail Costos Directos (USD/kg)",
        "Retail Costos Indirectos (USD/kg)",
        "MMPP (Fruta) (USD/kg)",
        "MMPP (Proceso Granel) (USD/kg)",
        "Guarda MMPP",
        "EBITDA (USD/kg)",
        "EBITDA Pct"
    ]
    
    available_for_sim = []
    missing_for_sim = []
    
    for col in required_cols:
        if col in df_original.columns:
            available_for_sim.append(col)
        else:
            missing_for_sim.append(col)
    
    if available_for_sim:
        st.success(f"✅ **{len(available_for_sim)} columnas disponibles** para simulación:")
        for col in available_for_sim:
            st.write(f"• {col}")
    
    if missing_for_sim:
        st.warning(f"⚠️ **{len(missing_for_sim)} columnas faltantes** para simulación completa:")
        for col in missing_for_sim:
            st.write(f"• {col}")
        st.info("💡 La simulación funcionará con las columnas disponibles, pero algunos cálculos pueden ser limitados.")

# ===================== Filtros =====================
st.header("🔍 Filtros")

# Obtener opciones de filtro
filter_options = get_filter_options(df_original)

# Crear filtros en columnas
if filter_options:
    filter_cols = st.columns(len(filter_options))
    selected_filters = {}
    
    for i, (field, options) in enumerate(filter_options.items()):
        with filter_cols[i]:
            selected = st.multiselect(
                field,
                options=options,
                default=[],
                key=f"filter_{field}"
            )
            selected_filters[field] = selected
    
    # Aplicar filtros
    df_filtered = apply_filters(df_original, selected_filters)
    st.info(f"📊 Mostrando {len(df_filtered)} SKUs de {len(df_original)} totales")
else:
    df_filtered = df_original.copy()
    st.info("📊 Mostrando todos los SKUs (sin filtros aplicados)")

# ===================== Simulación =====================
st.header("⚙️ Simulación de Escenarios")

# Parámetros de simulación global
st.subheader("📈 Variaciones Globales (%)")

sim_cols = st.columns(5)
with sim_cols[0]:
    price_up = st.number_input("Precio de Venta", value=0.0, step=0.5, format="%.2f", help="Variación en precio de venta")
with sim_cols[1]:
    retail_direct_up = st.number_input("Costos Retail Directos", value=0.0, step=0.5, format="%.2f", help="Variación en costos retail directos")
with sim_cols[2]:
    retail_indirect_up = st.number_input("Costos Retail Indirectos", value=0.0, step=0.5, format="%.2f", help="Variación en costos retail indirectos")
with sim_cols[3]:
    mmpp_up = st.number_input("Costos MMPP", value=0.0, step=0.5, format="%.2f", help="Variación en costos MMPP")
with sim_cols[4]:
    guarda_up = st.number_input("Costos Guarda", value=0.0, step=0.5, format="%.2f", help="Variación en costos de guarda")

# Botón para aplicar simulación
if st.button("🚀 Aplicar Simulación", type="primary"):
    with st.spinner("Calculando escenario simulado..."):
        try:
            df_simulated = apply_simulation(
                df_filtered, 
                price_up=price_up,
                retail_direct_up=retail_direct_up,
                retail_indirect_up=retail_indirect_up,
                mmpp_up=mmpp_up,
                guarda_up=guarda_up
            )
            st.session_state.df_simulated = df_simulated
            st.success("✅ Simulación aplicada exitosamente!")
        except Exception as e:
            st.error(f"❌ Error en la simulación: {e}")
            st.info("💡 Verifica que las columnas requeridas estén presentes en tus datos")

# ===================== Resultados de Simulación =====================
if 'df_simulated' in st.session_state:
    df_simulated = st.session_state.df_simulated
    
    st.header("📊 Resultados de la Simulación")
    
    # KPIs comparativos
    st.subheader("📈 KPIs Comparativos")
    
    # Calcular KPIs para ambos escenarios
    try:
        kpis_original = calculate_kpis(df_filtered)
        kpis_simulated = calculate_kpis(df_simulated)
        
        # Mostrar KPIs en métricas
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.metric(
                "EBITDA Promedio",
                f"${kpis_simulated['EBITDA Promedio (USD/kg)']:.3f}/kg",
                f"{kpis_simulated['EBITDA Promedio (USD/kg)'] - kpis_original['EBITDA Promedio (USD/kg)']:.3f}"
            )
        
        with kpi_cols[1]:
            st.metric(
                "EBITDA Total",
                f"${kpis_simulated['EBITDA Total (USD)']:.0f}",
                f"{kpis_simulated['EBITDA Total (USD)'] - kpis_original['EBITDA Total (USD)']:.0f}"
            )
        
        with kpi_cols[2]:
            st.metric(
                "SKUs Rentables",
                kpis_simulated['SKUs Rentables'],
                kpis_simulated['SKUs Rentables'] - kpis_original['SKUs Rentables']
            )
        
        with kpi_cols[3]:
            st.metric(
                "EBITDA Promedio (%)",
                f"{kpis_simulated['EBITDA Promedio (%)']:.1f}%",
                f"{kpis_simulated['EBITDA Promedio (%)'] - kpis_original['EBITDA Promedio (%)']:.1f}%"
            )
        
        # Resumen comparativo detallado
        st.subheader("📋 Resumen Comparativo")
        summary_comparison = create_scenario_summary(df_filtered, df_simulated)
        
        # Formato correcto para porcentajes
        st.dataframe(
            summary_comparison.style.format({
                "Escenario Original": "{:.3f}",
                "Escenario Simulado": "{:.3f}",
                "Variación": "{:.3f}",
                "Variación %": "{:.1f}%"  # Formato de porcentaje
            }),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Error calculando KPIs: {e}")
        st.info("💡 Verifica que las columnas requeridas estén presentes en tus datos")
    
    # ===================== Gráficos =====================
    st.header("📊 Visualizaciones")
    
    # Seleccionar campo para agrupar
    group_field = st.selectbox(
        "Agrupar por:",
        options=[col for col in ["Marca", "Especie", "Cliente"] if col in df_simulated.columns],
        key="group_field"
    )
    
    if group_field:
        try:
            # Gráfico de EBITDA por grupo
            ebitda_chart = create_ebitda_chart(df_simulated, group_by=group_field)
            if ebitda_chart:
                st.altair_chart(ebitda_chart, use_container_width=True)
            
            # Gráfico de distribución de márgenes
            margin_chart = create_margin_distribution_chart(df_simulated)
            if margin_chart:
                st.altair_chart(margin_chart, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creando gráficos: {e}")
    
    # ===================== Análisis de SKUs Críticos =====================
    st.subheader("⚠️ SKUs Críticos")
    
    threshold = st.slider(
        "Umbral de EBITDA (%) para considerar crítico:",
        min_value=-50.0,
        max_value=0.0,
        value=-10.0,
        step=5.0,
        help="SKUs con EBITDA por debajo de este umbral se consideran críticos"
    )
    
    try:
        critical_skus = identify_critical_skus(df_simulated, threshold=threshold/100)
        
        if not critical_skus.empty:
            st.warning(f"⚠️ Se encontraron {len(critical_skus)} SKUs críticos")
            
            # Mostrar SKUs críticos
            critical_cols = ["SKU", "Marca", "Especie", "Cliente", "PrecioVenta (USD/kg)", 
                            "Costos Totales (USD/kg)", "EBITDA (USD/kg)", "EBITDA Pct"]
            critical_display = critical_skus[critical_cols].copy()
            
            st.dataframe(
                critical_display.style.format({
                    "PrecioVenta (USD/kg)": "{:.3f}",
                    "Costos Totales (USD/kg)": "{:.3f}",
                    "EBITDA (USD/kg)": "{:.3f}",
                    "EBITDA Pct": "{:.1%}"  # Formato de porcentaje
                }),
                use_container_width=True
            )
        else:
            st.success("✅ No se encontraron SKUs críticos con el umbral seleccionado")
    except Exception as e:
        st.error(f"❌ Error identificando SKUs críticos: {e}")
    
    # ===================== Exportar Escenario =====================
    st.subheader("💾 Exportar Escenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_name = st.text_input(
            "Nombre del escenario:",
            value="escenario_simulado",
            help="Nombre para el archivo de exportación"
        )
    
    with col2:
        if st.button("📥 Descargar Excel", type="secondary"):
            try:
                excel_data = export_scenario(df_simulated, scenario_name)
                st.download_button(
                    label="⬇️ Descargar archivo Excel",
                    data=excel_data,
                    file_name=f"{scenario_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Archivo preparado para descarga")
            except Exception as e:
                st.error(f"❌ Error exportando escenario: {e}")

# ===================== Overrides por Fila =====================
st.header("🎯 Overrides Específicos por SKU")

st.info("💡 Puedes modificar valores específicos para SKUs individuales")

# Seleccionar SKU para override
if not df_filtered.empty:
    sku_options = sorted(df_filtered["SKU"].unique())
    selected_sku = st.selectbox("Seleccionar SKU:", options=sku_options, key="override_sku")
    
    if selected_sku:
        # Obtener datos del SKU seleccionado
        sku_data = df_filtered[df_filtered["SKU"] == selected_sku].iloc[0]
        
        st.subheader(f"Modificar valores para SKU: {selected_sku}")
        
        # Mostrar valores actuales y permitir modificación
        override_cols = st.columns(3)
        
        with override_cols[0]:
            st.metric("Precio Actual", f"${sku_data['PrecioVenta (USD/kg)']:.3f}/kg")
            new_price = st.number_input(
                "Nuevo Precio (USD/kg)",
                value=float(sku_data['PrecioVenta (USD/kg)']),
                step=0.01,
                format="%.3f",
                key="override_price"
            )
        
        with override_cols[1]:
            st.metric("Costos Totales Actuales", f"${sku_data['Costos Totales (USD/kg)']:.3f}/kg")
            new_costs = st.number_input(
                "Nuevos Costos Totales (USD/kg)",
                value=float(sku_data['Costos Totales (USD/kg)']),
                step=0.01,
                format="%.3f",
                key="override_costs"
            )
        
        with override_cols[2]:
            st.metric("EBITDA Actual", f"${sku_data['EBITDA (USD/kg)']:.3f}/kg")
            new_ebitda = new_price - new_costs
            st.metric("Nuevo EBITDA", f"${new_ebitda:.3f}/kg", f"{new_ebitda - sku_data['EBITDA (USD/kg)']:.3f}")
        
        # Botón para aplicar override
        if st.button("✅ Aplicar Override", key="apply_override"):
            if 'df_simulated' in st.session_state:
                df_override = st.session_state.df_simulated.copy()
                df_override.loc[df_override["SKU"] == selected_sku, "PrecioVenta (USD/kg)_Sim"] = new_price
                df_override.loc[df_override["SKU"] == selected_sku, "Costos Totales (USD/kg)_Sim"] = new_costs
                df_override.loc[df_override["SKU"] == selected_sku, "EBITDA (USD/kg)_Sim"] = new_ebitda
                df_override.loc[df_override["SKU"] == selected_sku, "EBITDA Pct_Sim"] = (new_ebitda / new_price) * 100
                
                st.session_state.df_simulated = df_override
                st.success(f"✅ Override aplicado para SKU {selected_sku}")
            else:
                st.warning("⚠️ Primero debes aplicar una simulación global")

# ===================== Información adicional =====================
st.markdown("---")
st.markdown("""
### 📚 Información del Simulador

Este simulador te permite:

1. **Simular variaciones globales** en precios y costos
2. **Aplicar overrides específicos** por SKU
3. **Analizar impacto** en EBITDA y rentabilidad
4. **Identificar SKUs críticos** con bajo rendimiento
5. **Exportar escenarios** para análisis posterior

### 🔧 Cómo usar

1. **Aplica filtros** para enfocar tu análisis
2. **Configura variaciones globales** en precios y costos
3. **Ejecuta la simulación** para ver resultados
4. **Analiza KPIs** y gráficos comparativos
5. **Identifica SKUs críticos** que requieren atención
6. **Exporta el escenario** para compartir o analizar

### 📊 Interpretación de resultados

- **EBITDA positivo**: El SKU es rentable
- **EBITDA negativo**: El SKU genera pérdidas
- **Variaciones positivas**: Mejora en rentabilidad
- **Variaciones negativas**: Deterioro en rentabilidad
""")

# -------- Información de navegación --------
st.markdown("---")
st.info("💡 **Navegación**: Usa el menú lateral para volver a la página principal o acceder a otras funcionalidades.")
st.info("💾 **Datos persistentes**: Los archivos cargados se mantienen al cambiar de página.")
