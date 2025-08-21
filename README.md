# Calculadora VF - Aplicación de Costos y Márgenes

Aplicación multipage de Streamlit para análisis de costos ponderados, precios y márgenes de rentabilidad.

## 🚀 Características

- **📊 Datos Históricos**: Vista principal con análisis de EBITDA y filtros dinámicos
- **⚙️ Simulador EBITDA**: Simulación de escenarios con variaciones en precios y costos
- **🔍 Filtros Avanzados**: Por Cliente, Marca, Especie y Condición
- **📈 KPIs y Gráficos**: Visualizaciones interactivas con Altair
- **💾 Exportación**: Descarga de escenarios simulados en Excel

## 📁 Estructura del Proyecto

```
calculadora-VF-app/
├── Inicio.py                          # Página Home (Datos Históricos)
├── pages/
│   └── 1_Simulador_EBITDA.py      # Simulador de EBITDA
├── src/
│   ├── __init__.py                 # Módulo src
│   ├── data_io.py                  # Entrada/salida de datos
│   └── simulator.py                # Funciones de simulación
├── data/                           # Directorio para archivos de datos
├── outputs/                        # Directorio para archivos generados
└── requirements.txt                # Dependencias del proyecto
```

## 🛠️ Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repositorio>
   cd calculadora-VF-app
   ```

2. **Crear entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**:
   ```bash
   streamlit run app.py
   ```

## 📊 Uso

### Página Home - Datos Históricos
- Sube tu archivo Excel con las hojas requeridas
- Visualiza análisis de EBITDA por SKU
- Aplica filtros dinámicos
- Expande detalles de costos por SKU

### Simulador EBITDA
- Simula variaciones globales en precios y costos
- Aplica overrides específicos por SKU
- Analiza impacto en rentabilidad
- Identifica SKUs críticos
- Exporta escenarios simulados

## 📋 Hojas Requeridas

Tu archivo Excel debe contener:

1. **FACT_COSTOS_POND**: Costos unitarios ponderados por SKU
2. **FACT_PRECIOS**: Precios mensuales (SKU, Año, Mes, PrecioVentaUSD)
3. **DIM_SKU**: Dimensión de SKU (Marca, Especie, Cliente, etc.)

## 🔧 Configuración

### Archivo Local de Costos
Si no tienes la hoja `FACT_COSTOS_POND`, coloca tu archivo `Costos ponderados.xlsx` en:
- `data/Costos ponderados.xlsx` (preferido)
- O en el directorio raíz del proyecto

### Personalización
- Modifica `src/data_io.py` para nuevas fuentes de datos
- Ajusta `src/simulator.py` para nuevas métricas
- Personaliza filtros en las páginas

## 📈 Funcionalidades del Simulador

- **Variaciones Globales**: Ajusta precios y costos por porcentaje
- **Overrides por SKU**: Modifica valores específicos por producto
- **Análisis de KPIs**: EBITDA promedio, total, SKUs rentables
- **Identificación de Críticos**: SKUs con bajo rendimiento
- **Visualizaciones**: Gráficos de EBITDA y distribución de márgenes
- **Exportación**: Escenarios completos en Excel

## 🚨 Solución de Problemas

### Error: "No se encontró columna SKU"
- Verifica que tu archivo tenga una columna con "SKU", "Código" o similar
- Revisa que no haya espacios extra en los nombres de columnas

### Error: "Faltan hojas requeridas"
- Asegúrate de que tu Excel contenga las hojas especificadas
- Verifica los nombres exactos de las hojas

### Datos numéricos no se convierten
- El app maneja automáticamente comas y puntos decimales
- Verifica que los números no tengan caracteres especiales

## 🔄 Agregar Nuevas Fuentes de Datos

### 1. Nueva Hoja de Costos
```python
# En src/data_io.py
def build_tbl_costos_pond(df_costos):
    # Agregar mapeo para nuevas columnas
    if lc == "nueva_columna":
        rename_map[c] = "Nueva Columna Normalizada"
```

### 2. Nueva Dimensión
```python
# En src/data_io.py
def build_dim_sku(df_dim):
    expected = ["SKU", "Condicion", "Descripcion", "Marca", "Especie", "Cliente", "NuevaDimension"]
```

### 3. Nueva Métrica
```python
# En src/data_io.py
def build_mart():
    # Agregar cálculo de nueva métrica
    mart["NuevaMetrica"] = calculo_nuevo
```

## 📞 Soporte

Para problemas o preguntas:
1. Revisa la documentación en los comentarios del código
2. Verifica que tu archivo Excel tenga la estructura esperada
3. Revisa los mensajes de error para identificar el problema específico

## 🎯 Roadmap

- [ ] Análisis de tendencias temporales
- [ ] Comparación entre múltiples escenarios
- [ ] Dashboard ejecutivo con métricas agregadas
- [ ] Integración con bases de datos externas
- [ ] Reportes automáticos por email

---

**Desarrollado para análisis de rentabilidad y simulación de escenarios de costos**
