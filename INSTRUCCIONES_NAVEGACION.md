# 🧭 Instrucciones de Navegación - Calculadora VF

## ✅ Problemas Solucionados

### 1. **Navegación entre Páginas**
- ✅ **Navegación basada en estado de sesión** (más confiable que `st.switch_page`)
- ✅ **Botones de navegación** en el sidebar de ambas páginas
- ✅ **Cambio de página instantáneo** sin errores de Streamlit
- ✅ **Datos persistentes** entre páginas

### 2. **Formato de Porcentajes**
- ✅ **Columnas de porcentaje** ahora se muestran correctamente (ej: 15.2%)
- ✅ **Formato aplicado** a todas las tablas con columnas de porcentaje
- ✅ **Formato numérico mejorado** para otras columnas

### 3. **Errores de Caché y Columnas Faltantes**
- ✅ **CachedWidgetWarning eliminado** - Widgets movidos fuera de funciones cacheadas
- ✅ **KeyError resuelto** - Función de simulación ahora verifica columnas disponibles
- ✅ **Simulación robusta** - Funciona con cualquier conjunto de columnas disponibles
- ✅ **Información de datos** - Muestra qué columnas están disponibles para simulación

### 4. **Persistencia de Datos entre Páginas**
- ✅ **Datos persistentes** - Los archivos cargados se mantienen al cambiar de página
- ✅ **Sin recarga** - No es necesario subir el archivo nuevamente
- ✅ **Estado compartido** - Filtros y configuraciones se mantienen
- ✅ **Recarga opcional** - Botón para recargar archivo si es necesario

## 🚀 Cómo Ejecutar la Aplicación

### Opción 1: Con entorno virtual
```bash
# Activar entorno virtual
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

### Opción 2: Con pip directamente
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## 🧭 Cómo Funciona la Navegación

### **Sistema de Navegación Implementado**
- **Estado de sesión**: Usa `st.session_state.current_page` para controlar qué página mostrar
- **Importación dinámica**: La página del simulador se importa y ejecuta dinámicamente
- **Sin dependencias externas**: No depende de la estructura de páginas automática de Streamlit
- **Navegación confiable**: Funciona en todas las versiones de Streamlit

### **Página Home (Datos Históricos)**
- Ubicación: `app.py`
- Funcionalidad: Análisis de EBITDA, filtros, resumen ejecutivo
- Navegación: Botón "📊 Simulador EBITDA" en el sidebar

### **Página Simulador EBITDA**
- Ubicación: `pages/1_Simulador_EBITDA.py`
- Funcionalidad: Simulación de escenarios, KPIs, gráficos
- Navegación: Botón "🏠 Home - Datos Históricos" en el sidebar

## 🔧 Estructura de Archivos

```
calculadora-VF-app/
├── app.py                          # 🏠 Página Home + Controlador de navegación
├── pages/
│   ├── __init__.py                 # Inicialización del paquete
│   └── 1_Simulador_EBITDA.py      # 📊 Simulador EBITDA
├── src/
│   ├── __init__.py                 # Módulo src
│   ├── data_io.py                  # Entrada/salida de datos
│   └── simulator.py                # Funciones de simulación (robustas)
├── .streamlit/
│   └── config.toml                 # Configuración de Streamlit
└── requirements.txt                 # Dependencias
```

## 🎯 Funcionalidades por Página

### **🏠 Home - Datos Históricos**
- ✅ Carga de archivo Excel
- ✅ **Persistencia de datos** - Los archivos se mantienen en sesión
- ✅ Filtros dinámicos (Marca, Cliente, Especie, Condición)
- ✅ Tabla principal con EBITDA y costos
- ✅ Expansión de detalles por SKU
- ✅ KPIs y resumen ejecutivo
- ✅ **Navegación al Simulador** (funcional)
- ✅ **Recarga opcional** - Botón para actualizar archivo

### **📊 Simulador EBITDA**
- ✅ **Datos automáticos** - Usa archivos ya cargados en Home
- ✅ **Información de datos** - Muestra columnas disponibles y faltantes
- ✅ Filtros avanzados
- ✅ **Simulación robusta** - Funciona con cualquier conjunto de columnas
- ✅ Simulación de variaciones globales
- ✅ Overrides específicos por SKU
- ✅ KPIs comparativos
- ✅ Gráficos interactivos
- ✅ Identificación de SKUs críticos
- ✅ Exportación de escenarios
- ✅ **Navegación al Home** (funcional)
- ✅ **Carga de respaldo** - Si no hay datos en sesión, permite subir archivo

## 🚨 Solución de Problemas

### **Problema: Error "Could not find page"**
**Solución Implementada:**
- ✅ **Navegación basada en estado de sesión** en lugar de `st.switch_page()`
- ✅ **Importación dinámica** de la página del simulador
- ✅ **Sin dependencias** de la estructura de páginas automática de Streamlit

### **Problema: CachedWidgetWarning**
**Solución Implementada:**
- ✅ **Widgets movidos fuera** de funciones cacheadas
- ✅ **Funciones separadas** para carga local y por upload
- ✅ **Sin decoradores @st.cache_data** en funciones con widgets

### **Problema: KeyError en columnas faltantes**
**Solución Implementada:**
- ✅ **Verificación de columnas** antes de usarlas en simulación
- ✅ **Simulación robusta** que funciona con cualquier conjunto de datos
- ✅ **Información clara** sobre qué columnas están disponibles
- ✅ **Manejo de errores** con mensajes informativos

### **Problema: Formato de porcentaje incorrecto**
**Solución:**
- ✅ Las columnas con "Pct" o "Porcentaje" se formatean automáticamente
- ✅ Formato `{:.1%}` aplicado a todas las columnas de porcentaje
- ✅ Formato numérico `{:.3f}` para otras columnas numéricas

### **Problema: Navegación no funciona**
**Solución:**
- ✅ Usa los botones del sidebar para navegar
- ✅ El estado de sesión mantiene la página activa
- ✅ `st.rerun()` actualiza la interfaz correctamente

### **Problema: Datos se pierden al cambiar de página**
**Solución Implementada:**
- ✅ **Persistencia automática** - Los archivos se guardan en `st.session_state`
- ✅ **Datos compartidos** - Home y Simulador usan los mismos datos
- ✅ **Sin recarga** - Los archivos se mantienen entre navegaciones
- ✅ **Recarga opcional** - Botón para actualizar si es necesario

## 📱 Uso de la Aplicación

### **Paso 1: Cargar Datos**
1. Ejecuta `streamlit run app.py`
2. Sube tu archivo Excel con las hojas requeridas
3. Verifica que se carguen correctamente

### **Paso 2: Navegar al Simulador**
1. En la página Home, usa el botón "📊 Simulador EBITDA" del sidebar
2. La aplicación cambiará instantáneamente a la página del simulador
3. **Los datos se mantienen automáticamente** - No es necesario recargar el archivo
4. **Filtros y configuraciones** se preservan entre páginas

### **Paso 3: Revisar Información de Datos**
1. **Nueva funcionalidad**: La página del simulador ahora muestra:
   - Columnas disponibles en tus datos
   - Columnas requeridas para simulación completa
   - Advertencias sobre columnas faltantes
2. Esto te ayuda a entender qué puedes simular
3. **Los datos ya están cargados** desde la página Home

### **Paso 4: Usar el Simulador**
1. Aplica filtros si es necesario
2. Configura variaciones globales en precios y costos
3. Ejecuta la simulación (ahora más robusta)
4. Analiza resultados y KPIs
5. Exporta el escenario si lo deseas

### **Paso 5: Volver al Home**
1. Usa el botón "🏠 Home - Datos Históricos" del sidebar
2. Regresa instantáneamente a la vista principal de datos históricos

## 🔍 Verificación de Funcionamiento

### **Indicadores de Éxito:**
- ✅ **Botones de navegación visibles** en el sidebar
- ✅ **Cambio de página instantáneo** al hacer clic en los botones
- ✅ **Columnas de porcentaje con formato correcto** (ej: 15.2%)
- ✅ **Datos persistentes** entre páginas
- ✅ **Funcionalidades completas** en ambas páginas
- ✅ **Sin errores de navegación** de Streamlit
- ✅ **Sin CachedWidgetWarning**
- ✅ **Simulación funciona** con cualquier conjunto de columnas
- ✅ **Información clara** sobre datos disponibles
- ✅ **Archivos se mantienen** al cambiar de página
- ✅ **Sin recarga necesaria** del archivo Excel

### **Si algo no funciona:**
1. Verifica la consola del navegador para errores
2. Revisa que todas las dependencias estén instaladas
3. Asegúrate de que la estructura de archivos sea correcta
4. Reinicia la aplicación
5. Verifica que no haya conflictos con versiones anteriores
6. **Nuevo**: Revisa la sección "Información de Datos" en el simulador

## 📞 Soporte

Para problemas adicionales:
1. **Revisa los mensajes de error** en la consola del navegador
2. **Verifica que la versión de Streamlit** sea ≥ 1.28.0
3. **Asegúrate de que todos los archivos** estén en su lugar
4. **Consulta la documentación** en el código
5. **Revisa el estado de sesión** con `st.write(st.session_state)`
6. **Nuevo**: Usa la sección "Información de Datos" para diagnosticar problemas de columnas

## 🔧 Detalles Técnicos de la Navegación

### **Cómo funciona internamente:**
1. **Estado de sesión**: `st.session_state.current_page` controla qué página mostrar
2. **Condicional**: `if st.session_state.current_page == "simulator":` determina qué renderizar
3. **Importación dinámica**: `importlib.util` carga la página del simulador cuando es necesario
4. **Persistencia**: Los datos y filtros se mantienen en `st.session_state`

### **Ventajas de este enfoque:**
- ✅ **Compatible con todas las versiones** de Streamlit
- ✅ **Sin dependencias externas** de navegación
- ✅ **Control total** sobre el flujo de la aplicación
- ✅ **Datos persistentes** entre cambios de página
- ✅ **Fácil de mantener** y extender

## 🆕 Nuevas Funcionalidades del Simulador

### **Información de Datos:**
- **Análisis automático** de columnas disponibles
- **Verificación de requisitos** para simulación completa
- **Advertencias claras** sobre columnas faltantes
- **Guía visual** de qué se puede simular

### **Simulación Robusta:**
- **Funciona con cualquier** conjunto de columnas
- **Verificación automática** de disponibilidad
- **Cálculos adaptativos** según datos disponibles
- **Manejo de errores** informativo

## 💾 Sistema de Persistencia de Datos

### **Cómo funciona:**
1. **Carga inicial**: Subes el archivo en la página Home
2. **Almacenamiento automático**: Los datos se guardan en `st.session_state`
3. **Navegación sin pérdida**: Al cambiar de página, los datos se mantienen
4. **Uso compartido**: Home y Simulador usan los mismos datos
5. **Recarga opcional**: Botón para actualizar archivo si es necesario

### **Ventajas:**
- ✅ **No más recargas** - Los archivos se mantienen automáticamente
- ✅ **Tiempo ahorrado** - No es necesario subir el archivo en cada página
- ✅ **Estado consistente** - Filtros y configuraciones se preservan
- ✅ **Experiencia fluida** - Navegación rápida entre funcionalidades
- ✅ **Datos sincronizados** - Cambios en una página se reflejan en la otra

### **Casos de uso:**
- **Análisis continuo**: Carga archivo una vez, analiza en ambas páginas
- **Comparación**: Simula escenarios y compara con datos históricos
- **Filtrado**: Aplica filtros en Home, úsalos en Simulador
- **Exportación**: Simula y exporta sin recargar datos

---

**¡La aplicación ahora tiene navegación completa y confiable entre páginas, formato correcto de porcentajes, y simulación robusta que funciona con cualquier conjunto de datos!** 🎉

**Estado**: ✅ **FUNCIONANDO COMPLETAMENTE** - Todos los problemas resueltos, funcionalidades robustas implementadas.
