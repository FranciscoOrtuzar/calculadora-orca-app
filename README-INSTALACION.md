# 🚀 CALCULADORA VF - Guía de Instalación

## 📋 Requisitos Previos

- **Python 3.9+** (recomendado: Python 3.11)
- **pip3** (incluido con Python 3.4+)
- **Git** (para clonar el repositorio)

## 🛠️ Instalación Automática (Recomendada)

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd calculadora-VF-app
```

### 2. Ejecutar script de instalación
```bash
./install_dependencies.sh
```

El script automáticamente:
- ✅ Verifica Python y pip
- ✅ Crea un entorno virtual
- ✅ Instala todas las dependencias
- ✅ Configura el entorno

## 🛠️ Instalación Manual

### 1. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate     # En Windows
```

### 2. Actualizar pip
```bash
pip install --upgrade pip
```

### 3. Instalar dependencias

**Para desarrollo (versiones específicas):**
```bash
pip install -r requirements-dev.txt
```

**Para producción (versiones mínimas):**
```bash
pip install -r requirements.txt
```

## 📦 Dependencias Principales

### Core
- **Streamlit 1.28+** - Framework web para la aplicación
- **Pandas 2.0+** - Manipulación y análisis de datos
- **NumPy 1.24+** - Computación numérica

### Visualización
- **Plotly 5.17+** - Gráficos interactivos (frutas, análisis)
- **Altair 5.0+** - Gráficos del simulador principal

### Excel y Archivos
- **OpenPyXL 3.1+** - Archivos Excel (.xlsx)
- **XLRD 2.0+** - Archivos Excel legacy (.xls)

## 🚀 Ejecutar la Aplicación

### 1. Activar entorno virtual
```bash
source venv/bin/activate  # macOS/Linux
# o
venv\Scripts\activate     # Windows
```

### 2. Ejecutar Streamlit
```bash
streamlit run Inicio.py
```

### 3. Abrir en navegador
La aplicación se abrirá automáticamente en `http://localhost:8501`

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Verificar que el entorno virtual esté activado
which python  # Debe mostrar ruta dentro de venv/

# Reinstalar dependencias
pip install --force-reinstall -r requirements.txt
```

### Error: "Permission denied" en script
```bash
chmod +x install_dependencies.sh
```

### Conflictos de versiones
```bash
# Limpiar e instalar desde cero
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

### Actualizar dependencias
```bash
pip install --upgrade -r requirements.txt
```

## 📱 Funcionalidades Disponibles

### 🏠 Página Principal (Inicio.py)
- Carga de archivos Excel
- Procesamiento de datos históricos
- Configuración inicial

### 📊 Simulador EBITDA (pages/1_Simulador_EBITDA.py)
- **Pestaña SKU**: Simulación principal con filtros y overrides
- **Pestaña Precio Fruta**: Ajustes de precios de frutas
- **Pestaña Receta**: Visor de recetas por SKU con paginación

### 🍓 Funcionalidades de Frutas
- Simulador de precios por porcentaje o USD/kg
- Análisis de impacto en SKUs
- Estadísticas y gráficos de frutas
- Overrides de precios con undo/redo

### 🧬 Sistema de Recetas
- Visor paginado de SKUs
- Modal detallado de recetas
- Análisis de composición por fruta
- Exportación de datos

## 🎯 Estructura del Proyecto

```
calculadora-VF-app/
├── Inicio.py                          # Página principal
├── pages/
│   └── 1_Simulador_EBITDA.py        # Simulador principal
├── src/
│   ├── state.py                      # Gestión de estado
│   ├── data_io.py                    # Carga de datos
│   └── simulator_fruit.py            # Lógica de frutas
├── requirements.txt                   # Dependencias mínimas
├── requirements-dev.txt               # Dependencias de desarrollo
├── install_dependencies.sh           # Script de instalación
└── README-INSTALACION.md             # Este archivo
```

## 🔄 Mantenimiento

### Actualizar dependencias
```bash
# Verificar versiones actuales
pip list

# Actualizar a versiones más recientes
pip install --upgrade streamlit pandas numpy plotly altair
```

### Limpiar entorno
```bash
# Desactivar entorno virtual
deactivate

# Eliminar y recrear
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que Python sea 3.9+
2. Asegúrate de que el entorno virtual esté activado
3. Revisa los logs de error
4. Ejecuta `pip list` para verificar dependencias instaladas

## 🌐 Despliegue Web

### Streamlit Cloud (Recomendado)
1. **Subir a GitHub**: Haz push de tu código a un repositorio público
2. **Conectar con Streamlit Cloud**: Ve a [share.streamlit.io](https://share.streamlit.io)
3. **Configurar despliegue**: 
   - Selecciona tu repositorio
   - Branch: `main`
   - Main file: `main.py`
4. **Desplegar**: La aplicación estará disponible en una URL pública

### Heroku
1. **Instalar Heroku CLI**
2. **Login**: `heroku login`
3. **Crear app**: `heroku create tu-app-name`
4. **Desplegar**: `git push heroku main`

### Otras plataformas
- **Railway**: Conecta tu repositorio GitHub
- **Render**: Despliegue automático desde GitHub
- **DigitalOcean App Platform**: Similar a Heroku

### Archivos de configuración incluidos:
- ✅ `Procfile` - Para Heroku y similares
- ✅ `runtime.txt` - Versión de Python
- ✅ `.streamlit/config.toml` - Configuración de Streamlit
- ✅ `setup.sh` - Script de configuración
- ✅ `requirements.txt` - Dependencias actualizadas

## 🎉 ¡Listo!

Tu entorno de desarrollo está configurado y listo para usar todas las funcionalidades de la Calculadora VF:

- ✅ Simulador de EBITDA por SKU
- ✅ Simulador de Granel con sincronización automática
- ✅ Ajustes de precios de frutas
- ✅ Visor de recetas con paginación
- ✅ Análisis de impacto en tiempo real
- ✅ Exportación de datos y escenarios
- ✅ Sistema de undo/redo completo
- ✅ Despliegue web listo
