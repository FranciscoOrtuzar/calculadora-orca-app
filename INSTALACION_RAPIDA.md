# 🚀 Instalación Rápida - Calculadora VF

## ⚡ Instalación en 3 pasos

### 1. Preparar entorno
```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno (macOS/Linux)
source .venv/bin/activate

# Activar entorno (Windows)
.venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Crear datos de ejemplo
```bash
python3 data/create_sample_data.py
```

### 4. Ejecutar aplicación
```bash
streamlit run app.py
```

## 🌐 Acceso a la aplicación

- **URL local**: http://localhost:8501
- **Home**: Vista de datos históricos
- **Simulador**: Análisis avanzado y simulación

## 📁 Estructura mínima requerida

```
calculadora-VF-app/
├── data/
│   ├── costos.xlsx          # Hoja: Costos_ponderados
│   └── precios.xlsx         # Hoja: FACT_PRECIOS
├── src/
├── pages/
└── app.py
```

## 🔧 Solución rápida de problemas

### Error: "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### Error: "No se pudieron cargar los datos base"
```bash
python3 data/create_sample_data.py
```

### Error: "streamlit: command not found"
```bash
pip install streamlit
```

## 📊 Datos de ejemplo incluidos

- **8 SKUs** con costos y precios
- **3 Clientes** (A, B, C)
- **3 Marcas** (1, 2, 3)
- **3 Especies** (1, 2, 3)
- **3 Condiciones** (1, 2, 3)

## 🎯 Funcionalidades disponibles

✅ **Home**: Datos históricos y análisis  
✅ **Simulador**: Filtros, overrides y EBITDA  
✅ **Filtros**: Por Cliente, Marca, Especie, Condición  
✅ **Overrides**: Global, por archivo, manual  
✅ **KPIs**: EBITDA, márgenes, top/bottom SKUs  
✅ **Gráficos**: Barras y distribución de márgenes  
✅ **Export**: Escenarios a CSV  

---

**¿Problemas?** Revisa el README.md completo o crea un issue en GitHub.
