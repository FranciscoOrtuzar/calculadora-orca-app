# 📁 Directorio de Datos

Este directorio debe contener los archivos de datos necesarios para la aplicación.

## 📋 Archivos Requeridos

### 1. `costos.xlsx`
**Hoja:** `Costos_ponderados`

Columnas mínimas:
- `SKU` - Código del producto
- `Cliente` - Cliente del SKU
- `Marca` - Marca del producto
- `Especie` - Especie del producto
- `Condicion` - Condición del producto
- `Costos Totales (USD/kg)` - Costo total por kilogramo

### 2. `precios.xlsx`
**Hoja:** `FACT_PRECIOS`

Columnas mínimas:
- `SKU` - Código del producto
- `Cliente` - Cliente del SKU
- `PrecioVenta (USD/kg)` - Precio de venta por kilogramo

## 🚀 Crear Datos de Ejemplo

### Opción 1: Usar tu archivo existente
1. Copia tu archivo `Costos ponderados.xlsx` a este directorio
2. Renómbralo a `costos.xlsx`
3. Asegúrate de que tenga la hoja `Costos_ponderados`

### Opción 2: Crear archivo manual
1. Crea un archivo Excel con las columnas mencionadas arriba
2. Guarda como `costos.xlsx` en este directorio
3. Asegúrate de que tenga la hoja `Costos_ponderados`

### Opción 3: Usar datos de la página Home
1. Ve a la página Home de la aplicación
2. Carga tu archivo Excel maestro
3. Los datos estarán disponibles para el simulador

## 📊 Estructura de Datos Esperada

```
costos.xlsx (Hoja: Costos_ponderados)
├── SKU: "SKU001", "SKU002", ...
├── Cliente: "Cliente A", "Cliente B", ...
├── Marca: "Marca 1", "Marca 2", ...
├── Especie: "Especie 1", "Especie 2", ...
├── Condicion: "Cond 1", "Cond 2", ...
└── Costos Totales (USD/kg): 1.50, 2.00, ...

precios.xlsx (Hoja: FACT_PRECIOS)
├── SKU: "SKU001", "SKU002", ...
├── Cliente: "Cliente A", "Cliente B", ...
└── PrecioVenta (USD/kg): 3.00, 4.50, ...
```

## 🔧 Solución de Problemas

### Error: "No se pudieron cargar los datos base"
- Verifica que los archivos existan en este directorio
- Verifica que tengan los nombres correctos
- Verifica que tengan las hojas correctas

### Error: "Columnas faltantes"
- Verifica que los archivos tengan las columnas mínimas
- Los nombres de columnas pueden variar (se mapean automáticamente)

### Error: "Hoja no encontrada"
- Verifica que `costos.xlsx` tenga la hoja `Costos_ponderados`
- Verifica que `precios.xlsx` tenga la hoja `FACT_PRECIOS`

## 💡 Consejos

1. **Usa la página Home primero**: Carga tu archivo maestro allí
2. **Navega al simulador**: Los datos estarán disponibles automáticamente
3. **Verifica columnas**: Asegúrate de que tus datos tengan las columnas necesarias
4. **Formato de números**: Usa punto decimal (1.50) o coma decimal (1,50) - ambos funcionan
