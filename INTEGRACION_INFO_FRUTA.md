# Integración con Hoja INFO_FRUTA

## Descripción

El sistema de cálculo de precios por receta ahora está integrado con la hoja **INFO_FRUTA** para obtener información más precisa sobre precios y eficiencias de las frutas base. **El sistema ahora maneja automáticamente las 44 frutas/ingredientes diferentes** de tu hoja INFO_FRUTA.

## Estructura de la Hoja INFO_FRUTA

La hoja debe contener las siguientes columnas:

**⚠️ Importante**: La columna `eficiencia` usa **coma (,) como separador decimal**, no punto (.). Por ejemplo: `0,85` en lugar de `0.85`.

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `fruta_id` | Identificador único de la fruta | F1, F2, F3... |
| `fruta` | Nombre de la fruta (puede incluir sufijos) | Arandano_precio, Frutilla_sub |
| `conv` | Precio convertido/ajustado (USD/kg) | 2.50, 3.20 |
| `org` | Precio original (USD/kg) | 2.45, 3.15 |
| `eficiencia` | Factor de eficiencia del proceso (usa coma como separador decimal) | 0,85, 0,90 |

## Mapeo de Frutas Base

El sistema reconoce automáticamente **todas las 44 frutas** de tu hoja INFO_FRUTA, incluyendo:

### Frutas Principales
- **Arandano** → Blueberry
- **Frutilla** → Strawberry  
- **Mora** → Blackberry
- **Frambuesa** → Raspberry
- **Mango** → Mango
- **Piña** → Pineapple
- **Banana** → Banana
- **Kiwi** → Kiwi

### Frutas Especiales
- **Cereza_oscura** → Dark Cherry
- **Cereza_roja** → Red Cherry
- **Cereza_acida** → Sour Cherry
- **Granada** → Pomegranate
- **Dragon_fruit** → Dragon Fruit
- **Maracuya** → Passion Fruit

### Ingredientes y Semillas
- **Kale** → Kale
- **Espinaca** → Spinach
- **Chia** → Chia
- **Cacao** → Cacao
- **sem_calab** → Pumpkin Seed
- **Cañamo** → Hemp

### Variedades y Grados
- **Framsub** → Raspberry Sub
- **frutsub** → Strawberry Sub
- **Arsub** → Blueberry Sub
- **frambAA** → Raspberry AA
- **sliced_frutilla** → Sliced Strawberry
- **diced_frutilla** → Diced Strawberry

### Y muchas más...
El sistema detecta automáticamente nuevas frutas y las agrega dinámicamente.

## Cálculo de Precios por Receta

### Fórmula Principal

```
Costo_Fruta = (Proporción / 100) × Precio_Fruta / Eficiencia
```

### Ejemplo

Para un SKU con:
- **Arandano**: 60% de la receta
- **Precio Arandano**: $2.50/kg
- **Eficiencia Arandano**: 85% (0.85)

```
Costo_Arandano = (60 / 100) × $2.50 / 0.85 = $1.76/kg
```

## Prioridad de Datos

1. **INFO_FRUTA** (prioridad alta): Si está disponible, se usan estos datos
2. **Detalle** (prioridad baja): Como respaldo si INFO_FRUTA no está disponible

## Columnas de la Hoja INFO_FRUTA

### Sufijos Reconocidos

- `_precio`: Precio principal de la fruta
- `_sub`: Subproducto de la fruta
- `_org`: Fruta orgánica
- `_prec`: Abreviatura de precio
- `_AA`: Grado AA de la fruta
- `_oscura`: Variedad oscura
- `_roja`: Variedad roja
- `_acida`: Variedad ácida

### Procesamiento Automático

El sistema:
1. **Identifica automáticamente** todas las frutas de la hoja INFO_FRUTA
2. **Extrae nombres base** eliminando sufijos automáticamente
3. **Agrega nuevas frutas** dinámicamente al sistema
4. **Usa la columna `conv`** como precio principal
5. **Usa la columna `org`** como respaldo si `conv` es 0 o nulo
6. **Extrae la eficiencia** de la columna `eficiencia` (maneja comas como separadores decimales)
7. **Aplica valores por defecto** para eficiencias no especificadas

## Ejemplo de Datos

```csv
fruta_id,fruta,conv,org,eficiencia
F1,Arandano_precio,2.50,2.45,0,85
F2,Frutilla_precio,3.20,3.15,0,90
F3,Mora_precio,1.088,1.50,0,80
F4,Frambuesa_precio,2.80,2.75,0,75
F5,sliced_frutilla_precio,3.50,3.45,0,90
F6,Cereza_oscura_precio,4.20,4.15,0,85
F7,Mango_precio,2.80,2.75,0,88
F8,Kale_precio,1.50,1.45,0,95
```

## Beneficios de la Integración

1. **Completitud**: Maneja las 44 frutas/ingredientes de tu sistema
2. **Precisión**: Precios y eficiencias específicos por fruta
3. **Flexibilidad**: Múltiples tipos de frutas (precio, sub, org, AA, etc.)
4. **Mantenibilidad**: Fácil actualización de precios y eficiencias
5. **Trazabilidad**: Origen claro de los datos utilizados
6. **Expansibilidad**: Detecta y agrega nuevas frutas automáticamente

## Uso en el Simulador

En la pestaña "Simulación por Receta":

1. **Fuente de datos**: Muestra si los datos vienen de INFO_FRUTA o Detalle
2. **Precios y eficiencias**: Muestra todas las frutas disponibles con sus eficiencias
3. **Frutas principales**: Vista resumida de las frutas más importantes
4. **Todas las frutas**: Expander con todas las 44 frutas disponibles
5. **Simulación**: Al cambiar un precio de fruta, se recalcula automáticamente
6. **Desglose**: Muestra las proporciones y eficiencias de cada receta

## Compatibilidad

- **Si no hay hoja INFO_FRUTA**: El sistema funciona con valores por defecto
- **Los valores por defecto** se mantienen como respaldo
- **No se requieren cambios** en las recetas existentes
- **Nuevas frutas** se detectan y agregan automáticamente
- **Sistema escalable** para futuras expansiones
