"""
Módulo de cálculo de precios por receta.
Calcula precios de SKUs basándose en proporciones de frutas y eficiencias del proceso.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

# ===================== FUNCIONES DE CÁLCULO =====================

def load_fruit_info(sheets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """
    Carga información de frutas desde la hoja INFO_FRUTA.
    Implementación limpia que siempre convierte precios y eficiencias a float.
    
    Args:
        sheets: Diccionario con DataFrames de las hojas del Excel
        
    Returns:
        Tuple con (fruit_prices, fruit_efficiency, fruit_names)
    """
    if "INFO_FRUTA" not in sheets:
        st.error("❌ Hoja INFO_FRUTA no encontrada")
        return {}, {}, {}
    
    try:
        # Leer INFO_FRUTA
        info_fruta_df = sheets["INFO_FRUTA"].copy()
        
        # Limpiar nombres de columnas
        info_fruta_df.columns = [col.strip() for col in info_fruta_df.columns]
        
        # Verificar columnas requeridas
        required_cols = ["fruta_id", "precio", "eficiencia"]
        if not all(col in info_fruta_df.columns for col in required_cols):
            st.error(f"❌ INFO_FRUTA debe contener las columnas: {required_cols}")
            return {}, {}, {}
        
        # Inicializar diccionarios
        fruit_prices = {}
        fruit_efficiency = {}
        fruit_names = {}
        
        # Contadores
        total_frutas = 0
        frutas_con_precio = 0
        frutas_con_eficiencia = 0
        
        # Procesar cada fila
        for idx, row in info_fruta_df.iterrows():
            # Obtener fruta_id
            fruta_id = str(row.get("fruta_id", "")).strip()
            
            # Saltar filas sin fruta_id válido
            if not fruta_id or fruta_id == "nan":
                continue
            
            total_frutas += 1
            
            # CONVERTIR PRECIO A FLOAT
            precio_raw = row.get("precio", 0.0)
            precio = 0.0
            
            if pd.notna(precio_raw):
                try:
                    if isinstance(precio_raw, str):
                        # Limpiar string
                        precio_str = precio_raw.strip().replace('$', '').replace(' ', '')
                        
                        # Manejar separadores decimales
                        if precio_str.count(',') == 1 and precio_str.count('.') == 0:
                            # Solo coma - separador decimal (formato europeo)
                            precio_str = precio_str.replace(',', '.')
                        elif precio_str.count(',') == 1 and precio_str.count('.') == 1:
                            # Coma y punto - coma es decimal, punto es miles
                            precio_str = precio_str.replace('.', '').replace(',', '.')
                        
                        precio = float(precio_str)
                    else:
                        precio = float(precio_raw)
                        
                    if precio > 0:
                        frutas_con_precio += 1
                        
                except (ValueError, TypeError):
                    precio = 0.0
            
            # CONVERTIR EFICIENCIA A FLOAT
            eficiencia_raw = row.get("eficiencia", 0.9)
            eficiencia = 0.9
            
            if pd.notna(eficiencia_raw):
                try:
                    if isinstance(eficiencia_raw, str):
                        # Limpiar string
                        eficiencia_str = eficiencia_raw.strip().replace(' ', '')
                        
                        # Manejar separadores decimales
                        if eficiencia_str.count(',') == 1:
                            eficiencia_str = eficiencia_str.replace(',', '.')
                        
                        eficiencia = float(eficiencia_str)
                    else:
                        eficiencia = float(eficiencia_raw)
                        
                    if eficiencia > 0:
                        frutas_con_eficiencia += 1
                        
                except (ValueError, TypeError):
                    eficiencia = 0.9
            
            # OBTENER NOMBRE DE FRUTA
            name = row.get("name", "")
            nombre = row.get("nombre", "")
            
            if pd.notna(name) and str(name).strip() and str(name).strip() != "nan":
                fruit_names[fruta_id] = str(name).strip()
            elif pd.notna(nombre) and str(nombre).strip() and str(nombre).strip() != "nan":
                fruit_names[fruta_id] = str(nombre).strip()
            else:
                fruit_names[fruta_id] = fruta_id
            
            # GUARDAR EN DICCIONARIOS (SIEMPRE)
            fruit_prices[fruta_id] = precio
            fruit_efficiency[fruta_id] = eficiencia
        
        # Resumen final
        return fruit_prices, fruit_efficiency, fruit_names
        
    except Exception as e:
        st.error(f"❌ Error cargando INFO_FRUTA: {e}")
        return {}, {}, {}

def diagnose_fruit_data(fruit_prices: Dict[str, float], fruit_efficiency: Dict[str, float], fruit_names: Dict[str, str], info_fruta_df: pd.DataFrame = None) -> str:
    """
    Genera un diagnóstico detallado de los datos de frutas cargados.
    
    Args:
        fruit_prices: Diccionario con precios de frutas
        fruit_efficiency: Diccionario con eficiencias de frutas
        fruit_names: Diccionario con nombres de frutas
        info_fruta_df: DataFrame completo de INFO_FRUTA (opcional)
        
    Returns:
        String con el diagnóstico
    """
    total_frutas = len(fruit_prices)
    frutas_con_precio = len([p for p in fruit_prices.values() if p > 0])
    frutas_con_eficiencia = len([e for e in fruit_efficiency.values() if e > 0])
    frutas_con_nombre = len([n for n in fruit_names.values() if n and n != "nan"])
    
    # Analizar distribución de precios
    precios_cero = len([p for p in fruit_prices.values() if p == 0])
    precios_positivos = len([p for p in fruit_prices.values() if p > 0])
    precios_negativos = len([p for p in fruit_prices.values() if p < 0])
    
    # Analizar eficiencias
    eficiencias_default = len([e for e in fruit_efficiency.values() if e == 0.9])
    eficiencias_custom = frutas_con_eficiencia - eficiencias_default
    
    diagnosis = f"""
# 🔍 Diagnóstico de Datos de Frutas

## 📊 Estadísticas Generales
- **Total de frutas procesadas**: {total_frutas}
- **Frutas con precio > 0**: {frutas_con_precio} ({frutas_con_precio/total_frutas*100:.1f}%)
- **Frutas con precio = 0**: {precios_cero} ({precios_cero/total_frutas*100:.1f}%)
- **Frutas con precio < 0**: {precios_negativos} ({precios_negativos/total_frutas*100:.1f}%)

## ⚙️ Eficiencias
- **Frutas con eficiencia > 0**: {frutas_con_eficiencia} ({frutas_con_eficiencia/total_frutas*100:.1f}%)
- **Eficiencias personalizadas**: {eficiencias_custom}
- **Eficiencias por defecto (0.9)**: {eficiencias_default}

## 🏷️ Nombres
- **Frutas con nombres**: {frutas_con_nombre} ({frutas_con_nombre/total_frutas*100:.1f}%)
- **Frutas sin nombres**: {total_frutas - frutas_con_nombre} ({(total_frutas - frutas_con_nombre)/total_frutas*100:.1f}%)
"""
    
    # Si tenemos el DataFrame completo de INFO_FRUTA, mostrar información adicional
    if info_fruta_df is not None:
        total_filas_info_fruta = len(info_fruta_df)
        frutas_con_recetas = len(fruit_prices)
        frutas_sin_recetas = total_filas_info_fruta - frutas_con_recetas
        
        diagnosis += f"""
## 📋 Análisis de INFO_FRUTA vs Recetas
- **Total filas en INFO_FRUTA**: {total_filas_info_fruta}
- **Frutas con recetas**: {frutas_con_recetas}
- **Frutas sin recetas**: {frutas_sin_recetas} ({frutas_sin_recetas/total_filas_info_fruta*100:.1f}%)

### 🔍 Explicación:
- **INFO_FRUTA** contiene {total_filas_info_fruta} frutas base disponibles
- **RECETA_SKU** solo usa {frutas_con_recetas} de esas frutas
- **{frutas_sin_recetas} frutas** están disponibles pero no se usan en recetas
"""
        
        # Mostrar algunas frutas sin recetas
        if frutas_sin_recetas > 0:
            frutas_sin_recetas_list = [f for f in info_fruta_df["fruta_id"].unique() if f not in fruit_prices][:5]
            diagnosis += f"\n**Ejemplos de frutas sin recetas (primeras 5):**\n"
            for fruta_id in frutas_sin_recetas_list:
                nombre = info_fruta_df[info_fruta_df["fruta_id"] == fruta_id]["name"].iloc[0] if "name" in info_fruta_df.columns else fruta_id
                diagnosis += f"- {nombre} ({fruta_id})\n"
            
            if frutas_sin_recetas > 5:
                diagnosis += f"... y {frutas_sin_recetas - 5} frutas más\n"
    
    diagnosis += "\n## ⚠️ Problemas Detectados\n"
    
    if precios_cero > 0:
        diagnosis += f"- **{precios_cero} frutas sin precios** - Estas frutas no contribuirán al cálculo de MMPP\n"
    
    if precios_negativos > 0:
        diagnosis += f"- **{precios_negativos} frutas con precios negativos** - Esto puede causar cálculos incorrectos\n"
    
    if frutas_con_precio < total_frutas * 0.5:
        diagnosis += f"- **Baja cobertura de precios** - Solo {frutas_con_precio/total_frutas*100:.1f}% de las frutas tienen precios\n"
    
    if eficiencias_default > total_frutas * 0.8:
        diagnosis += f"- **Muchas eficiencias por defecto** - {eficiencias_default/total_frutas*100:.1f}% de las frutas usan eficiencia 0.9\n"
    
    # Mostrar algunas frutas de ejemplo
    diagnosis += "\n## 🍎 Ejemplos de Frutas\n"
    
    # Frutas con precios
    if precios_positivos > 0:
        frutas_ejemplo = [f for f, p in fruit_prices.items() if p > 0][:3]
        diagnosis += f"**Frutas con precios (primeras 3):**\n"
        for fruta_id in frutas_ejemplo:
            nombre = fruit_names.get(fruta_id, fruta_id)
            precio = fruit_prices[fruta_id]
            eficiencia = fruit_efficiency.get(fruta_id, 0.9)
            diagnosis += f"- {nombre} ({fruta_id}): ${precio:.3f}/kg, eficiencia {eficiencia:.2f}\n"
    
    # Frutas sin precios
    if precios_cero > 0:
        frutas_sin_precio = [f for f, p in fruit_prices.items() if p == 0][:3]
        diagnosis += f"\n**Frutas sin precios (primeras 3):**\n"
        for fruta_id in frutas_sin_precio:
            nombre = fruit_names.get(fruta_id, fruta_id)
            diagnosis += f"- {nombre} ({fruta_id}): Sin precio\n"
    
    return diagnosis

def load_recipe_data(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Carga y procesa los datos de recetas desde la hoja RECETA_SKU.
    Ahora usa la columna "Porcentaje" para las proporciones.
    
    Args:
        sheets: Diccionario con todas las hojas del Excel
        
    Returns:
        DataFrame procesado con recetas
    """
    if "RECETA_SKU" not in sheets:
        raise ValueError("Hoja 'RECETA_SKU' no encontrada en el archivo Excel")
    
    recipe_df = sheets["RECETA_SKU"].copy()
    
    # Limpiar y normalizar columnas
    recipe_df.columns = [col.strip() for col in recipe_df.columns]
    
    # Mapear nombres de columnas esperados
    column_mapping = {
        "Código": "SKU",
        "Producto": "Descripcion",
        "Marca": "Marca",
        "Cliente": "Cliente",
        "Condición_": "Condicion"
    }
    
    # Aplicar mapeo de columnas
    for old_name, new_name in column_mapping.items():
        if old_name in recipe_df.columns:
            recipe_df = recipe_df.rename(columns={old_name: new_name})
    
    # Asegurar que SKU sea string
    if "SKU" in recipe_df.columns:
        recipe_df["SKU"] = recipe_df["SKU"].astype(str).str.strip()
    
    # Normalizar columna de porcentaje
    if "Porcentaje" in recipe_df.columns:
        recipe_df["Porcentaje"] = pd.to_numeric(recipe_df["Porcentaje"], errors='coerce').fillna(0.0)
    
    # Validar que los porcentajes sean lógicos (0-100)
    if "Porcentaje" in recipe_df.columns:
        recipe_df["Porcentaje_Valido"] = (
            (recipe_df["Porcentaje"] >= 0) & 
            (recipe_df["Porcentaje"] <= 100)
        )
    
    return recipe_df

def calculate_sku_price_by_recipe(
    sku: str, 
    recipe_df: pd.DataFrame, 
    fruit_prices: Dict[str, float],
    fruit_efficiency: Dict[str, float]
) -> Dict:
    """
    Calcula el precio de fruta de un SKU específico basado en su receta usando fruta_id.
    
    Args:
        sku: SKU del producto
        recipe_df: DataFrame con recetas
        fruit_prices: Diccionario con precios de frutas usando fruta_id como clave
        fruit_efficiency: Diccionario con eficiencias de frutas usando fruta_id como clave
        
    Returns:
        Diccionario con información del cálculo
    """
    # Buscar todas las recetas del SKU (puede tener múltiples frutas)
    sku_recipes = recipe_df[recipe_df["SKU"] == sku]
    
    if sku_recipes.empty:
        return {
            "success": False,
            "error": f"SKU {sku} no encontrado en las recetas"
        }
    
    # Calcular precio por receta
    total_cost = 0.0
    fruit_breakdown = {}
    total_percentage = 0.0
    recipe_valid = True
    
    # Iterar sobre todas las filas del SKU (puede tener múltiples frutas)
    for _, recipe_row in sku_recipes.iterrows():
        # Obtener fruta_id y porcentaje de esta fila
        fruta_id = recipe_row.get("fruta_id", "")
        porcentaje = recipe_row.get("Porcentaje", 0).astype(float)
        
        if fruta_id and fruta_id in fruit_prices and porcentaje > 0:
            # Calcular costo de esta fruta considerando la eficiencia
            fruit_cost_per_kg = fruit_prices[fruta_id]
            efficiency = fruit_efficiency.get(fruta_id, 0.9)
            
            # Costo = (Porcentaje / 100) * Precio fruta / Eficiencia
            fruit_cost = (porcentaje / 100) * fruit_cost_per_kg / efficiency
            
            total_cost += fruit_cost
            total_percentage += porcentaje
            
            fruit_breakdown[fruta_id] = {
                "proportion": porcentaje,
                "price_per_kg": fruit_cost_per_kg,
                "efficiency": efficiency,
                "cost": fruit_cost
            }
        
        # Verificar si la receta es válida (porcentaje entre 0-100)
        if not recipe_row.get("Porcentaje_Valido", True):
            recipe_valid = False
    
    # Verificar que el porcentaje total sea razonable (no necesariamente 100% exacto)
    if total_percentage > 100.1:  # Tolerancia de 0.1%
        recipe_valid = False
    
    return {
        "success": True,
        "sku": sku,
        "total_cost": total_cost,
        "total_percentage": total_percentage,
        "fruit_breakdown": fruit_breakdown,
        "recipe_valid": recipe_valid,
        "num_fruits": len(fruit_breakdown)
    }

def calculate_all_recipe_prices(
    recipe_df: pd.DataFrame, 
    fruit_prices: Dict[str, float],
    fruit_efficiency: Dict[str, float],
    fruit_names: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Calcula precios para todos los SKUs basados en sus recetas usando fruta_id.
    
    Args:
        recipe_df: DataFrame con recetas
        fruit_prices: Diccionario con precios de frutas usando fruta_id como clave
        fruit_efficiency: Diccionario con eficiencias de frutas usando fruta_id como clave
        fruit_names: Diccionario con nombres de frutas usando fruta_id como clave
        
    Returns:
        DataFrame con resultados del cálculo
    """
    results = []
    
    # Agrupar por SKU para manejar recetas con múltiples frutas
    for sku in recipe_df["SKU"].unique():
        sku_recipes = recipe_df[recipe_df["SKU"] == sku]
        
        if not sku_recipes.empty:
            # Calcular precio total para este SKU
            calculation = calculate_sku_price_by_recipe(sku, recipe_df, fruit_prices, fruit_efficiency)
            
            if calculation["success"]:
                # Crear resultado base
                result = {
                    "SKU": sku,
                    "fruta_id": ", ".join(calculation["fruit_breakdown"].keys()),
                    "Porcentaje": calculation["total_percentage"],
                    "MMPP_Fruta_Calculado": calculation["total_cost"],
                    "Num_Frutas": calculation["num_fruits"],
                    "Recipe_Valid": calculation["recipe_valid"]
                }
                
                # Agregar nombres de frutas si están disponibles
                if fruit_names:
                    result["Nombre_Fruta"] = ", ".join([
                        fruit_names.get(fruta_id, fruta_id) 
                        for fruta_id in calculation["fruit_breakdown"].keys()
                    ])
                
                # Agregar desglose por fruta
                for fruta_id, fruit_data in calculation["fruit_breakdown"].items():
                    result[f"{fruta_id}_Porcentaje"] = fruit_data["proportion"]
                    result[f"{fruta_id}_Precio"] = fruit_data["price_per_kg"]
                    result[f"{fruta_id}_Eficiencia"] = fruit_data["efficiency"]
                    result[f"{fruta_id}_Costo"] = fruit_data["cost"]
                    
                    # Agregar nombre de fruta si está disponible
                    if fruit_names:
                        result[f"{fruta_id}_Nombre"] = fruit_names.get(fruta_id, fruta_id)
                
                results.append(result)
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

def compare_recipe_vs_original_prices(
    recipe_prices: pd.DataFrame,
    detalle_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compara precios calculados por receta vs precios originales.
    
    Args:
        recipe_prices: DataFrame con precios por receta
        detalle_df: DataFrame con precios originales
        
    Returns:
        DataFrame con comparación
    """
    # Verificar que recipe_prices tenga la columna correcta
    if "MMPP_Fruta_Calculado" not in recipe_prices.columns:
        st.error("❌ **ERROR**: recipe_prices no tiene columna 'MMPP_Fruta_Calculado'")
        return pd.DataFrame()
    
    # Verificar que recipe_prices no esté vacío
    if recipe_prices.empty:
        st.warning("⚠️ **ADVERTENCIA**: recipe_prices está vacío")
        return pd.DataFrame()
    
    # Buscar columna de MMPP en detalle_df
    mmpp_columns = [col for col in detalle_df.columns if "MMPP" in col and "Fruta" in col]
    
    if not mmpp_columns:
        st.warning("⚠️ **ADVERTENCIA**: No se encontró columna de MMPP (Fruta) en detalle_df")
        return pd.DataFrame()
    
    # Usar la primera columna de MMPP Fruta encontrada
    mmpp_column = mmpp_columns[0]
    
    # Unir por SKU
    comparison = recipe_prices.merge(
        detalle_df[["SKU", mmpp_column]], 
        on="SKU", 
        how="left"
    )
    
    # Calcular diferencias usando la columna correcta
    comparison["Precio_Original"] = comparison[mmpp_column]
    comparison["Precio_Receta"] = comparison["MMPP_Fruta_Calculado"]
    comparison["Diferencia"] = comparison["Precio_Receta"] - comparison["Precio_Original"]
    comparison["Diferencia_Pct"] = np.where(
        comparison["Precio_Original"] > 0,
        (comparison["Diferencia"] / comparison["Precio_Original"]) * 100,
        0
    )
    
    # Clasificar diferencias
    comparison["Clasificacion"] = np.where(
        abs(comparison["Diferencia_Pct"]) <= 5,
        "Similar (±5%)",
        np.where(
            comparison["Diferencia_Pct"] > 5,
            "Receta más cara",
            "Receta más barata"
        )
    )
    
    return comparison

def simulate_fruit_price_change(
    recipe_df: pd.DataFrame,
    fruit_prices: Dict[str, float],
    fruit_efficiency: Dict[str, float],
    fruta_id: str,
    price_change_pct: float,
    fruit_names: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Simula el cambio en el precio de una fruta específica y calcula el impacto en todos los SKUs.
    
    Args:
        recipe_df: DataFrame con recetas
        fruit_prices: Diccionario con precios de frutas usando fruta_id como clave
        fruit_efficiency: Diccionario con eficiencias de frutas usando fruta_id como clave
        fruta_id: ID de la fruta cuyo precio se va a cambiar
        price_change_pct: Cambio de precio en porcentaje (ej: 10.0 para +10%, -5.0 para -5%)
        fruit_names: Diccionario con nombres de frutas usando fruta_id como clave
        
    Returns:
        DataFrame con comparación de precios antes y después del cambio
    """
    # Crear copia de precios con el cambio aplicado
    new_fruit_prices = fruit_prices.copy()
    if fruta_id in new_fruit_prices:
        current_price = new_fruit_prices[fruta_id]
        new_price = current_price * (1 + price_change_pct / 100)
        new_fruit_prices[fruta_id] = new_price
    
    # Calcular precios originales
    original_prices = calculate_all_recipe_prices(recipe_df, fruit_prices, fruit_efficiency, fruit_names)
    
    # Calcular precios con el cambio
    new_recipe_prices = calculate_all_recipe_prices(recipe_df, new_fruit_prices, fruit_efficiency, fruit_names)
    
    if original_prices.empty or new_recipe_prices.empty:
        return pd.DataFrame()
    
    # Combinar resultados para comparación
    comparison = original_prices.merge(
        new_recipe_prices[["SKU", "MMPP_Fruta_Calculado"]], 
        on="SKU", 
        suffixes=("_Original", "_Modificado")
    )
    
    # Calcular diferencias
    comparison["Cambio_MMPP"] = comparison["MMPP_Fruta_Calculado_Modificado"] - comparison["MMPP_Fruta_Calculado_Original"]
    comparison["Cambio_MMPP_Pct"] = np.where(
        comparison["MMPP_Fruta_Calculado_Original"] > 0,
        (comparison["Cambio_MMPP"] / comparison["MMPP_Fruta_Calculado_Original"]) * 100,
        0.0
    )
    
    # Agregar información sobre la fruta modificada
    if fruit_names:
        fruit_name = fruit_names.get(fruta_id, fruta_id)
        comparison["Fruta_Modificada"] = f"{fruit_name} ({fruta_id})"
        comparison["Nombre_Fruta_Modificada"] = fruit_name
    else:
        comparison["Fruta_Modificada"] = fruta_id
        comparison["Nombre_Fruta_Modificada"] = fruta_id
    
    # Agregar información del cambio de precio
    if fruta_id in fruit_prices:
        current_price = fruit_prices[fruta_id]
        new_price = new_fruit_prices[fruta_id]
        comparison["Precio_Original"] = current_price
        comparison["Precio_Nuevo"] = new_price
        comparison["Cambio_Precio_Pct"] = price_change_pct
    
    # Ordenar por impacto del cambio
    comparison = comparison.sort_values("Cambio_MMPP", ascending=False)
    
    return comparison

def generate_recipe_analysis_report(
    recipe_df: pd.DataFrame,
    fruit_prices: Dict[str, float],
    fruit_efficiency: Dict[str, float],
    fruit_names: Dict[str, str] = None
) -> str:
    """
    Genera un reporte de análisis de las recetas usando fruta_id.
    
    Args:
        recipe_df: DataFrame con recetas
        fruit_prices: Diccionario con precios de frutas usando fruta_id como clave
        fruit_efficiency: Diccionario con eficiencias de frutas usando fruta_id como clave
        fruit_names: Diccionario con nombres de frutas usando fruta_id como clave
        
    Returns:
        String con el reporte de análisis
    """
    if recipe_df.empty:
        return "❌ No hay datos de recetas para analizar."
    
    # Estadísticas básicas
    total_skus = recipe_df["SKU"].nunique()
    total_recipes = len(recipe_df)
    
    # Validar recetas
    valid_recipes = recipe_df[recipe_df["Porcentaje_Valido"] == True]
    invalid_recipes = recipe_df[recipe_df["Porcentaje_Valido"] == False]
    
    # Análisis por fruta (agrupando por fruta_id)
    fruit_analysis = []
    for fruta_id in recipe_df["fruta_id"].unique():
        if pd.notna(fruta_id):
            fruta_recipes = recipe_df[recipe_df["fruta_id"] == fruta_id]
            total_percentage = fruta_recipes["Porcentaje"].sum()
            avg_percentage = fruta_recipes["Porcentaje"].mean()
            min_percentage = fruta_recipes["Porcentaje"].min()
            max_percentage = fruta_recipes["Porcentaje"].max()
            num_skus = fruta_recipes["SKU"].nunique()
            
            fruit_name = fruit_names.get(fruta_id, fruta_id) if fruit_names else fruta_id
            
            fruit_analysis.append({
                "fruta_id": fruta_id,
                "nombre": fruit_name,
                "total_porcentaje": total_percentage,
                "porcentaje_promedio": avg_percentage,
                "porcentaje_min": min_percentage,
                "porcentaje_max": max_percentage,
                "num_skus": num_skus
            })
    
    # Ordenar por número de SKUs
    fruit_analysis.sort(key=lambda x: x["num_skus"], reverse=True)
    
    # Generar reporte
    report = f"""
# 📊 Reporte de Análisis de Recetas

## 📊 Estadísticas Generales
- **Total de SKUs únicos**: {total_skus}
- **Total de recetas**: {total_recipes}
- **Promedio de recetas por SKU**: {total_recipes/total_skus:.2f}

## ✅ Validación de Recetas
- **Recetas válidas (0-100%)**: {len(valid_recipes)} ({len(valid_recipes)/len(recipe_df)*100:.1f}%)
- **Recetas con porcentajes fuera de rango**: {len(invalid_recipes)} ({len(invalid_recipes)/len(recipe_df)*100:.1f}%)

## 🍎 Análisis por Fruta
"""
    
    for fruit in fruit_analysis[:10]:  # Mostrar top 10
        report += f"""
### {fruit['nombre']} ({fruit['fruta_id']})
- **SKUs que la usan**: {fruit['num_skus']}
- **Porcentaje total**: {fruit['total_porcentaje']:.1f}%
- **Porcentaje promedio**: {fruit['porcentaje_promedio']:.1f}%
- **Rango**: {fruit['porcentaje_min']:.1f}% - {fruit['porcentaje_max']:.1f}%
"""
    
    if len(fruit_analysis) > 10:
        report += f"\n... y {len(fruit_analysis) - 10} frutas más\n"
    
    # Análisis de cobertura de precios
    frutas_con_precio = len([f for f in fruit_analysis if f["fruta_id"] in fruit_prices])
    frutas_sin_precio = len(fruit_analysis) - frutas_con_precio
    
    report += f"""
## 💰 Cobertura de Precios
- **Frutas con precio disponible**: {frutas_con_precio} ({frutas_con_precio/len(fruit_analysis)*100:.1f}%)
- **Frutas sin precio**: {frutas_sin_precio} ({frutas_sin_precio/len(fruit_analysis)*100:.1f}%)

## 🔧 Recomendaciones
"""
    
    if len(invalid_recipes) > 0:
        report += "- ⚠️ **Revisar porcentajes**: Hay recetas con porcentajes fuera del rango 0-100%\n"
    
    if frutas_sin_precio > 0:
        report += "- ⚠️ **Completar precios**: Agregar precios para las frutas faltantes en INFO_FRUTA\n"
    
    if total_recipes/total_skus < 1.5:
        report += "- ℹ️ **Recetas simples**: La mayoría de SKUs tienen recetas con una sola fruta\n"
    else:
        report += "- ℹ️ **Recetas complejas**: Muchos SKUs tienen recetas con múltiples frutas\n"
    
    return report

# ===================== FUNCIÓN PRINCIPAL DE INTEGRACIÓN =====================

def process_recipe_data(sheets: Dict[str, pd.DataFrame], detalle_df: pd.DataFrame) -> Dict:
    """
    Función principal que procesa todos los datos de recetas y frutas.
    
    Args:
        sheets: Diccionario con todas las hojas del Excel
        detalle_df: DataFrame con datos de detalle
        
    Returns:
        Diccionario con todos los datos procesados
    """
    try:
        # 1. Cargar información de frutas desde INFO_FRUTA
        fruit_prices, fruit_efficiency, fruit_names = load_fruit_info(sheets)
        
        # 2. Si no hay precios desde INFO_FRUTA, extraer desde detalle como respaldo
        if not fruit_prices:
            st.warning("⚠️ Extrayendo precios de frutas desde datos de detalle como respaldo...")
            fruit_prices, fruit_names = extract_fruit_prices_from_detalle(detalle_df)
        else:
            pass
        
        # 3. Cargar datos de recetas
        recipe_df = load_recipe_data(sheets)
        
        # 4. Calcular precios por receta
        # recipe_prices = calculate_all_recipe_prices(recipe_df, fruit_prices, fruit_efficiency, fruit_names)
        
        # 5. Comparar con precios originales
        # comparison = compare_recipe_vs_original_prices(recipe_prices, detalle_df)
        
        # 6. Generar reporte
        # report = generate_recipe_analysis_report(recipe_df, fruit_prices, fruit_efficiency, fruit_names)
        
        return {
            "success": True,
            "recipe_df": recipe_df,
            "fruit_prices": fruit_prices,
            "fruit_efficiency": fruit_efficiency,
            "fruit_names": fruit_names,
            # "recipe_prices": recipe_prices,
            # "comparison": comparison,
            # "report": report,
            "source": "INFO_FRUTA" if "INFO_FRUTA" in sheets else "Detalle"
        }
        
    except Exception as e:
        st.error(f"❌ Error procesando datos de recetas: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ===================== FUNCIONES DE RESPALDO (mantenidas para compatibilidad) =====================

def extract_fruit_prices_from_detalle(detalle_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Extrae precios de frutas base desde el DataFrame de detalle.
    FUNCIÓN DE RESPALDO - Se usa solo si INFO_FRUTA no está disponible.
    
    Args:
        detalle_df: DataFrame con datos de detalle
        
    Returns:
        Tuple con (precios_frutas, nombres_frutas) usando fruta_id como clave
    """
    fruit_prices = {}
    fruit_names = {}
    
    # Buscar precios de frutas base en el detalle
    # Como no tenemos INFO_FRUTA, creamos fruta_id genéricos basados en nombres de productos
    
    # Buscar SKUs que contengan palabras clave de frutas
    fruit_keywords = ["arandano", "frutilla", "mora", "frambuesa", "mango", "piña", "banana"]
    
    for keyword in fruit_keywords:
        # Buscar SKUs que contengan la palabra clave
        fruit_skus = detalle_df[
            detalle_df["Descripcion"].str.contains(keyword, case=False, na=False)
        ]
        
        if not fruit_skus.empty:
            # Tomar el precio del primer SKU encontrado
            if "PrecioVenta (USD/kg)" in fruit_skus.columns:
                price = fruit_skus.iloc[0]["PrecioVenta (USD/kg)"]
                # Crear un fruta_id genérico
                fruta_id = f"fruta_{keyword.upper()}"
                fruit_prices[fruta_id] = price
                # Crear nombre en inglés basado en la palabra clave
                fruit_names[fruta_id] = keyword.title()
        else:
            pass
    
    # Si no se encontraron frutas específicas, crear un precio genérico
    if not fruit_prices:
        # Crear un precio genérico para evitar errores
        fruit_prices["fruta_generica"] = 1.0
        fruit_names["fruta_generica"] = "Generic Fruit"
    
    return fruit_prices, fruit_names
