"""
Simulador de costos de fruta para el simulador de EBITDA.
Calcula MMPP (Fruta) por SKU basado en recetas y precios de fruta.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.data_io import recalculate_totals


def validate_fruit_inputs(receta_df: pd.DataFrame, info_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida que los DataFrames de entrada tengan las columnas requeridas y rangos válidos.
    
    Args:
        receta_df: DataFrame con columnas [SKU, Fruta_id, Porcentaje]
        info_df: DataFrame con columnas [Fruta_id, Precio, Eficiencia, Name]
        
    Returns:
        Tuple[bool, str]: (es_válido, mensaje_error)
    """
    # Verificar columnas requeridas en receta_df
    required_receta_cols = ["SKU", "Fruta_id", "Porcentaje"]
    missing_receta_cols = [col for col in required_receta_cols if col not in receta_df.columns]
    if missing_receta_cols:
        return False, f"Columnas faltantes en RECETA_SKU: {missing_receta_cols}"
    
    # Verificar columnas requeridas en info_df
    required_info_cols = ["Fruta_id", "Precio", "Eficiencia", "Name"]
    missing_info_cols = [col for col in required_info_cols if col not in info_df.columns]
    if missing_info_cols:
        return False, f"Columnas faltantes en INFO_FRUTA: {missing_info_cols}"
    
    # Validar rangos en receta_df
    if (receta_df["Porcentaje"].astype(float) < 0).any():
        return False, "Porcentaje no puede ser negativo en RECETA_SKU"
    
    # Validar rangos en info_df
    if (info_df["Precio"].astype(float) < 0).any():
        return False, "Precio no puede ser negativo en INFO_FRUTA"
    
    if (info_df["Eficiencia"].astype(float) <= 0).any() or (info_df["Eficiencia"].astype(float) > 1).any():
        return False, "Eficiencia debe estar en (0, 1] en INFO_FRUTA"
    
    return True, "Datos válidos"


def get_adjusted_fruit_params(info_df: pd.DataFrame, fruit_overrides: Dict) -> pd.DataFrame:
    """
    Aplica overrides de PRECIO por fruta.
    Overrides esperados por fruta_id:
    {"price": {"type": "percentage"|"dollars", "value": float}}
    {"rendimiento": {"type": "absolute", "value": float}}
    
    Args:
        info_df: DataFrame base con [Fruta_id, Precio, Rendimiento, Name]
        fruit_overrides: Diccionario de overrides por Fruta_id
        
    Returns:
        DataFrame con columnas:
        [Fruta_id, FrutaNombre, PrecioBaseUSD_kg, RendimientoBase, PrecioAjustadoUSD_kg, RendimientoAjustado, CostoEfectivoBase, CostoEfectivoAjustado]
    """
    params = info_df.copy()
    params = params.rename(columns={
        "Precio": "PrecioBaseUSD_kg",
        "Rendimiento": "RendimientoBase"
    })
    if "Name" not in params.columns:
        params["FrutaNombre"] = params["Fruta_id"]
    else:
        params["FrutaNombre"] = params["Name"]

    # Sanitizar base
    params["PrecioBaseUSD_kg"] = pd.to_numeric(params["PrecioBaseUSD_kg"], errors="coerce").fillna(0.0).clip(lower=0.0)
    params["RendimientoBase"] = pd.to_numeric(params["RendimientoBase"], errors="coerce").fillna(1.0).clip(0.01, 1.0)
    params["CostoEfectivoBase"] = params["PrecioBaseUSD_kg"] / params["RendimientoBase"]


    # Precio ajustado = base por defecto
    params["PrecioAjustadoUSD_kg"] = params["PrecioBaseUSD_kg"]
    params["RendimientoAjustado"] = params["RendimientoBase"]
    params["CostoEfectivoAjustado"] = params["CostoEfectivoBase"]

    overrides = fruit_overrides or {}
    if overrides:
        for fruta_id, ov in overrides.items():
            if not isinstance(ov, dict): 
                continue
            price_ov = ov.get("price")
            rendimiento_ov = ov.get("rendimiento")
            if price_ov:
                mask = params["Fruta_id"] == fruta_id
                if price_ov.get("type") == "percentage":
                    pct = float(price_ov.get("value", 0.0))
                    params.loc[mask, "PrecioAjustadoUSD_kg"] = (
                        params.loc[mask, "PrecioBaseUSD_kg"] * (1.0 + pct/100.0)
                    )
                    params.loc[mask, "CostoEfectivoAjustado"] = params.loc[mask, "PrecioAjustadoUSD_kg"] / params.loc[mask, "RendimientoAjustado"]
                elif price_ov.get("type") == "dollars":
                    val = max(0.0, float(price_ov.get("value", 0.0)))
                    params.loc[mask, "PrecioAjustadoUSD_kg"] = val
                    params.loc[mask, "CostoEfectivoAjustado"] = params.loc[mask, "PrecioAjustadoUSD_kg"] / params.loc[mask, "RendimientoAjustado"]
            if rendimiento_ov:
                mask = params["Fruta_id"] == fruta_id
                if rendimiento_ov.get("type") == "absolute":
                    val = max(0.01, float(rendimiento_ov.get("value", 1.0)))
                    params.loc[mask, "RendimientoAjustado"] = val
                    params.loc[mask, "CostoEfectivoAjustado"] = params.loc[mask, "PrecioAjustadoUSD_kg"] / params.loc[mask, "RendimientoAjustado"]

    # Clips defensivos
    params["PrecioAjustadoUSD_kg"] = params["PrecioAjustadoUSD_kg"].clip(lower=0.0)
    params["RendimientoAjustado"] = params["RendimientoAjustado"].clip(0.01, 1.0)
    params["CostoEfectivoAjustado"] = params["CostoEfectivoAjustado"].clip(lower=0.0)
    
    return params[[
        "Fruta_id", "FrutaNombre", "PrecioBaseUSD_kg", "RendimientoBase",
        "PrecioAjustadoUSD_kg", "RendimientoAjustado", "CostoEfectivoBase", "CostoEfectivoAjustado"
    ]]


def compute_mmpp_fruta_per_sku(receta_df: pd.DataFrame, params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une receta (largo) con params ajustados y calcula:
      contrib_pos = PrecioAjustadoUSD_kg * Porcentaje / RendimientoAjustado
    MMPP (Fruta) (USD/kg) por SKU = - SUMA(contrib_pos)  (negativo)
    
    Args:
        receta_df: DataFrame con [SKU, Fruta_id, Porcentaje]
        params_df: DataFrame con parámetros ajustados de fruta
        
    Returns:
        DataFrame con [SKU, MMPP (Fruta) (USD/kg)] (valores negativos para costos)
    """
    r = receta_df.copy()
    r["Óptimo"] = pd.to_numeric(r["Óptimo"], errors="coerce").fillna(0.0).clip(lower=0.0)
    
    merged = r.merge(params_df, on="Fruta_id", how="left")
    merged["contrib_pos"] = (
        pd.to_numeric(merged["PrecioAjustadoUSD_kg"], errors="coerce").fillna(0.0) *
        (merged["Óptimo"] /100) / 
        pd.to_numeric(merged["RendimientoAjustado"], errors="coerce").fillna(1.0).replace(0, 0.01))
    
    per_sku = merged.groupby("SKU", as_index=False)["contrib_pos"].sum()
    per_sku["MMPP (Fruta) (USD/kg)"] = -per_sku["contrib_pos"]
    
    return per_sku[["SKU", "MMPP (Fruta) (USD/kg)"]]


def apply_fruit_overrides_to_sim(df_sim, receta_df, info_df, fruit_overrides):
    df_out = df_sim.copy()

    # 1) Construir params de fruta con overrides ya aplicados (precio/eficiencia)
    params = get_adjusted_fruit_params(info_df, fruit_overrides)  # debe devolver PrecioAjustadoUSD_kg y EficienciaAjustada

    # 2) Recalcular MMPP (Fruta) y Almacenaje MMPP por SKU con esos params
    mmpp_df = compute_mmpp_fruta_per_sku(receta_df, params)  # devuelve columnas: ["SKU","MMPP (Fruta) (USD/kg)","Almacenaje MMPP"]

    # 3) Merge SIN crear sufijos… actualizando columnas en su nombre canónico
    for col in ["MMPP (Fruta) (USD/kg)"]:
        if col in mmpp_df.columns:
            # aseguramos signo: costos siempre NEGATIVOS
            mmpp_df[col] = -mmpp_df[col].abs()

            # actualizamos por map en vez de merge con sufijo
            m = mmpp_df.set_index("SKU")[col]
            df_out[col] = df_out["SKU"].astype(str).map(m.reindex(m.index.astype(str))).fillna(df_out.get(col))

    # 4) Recalcular totales con la MISMA definición que usas en build_detalle()
    df_out = recalculate_totals(df_out)
    return df_out


def get_fruit_summary_table(info_df: pd.DataFrame, 
                           receta_df: pd.DataFrame, 
                           fruit_overrides: Dict,
                           skus_visibles: list = None) -> pd.DataFrame:
    """
    Genera tabla resumen de frutas con información de ajustes y SKUs afectados.
    
    Args:
        info_df: DataFrame de información de fruta
        receta_df: DataFrame de recetas
        fruit_overrides: Diccionario de overrides
        skus_visibles: Lista de SKUs visibles (para respetar filtros)
        
    Returns:
        DataFrame resumen con columnas:
        [Fruta_id, FrutaNombre, PrecioBase, PrecioAjustado, RendimientoBase, SKUsAfectados, Contrib_total_USDkg]
    """
    # Filtrar por SKUs visibles si se especifica
    if skus_visibles:
        receta_filtrada = receta_df[receta_df["SKU"].isin(skus_visibles)].copy()
    else:
        receta_filtrada = receta_df.copy()
    
    # Obtener parámetros ajustados
    params_df = get_adjusted_fruit_params(info_df, fruit_overrides)
    
    # Contar SKUs afectados por fruta
    skus_por_fruta = receta_filtrada.groupby("Fruta_id")["SKU"].nunique().reset_index()
    skus_por_fruta = skus_por_fruta.rename(columns={"SKU": "SKUsAfectados"})
    
    # Calcular contribuciones
    receta_con_params = receta_filtrada.merge(
        params_df[["Fruta_id", "PrecioAjustadoUSD_kg", "RendimientoAjustado"]], 
        on="Fruta_id", how="left"
    )
    receta_con_params["contrib_pos"] = (
        receta_con_params["PrecioAjustadoUSD_kg"] * 
        receta_con_params["Óptimo"] / 
        receta_con_params["RendimientoAjustado"]
    )
    
    contrib_por_fruta = receta_con_params.groupby("Fruta_id")["contrib_pos"].sum().reset_index()
    contrib_por_fruta = contrib_por_fruta.rename(columns={"contrib_pos": "Contrib_total_USDkg"})
    
    # Crear tabla resumen
    summary = params_df.merge(skus_por_fruta, on="Fruta_id", how="left")
    summary = summary.merge(contrib_por_fruta, on="Fruta_id", how="left")
    
    # Llenar valores vacíos
    summary["SKUsAfectados"] = summary["SKUsAfectados"].fillna(0)
    summary["Contrib_total_USDkg"] = summary["Contrib_total_USDkg"].fillna(0.0)
    
    return summary


def validate_bulk_upload_df(upload_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida DataFrame de carga masiva.
    
    Args:
        upload_df: DataFrame con columnas esperadas
        
    Returns:
        Tuple[bool, str]: (es_válido, mensaje_error)
    """
    # Verificar columna obligatoria
    if "Fruta_id" not in upload_df.columns:
        return False, "Columna 'Fruta_id' es obligatoria"
    
    # Verificar que al menos una columna de ajuste esté presente
    if "PrecioNuevo" not in upload_df.columns and "EficienciaPct" not in upload_df.columns:
        return False, "Debe incluir al menos 'PrecioNuevo' o 'EficienciaPct'"
    
    # Validar rangos si están presentes
    if "PrecioNuevo" in upload_df.columns:
        if (upload_df["PrecioNuevo"] < 0).any():
            return False, "PrecioNuevo no puede ser negativo"
    
    if "EficienciaPct" in upload_df.columns:
        # EficienciaPct es variación nominal %, no necesita validación de rango específico
        pass
    
    return True, "Datos válidos"


def process_bulk_upload(upload_df: pd.DataFrame, 
                       current_overrides: Dict,
                       info_df: pd.DataFrame) -> Tuple[Dict, str]:
    """
    Procesa carga masiva y retorna overrides actualizados.
    
    Args:
        upload_df: DataFrame de carga masiva
        current_overrides: Overrides actuales
        info_df: DataFrame de información de fruta
        
    Returns:
        Tuple[Dict, str]: (overrides_actualizados, mensaje_resultado)
    """
    # Crear copia de overrides actuales
    updated_overrides = current_overrides.copy()
    
    # Procesar cada fila
    processed_count = 0
    errors = []
    
    for _, row in upload_df.iterrows():
        fruta_id = row["Fruta_id"]
        
        # Verificar que la fruta existe
        if fruta_id not in info_df["Fruta_id"].values:
            errors.append(f"Fruta_id '{fruta_id}' no encontrada en INFO_FRUTA")
            continue
        
        # Inicializar override para esta fruta si no existe
        if fruta_id not in updated_overrides:
            updated_overrides[fruta_id] = {}
        
        # Procesar precio si está presente
        if "PrecioNuevo" in row and pd.notna(row["PrecioNuevo"]):
            updated_overrides[fruta_id]["price"] = {
                "type": "dollars",
                "value": float(row["PrecioNuevo"])
            }
        
        # Procesar eficiencia si está presente
        if "EficienciaPct" in row and pd.notna(row["EficienciaPct"]):
            updated_overrides[fruta_id]["efficiency"] = {
                "type": "percentage",
                "value": float(row["EficienciaPct"])
            }
        
        # Agregar timestamp
        updated_overrides[fruta_id]["timestamp"] = pd.Timestamp.now()
        
        processed_count += 1
    
    # Generar mensaje de resultado
    if errors:
        result_msg = f"✅ {processed_count} frutas procesadas. ⚠️ Errores: {'; '.join(errors)}"
    else:
        result_msg = f"✅ {processed_count} frutas procesadas exitosamente"
    
    return updated_overrides, result_msg
