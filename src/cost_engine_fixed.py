"""
Motor de Costos - API funcional para cálculo de costos Retail y Granel
Implementa una API testeable para calcular conceptos de costos desde un Excel unificado.
"""

from typing import Dict, Union, Optional, Literal
import pandas as pd
import numpy as np
import io


def read_source(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Lee todas las hojas del archivo Excel desde bytes.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        Diccionario con nombre_hoja -> DataFrame
        
    Raises:
        ValueError: Si el archivo no se puede leer
    """
    # try:  # Comentado para debugging
    # Crear BytesIO desde los bytes
    bio = io.BytesIO(uploaded_bytes)
    
    # Leer todas las hojas del Excel
    xls = pd.ExcelFile(bio, engine='openpyxl')
    dfs = {name: xls.parse(name) for name in xls.sheet_names}
    
    # Convertir nombres de hojas a mayúsculas para consistencia
    normalized_dfs = {}
    for sheet_name, df in dfs.items():
        normalized_dfs[sheet_name.upper()] = df
        
    return normalized_dfs
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error al leer el archivo Excel: {str(e)}")


def validate_inputs(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Valida que las hojas requeridas existan y tengan las columnas mínimas.
    
    Args:
        dfs: Diccionario de DataFrames por hoja
        
    Raises:
        ValueError: Si faltan hojas o columnas requeridas
    """
    # Hojas requeridas (obligatorias)
    required_sheets = {
        'MAYOR': ['familia_cc', 'mes', 'monto'],
        'INDICADORES_RETAIL': ['SKU', 'mes', 'kg_producidos', 'kg_despachados'],
        'INDICADORES_GRANEL': ['mes', 'kg_producidos'],
        'RECETAS': [],
        'FRUTA': [],
        'CONFIG_SPLITS': ['concepto']
    }
    
    missing_sheets = []
    missing_columns = []
    
    # Verificar hojas requeridas
    for sheet_name, required_cols in required_sheets.items():
        if sheet_name not in dfs:
            missing_sheets.append(sheet_name)
        else:
            df = dfs[sheet_name]
            missing_cols_in_sheet = [col for col in required_cols if col not in df.columns]
            if missing_cols_in_sheet:
                missing_columns.append(f"{sheet_name}: {missing_cols_in_sheet}")
    
    # Construir mensaje de error si hay problemas
    error_messages = []
    if missing_sheets:
        error_messages.append(f"Hojas faltantes: {missing_sheets}")
    if missing_columns:
        error_messages.append(f"Columnas faltantes: {missing_columns}")
    
    if error_messages:
        raise ValueError(". ".join(error_messages))


def build_detalle_from_cost_engine(uploaded_bytes: bytes) -> pd.DataFrame:
    """
    Construye detalle usando cost_engine con integración de data_io.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        DataFrame con detalle completo y funcional
    """
    # try:  # Comentado para debugging
    # Importar funciones clave de data_io
    from src.data_io import (
        read_workbook, build_tbl_costos_pond, build_fact_precios, 
        compute_latest_price, build_dim_sku, build_fact_volumen,
        compute_mmpp_unified, correct_species_from_recipes, 
        ensure_list_species, recalculate_totals, load_receta_sku, 
        load_info_fruta
    )
    
    # 1. Leer todas las hojas usando la función probada de data_io
    sheets = read_workbook(uploaded_bytes)
    
    # 2. Intentar usar cost_engine para costos con promedio móvil (si hay datos suficientes)
    resultados_cost_engine = None
    # try:  # Comentado para debugging
    #     resultados_cost_engine = compute_full_cost_analysis(uploaded_bytes)
    # except Exception:
    #     # Si falla el cost_engine, continuar con data_io solamente
    #     pass
    
    # 3. Construir costos base usando data_io (funcionalidad probada)
    costos_detalle = None
    
    # Prioridad 1: FACT_COSTOS_POND (datos históricos)
    if 'FACT_COSTOS_POND' in sheets and not sheets['FACT_COSTOS_POND'].empty:
        # try:  # Comentado para debugging
        costos_detalle = build_tbl_costos_pond(sheets["FACT_COSTOS_POND"])
        # except Exception:
        #     pass
    
    # Prioridad 2: Resultados del cost_engine (si están disponibles)
    if costos_detalle is None and resultados_cost_engine and 'retail' in resultados_cost_engine:
        # try:  # Comentado para debugging
        df_retail = resultados_cost_engine['retail']
        costos_detalle = df_retail.reset_index()
        costos_detalle = costos_detalle.rename(columns={'index': 'SKU'})
        
        # Renombrar columnas del cost_engine al formato de data_io
        column_mapping = {
            'Costos_Totales': 'Costos Totales (USD/kg)',
            'Costos_Directos': 'Retail Costos Directos (USD/kg)',
            'Costos_Indirectos': 'Retail Costos Indirectos (USD/kg)',
            'MMPP_Fruta': 'MMPP (Fruta) (USD/kg)',
            'Proceso_Granel': 'Proceso Granel (USD/kg)',
            'Materiales_Directos': 'Materiales Directos',
            'Materiales_Indirectos': 'Materiales Indirectos',
            'Almacenaje_MMPP': 'Almacenaje MMPP',
            'Servicios_Generales': 'Servicios Generales',
            'Guarda_PT': 'Guarda PT'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in costos_detalle.columns:
                costos_detalle = costos_detalle.rename(columns={old_col: new_col})
        # except Exception:
        #     pass
    
    # Prioridad 3: Crear estructura básica desde DIM_SKU si no hay costos
    if costos_detalle is None:
        if 'DIM_SKU' in sheets and not sheets['DIM_SKU'].empty:
            # Crear estructura básica desde DIM_SKU
            costos_detalle = sheets['DIM_SKU'][['SKU']].drop_duplicates().copy()
            # Agregar columnas de costos con valores 0
            cost_cols = ['MMPP (Fruta) (USD/kg)', 'Proceso Granel (USD/kg)', 'Almacenaje MMPP',
                       'Materiales Directos', 'Materiales Indirectos', 'Servicios Generales']
            for col in cost_cols:
                costos_detalle[col] = 0.0
        else:
            # Último recurso: crear desde precios si existen
            if 'PRECIOS' in sheets and not sheets['PRECIOS'].empty:
                skus_unicos = sheets['PRECIOS']['SKU'].unique() if 'SKU' in sheets['PRECIOS'].columns else []
                if len(skus_unicos) > 0:
                    costos_detalle = pd.DataFrame({'SKU': skus_unicos})
                    cost_cols = ['MMPP (Fruta) (USD/kg)', 'Proceso Granel (USD/kg)', 'Almacenaje MMPP']
                    for col in cost_cols:
                        costos_detalle[col] = 0.0
                else:
                    raise ValueError("No se encontraron datos suficientes para construir el detalle")
            else:
                raise ValueError("No se encontraron datos suficientes para construir el detalle")
    
    # 4. Procesar precios usando la lógica probada de data_io
    if 'PRECIOS' in sheets:
        precios = build_fact_precios(sheets["PRECIOS"])
        latest_prices = compute_latest_price(precios, mode="global")
    elif 'FACT_PRECIOS' in sheets:
        precios = build_fact_precios(sheets["FACT_PRECIOS"])
        latest_prices = compute_latest_price(precios, mode="global")
    else:
        latest_prices = pd.DataFrame(columns=['SKU-Cliente', 'PrecioVenta (USD/kg)'])
    
    # 5. Procesar dimensiones usando data_io
    if 'DIM_SKU' in sheets:
        dim = build_dim_sku(sheets["DIM_SKU"])
    else:
        # Crear dimensiones básicas desde INDICADORES_RETAIL si existe
        dim = pd.DataFrame(columns=['SKU', 'SKU-Cliente', 'Descripcion', 'Marca', 'Cliente', 'Especie', 'Condicion'])
        if 'INDICADORES_RETAIL' in sheets:
            retail_info = sheets['INDICADORES_RETAIL']
            if 'SKU' in retail_info.columns and 'SKU-Cliente' in retail_info.columns:
                dim = retail_info[['SKU', 'SKU-Cliente']].drop_duplicates()
                # Agregar columnas faltantes
                for col in ['Descripcion', 'Marca', 'Cliente', 'Especie', 'Condicion']:
                    if col not in dim.columns:
                        dim[col] = ''
    
    # 6. Procesar volúmenes usando data_io
    if 'VOLÚMENES' in sheets:
        volumenes = build_fact_volumen(sheets["VOLÚMENES"])
        # Tomar el último mes disponible
        if not volumenes.empty:
            ultimo_mes = volumenes['FechaClave'].max()
            volumenes_recientes = volumenes[volumenes['FechaClave'] == ultimo_mes]
        else:
            volumenes_recientes = volumenes
    elif 'FACT_VOL' in sheets:
        volumenes = build_fact_volumen(sheets["FACT_VOL"])
        if not volumenes.empty:
            ultimo_mes = volumenes['FechaClave'].max()
            volumenes_recientes = volumenes[volumenes['FechaClave'] == ultimo_mes]
        else:
            volumenes_recientes = volumenes
    else:
        volumenes_recientes = pd.DataFrame(columns=['SKU', 'SKU-Cliente', 'KgEmbarcados'])
    
    # 7. Calcular MMPP usando la función probada de data_io
    if 'RECETAS' in sheets and 'FRUTA' in sheets:
        receta_df = load_receta_sku(sheets["RECETAS"]) if not sheets["RECETAS"].empty else pd.DataFrame()
        info_df = load_info_fruta(sheets["FRUTA"]) if not sheets["FRUTA"].empty else pd.DataFrame()
    elif 'RECETA_SKU' in sheets and 'INFO_FRUTA' in sheets:
        receta_df = load_receta_sku(sheets["RECETA_SKU"]) if not sheets["RECETA_SKU"].empty else pd.DataFrame()
        info_df = load_info_fruta(sheets["INFO_FRUTA"]) if not sheets["INFO_FRUTA"].empty else pd.DataFrame()
    else:
        receta_df = pd.DataFrame()
        info_df = pd.DataFrame()
    
    # Obtener datos de granel para MMPP
    if resultados_cost_engine and 'granel' in resultados_cost_engine:
        df_granel = resultados_cost_engine['granel']
    else:
        df_granel = pd.DataFrame()
    
    # Calcular MMPP usando la función unificada de data_io
    if not receta_df.empty and not info_df.empty:
        mmpp_almacenaje = compute_mmpp_unified(receta_df, info_df, df_granel)
        mmpp_almacenaje = mmpp_almacenaje.rename(columns={
            "MMPP (Fruta) (USD/kg)": "MMPP (Fruta) (USD/kg) (Calculado)",
            "Almacenaje": "Almacenaje (Calculado)", 
            "Proceso Granel (USD/kg)": "Proceso Granel (USD/kg) (Calculado)"
        })
    else:
        mmpp_almacenaje = pd.DataFrame()
    
    # 8. Unir todas las tablas usando la lógica de data_io
    if not mmpp_almacenaje.empty:
        # Merge con MMPP calculado
        detalle = costos_detalle.merge(mmpp_almacenaje, on="SKU", how="left")
        
        # Usar valores calculados si están disponibles, sino usar los del cost_engine
        if "MMPP (Fruta) (USD/kg) (Calculado)" in detalle.columns:
            detalle["MMPP (Fruta) (USD/kg)"] = detalle["MMPP (Fruta) (USD/kg) (Calculado)"].fillna(
                detalle.get("MMPP (Fruta) (USD/kg)", 0)
            )
            detalle = detalle.drop(columns=["MMPP (Fruta) (USD/kg) (Calculado)"])
        
        if "Almacenaje (Calculado)" in detalle.columns:
            detalle["Almacenaje MMPP"] = detalle["Almacenaje (Calculado)"].fillna(
                detalle.get("Almacenaje MMPP", 0)
            )
            detalle = detalle.drop(columns=["Almacenaje (Calculado)"])
        
        if "Proceso Granel (USD/kg) (Calculado)" in detalle.columns:
            detalle["Proceso Granel (USD/kg)"] = detalle["Proceso Granel (USD/kg) (Calculado)"].fillna(
                detalle.get("Proceso Granel (USD/kg)", 0)
            )
            detalle = detalle.drop(columns=["Proceso Granel (USD/kg) (Calculado)"])
    else:
        detalle = costos_detalle
    
    # 9. Merge con dimensiones
    if not dim.empty:
        detalle = detalle.merge(dim, on="SKU", how="left")
    
    # 10. Corregir especies usando recetas (funcionalidad clave de data_io)
    if not receta_df.empty and not info_df.empty:
        detalle = correct_species_from_recipes(detalle, receta_df, info_df)
        detalle = ensure_list_species(detalle, "Especie")
    
    # 11. Merge con precios
    if not latest_prices.empty:
        if "SKU-Cliente" in dim.columns and "SKU-Cliente" in latest_prices.columns:
            detalle = detalle.merge(latest_prices, on="SKU-Cliente", how="left")
        else:
            # Fallback: merge por SKU si no hay SKU-Cliente
            if "SKU" in latest_prices.columns:
                latest_by_sku = latest_prices.groupby("SKU")["PrecioVenta (USD/kg)"].last().reset_index()
                detalle = detalle.merge(latest_by_sku, on="SKU", how="left")
    
    # 12. Merge con volúmenes
    if not volumenes_recientes.empty:
        if "SKU-Cliente" in volumenes_recientes.columns and "SKU-Cliente" in detalle.columns:
            detalle = detalle.merge(
                volumenes_recientes[["SKU-Cliente", "KgEmbarcados"]], 
                on="SKU-Cliente", how="left"
            )
        elif "SKU" in volumenes_recientes.columns:
            vol_by_sku = volumenes_recientes.groupby("SKU")["KgEmbarcados"].sum().reset_index()
            detalle = detalle.merge(vol_by_sku, on="SKU", how="left")
    
    # 13. Aplicar signos correctos y recalcular totales usando data_io
    detalle = recalculate_totals(detalle)
    
    return detalle
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error construyendo detalle desde cost_engine: {str(e)}")
