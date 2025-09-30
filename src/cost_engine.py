"""
Motor de Costos - API funcional para cálculo de costos Retail y Granel
Implementa una API testeable para calcular conceptos de costos desde un Excel unificado.
"""

from typing import Dict, Union, Optional, Literal
import pandas as pd
import numpy as np
import io
import streamlit as st

# Importar funciones de data_io
from src.data_io import recalculate_totals


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
    resultados_cost_engine = compute_full_cost_analysis(uploaded_bytes)
    df_granel = build_granel_from_cost_engine(uploaded_bytes)[1]
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
        # Asegurar que ambas columnas SKU tengan el mismo tipo de datos
        if 'SKU' in costos_detalle.columns and 'SKU' in mmpp_almacenaje.columns:
            costos_detalle['SKU'] = costos_detalle['SKU'].astype(str)
            mmpp_almacenaje['SKU'] = mmpp_almacenaje['SKU'].astype(str)
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
        # Asegurar que ambas columnas SKU tengan el mismo tipo de datos
        if 'SKU' in detalle.columns and 'SKU' in dim.columns:
            detalle['SKU'] = detalle['SKU'].astype(str)
            dim['SKU'] = dim['SKU'].astype(str)
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
                # Asegurar que ambas columnas SKU tengan el mismo tipo de datos
                if 'SKU' in detalle.columns and 'SKU' in latest_by_sku.columns:
                    detalle['SKU'] = detalle['SKU'].astype(str)
                    latest_by_sku['SKU'] = latest_by_sku['SKU'].astype(str)
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
            # Asegurar que ambas columnas SKU tengan el mismo tipo de datos
            if 'SKU' in detalle.columns and 'SKU' in vol_by_sku.columns:
                detalle['SKU'] = detalle['SKU'].astype(str)
                vol_by_sku['SKU'] = vol_by_sku['SKU'].astype(str)
            detalle = detalle.merge(vol_by_sku, on="SKU", how="left")
    
    # 13. Aplicar signos correctos y recalcular totales usando data_io
    detalle = recalculate_totals(detalle)    
    return detalle
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error construyendo detalle desde cost_engine: {str(e)}")


def build_drivers_retail(df_retail: pd.DataFrame, meses: list = None) -> Dict[str, pd.DataFrame]:
    """
    Construye drivers para distribución de costos en Retail por SKU usando DataFrames.
    
    Args:
        df_retail: DataFrame de INDICADORES_RETAIL
        meses: Lista de meses a incluir (si None, usa todos los disponibles)
        
    Returns:
        Diccionario con driver_name -> DataFrame (mes x SKU)
    """
    # Manejar DataFrame vacío
    if df_retail.empty or 'mes' not in df_retail.columns:
        return {}
    
    # Si no se especifican meses, usar todos los disponibles
    if meses is None:
        meses = df_retail['mes'].unique().tolist()
    
    # Filtrar por meses especificados
    df_filtered = df_retail[df_retail['mes'].isin(meses)].copy()
    
    if df_filtered.empty:
        return {}
    
    # Limpiar SKUs con valores NaN
    df_filtered = df_filtered.dropna(subset=['SKU'])
    df_filtered = df_filtered[df_filtered['SKU'].notna()]
    
    if df_filtered.empty:
        return {}
    
    # Obtener todos los SKUs únicos
    all_skus = sorted(df_filtered['SKU'].unique())
    
    # Crear DataFrames para cada driver (mes x SKU)
    drivers_dataframes = {
        'kg_producidos': pd.DataFrame(index=meses, columns=all_skus, dtype=float),
        'kg_despachados': pd.DataFrame(index=meses, columns=all_skus, dtype=float),
        'hh_directas': pd.DataFrame(index=meses, columns=all_skus, dtype=float),
        'tiempo_maquina_min': pd.DataFrame(index=meses, columns=all_skus, dtype=float)
    }
    
    # Llenar los DataFrames con datos reales
    for mes in meses:
        df_mes = df_filtered[df_filtered['mes'] == mes]
        if not df_mes.empty:
            # Agrupar por SKU y sumar valores
            grouped = df_mes.groupby('SKU').agg({
                'kg_producidos': 'sum',
                'kg_despachados': 'sum', 
                'hh_directas': 'sum',
                'tiempo_maquina_min': 'sum'
            }).fillna(0)
            
            # Llenar el DataFrame para este mes
            for driver_name in drivers_dataframes.keys():
                drivers_dataframes[driver_name].loc[mes] = grouped[driver_name]
    
    # Calcular porcentajes para distribución
    drivers_pct_dataframes = {}
    for driver_name, df_driver in drivers_dataframes.items():
        # Crear DataFrame de porcentajes
        df_pct = pd.DataFrame(index=meses, columns=all_skus, dtype=float)
        
        for mes in meses:
            row_values = df_driver.loc[mes].fillna(0)
            total = row_values.sum()
            
            if total > 0:
                # Calcular porcentajes
                df_pct.loc[mes] = row_values / total
            else:
                # Si no hay datos, todos los SKUs tienen 0%
                df_pct.loc[mes] = 0
        
        drivers_pct_dataframes[f'pct_{driver_name}'] = df_pct
    
    # Combinar drivers originales y porcentajes
    result = {}
    result.update(drivers_dataframes)
    result.update(drivers_pct_dataframes)
    
    return result


def build_drivers_granel(df_granel: pd.DataFrame, meses: list = None) -> Dict[str, pd.DataFrame]:
    """
    Construye drivers para distribución de costos en Granel por Especie usando DataFrames.
    
    Args:
        df_granel: DataFrame de INDICADORES_GRANEL
        meses: Lista de meses a incluir (si None, usa todos los disponibles)
        
    Returns:
        Diccionario con driver_name -> DataFrame (mes x Fruta_id)
    """
    # Manejar DataFrame vacío
    if df_granel.empty or 'mes' not in df_granel.columns:
        return {}
    
    # Si no se especifican meses, usar todos los disponibles
    if meses is None:
        meses = df_granel['mes'].unique().tolist()
    
    # Filtrar por meses especificados
    df_filtered = df_granel[df_granel['mes'].isin(meses)].copy()
    
    if df_filtered.empty:
        return {}
    
    # Limpiar Fruta_id con valores NaN
    df_filtered = df_filtered.dropna(subset=['Fruta_id'])
    df_filtered = df_filtered[df_filtered['Fruta_id'].notna()]
    
    if df_filtered.empty:
        return {}
    
    # Obtener todas las Fruta_ids únicas
    all_frutas = sorted(df_filtered['Fruta_id'].unique())
    
    # Crear DataFrames para cada driver (mes x Fruta_id)
    drivers_dataframes = {
        'kg_producidos': pd.DataFrame(index=meses, columns=all_frutas, dtype=float),
        'hh_directas': pd.DataFrame(index=meses, columns=all_frutas, dtype=float),
        'tiempo_maquina_min': pd.DataFrame(index=meses, columns=all_frutas, dtype=float)
    }
    
    # Llenar los DataFrames con datos reales
    for mes in meses:
        df_mes = df_filtered[df_filtered['mes'] == mes]
        if not df_mes.empty:
            # Agrupar por Fruta_id y sumar valores
            grouped = df_mes.groupby('Fruta_id').agg({
                'kg_producidos': 'sum',
                'hh_directas': 'sum',
                'tiempo_maquina_min': 'sum'
            }).fillna(0)
            
            # Llenar el DataFrame para este mes
            for driver_name in drivers_dataframes.keys():
                drivers_dataframes[driver_name].loc[mes] = grouped[driver_name]
    
    # Calcular porcentajes para distribución
    drivers_pct_dataframes = {}
    for driver_name, df_driver in drivers_dataframes.items():
        # Crear DataFrame de porcentajes
        df_pct = pd.DataFrame(index=meses, columns=all_frutas, dtype=float)
        
        for mes in meses:
            row_values = df_driver.loc[mes].fillna(0)
            total = row_values.sum()
            
            if total > 0:
                # Calcular porcentajes
                df_pct.loc[mes] = row_values / total
            else:
                # Si no hay datos, todas las frutas tienen 0%
                df_pct.loc[mes] = 0
        
        drivers_pct_dataframes[f'pct_{driver_name}'] = df_pct
    
    # Combinar drivers originales y porcentajes
    result = {}
    result.update(drivers_dataframes)
    result.update(drivers_pct_dataframes)
    
    return result

def compute_costos_retail(dfs: Dict[str, pd.DataFrame], rolling_months: list = None) -> pd.DataFrame:
    """
    Calcula todos los costos Retail por SKU usando promedio móvil.
    
    Args:
        dfs: Diccionario de DataFrames
        rolling_months: Lista de meses para promedio móvil (si None, usa últimos 12 meses)
        
    Returns:
        DataFrame con costos por SKU y conceptos
    """
    # Obtener meses para promedio móvil
    if rolling_months is None:
        rolling_months = get_rolling_months(dfs, max_months=12)
    
    
    # Construir drivers usando promedio móvil
    drivers_retail = build_drivers_retail(dfs['INDICADORES_RETAIL'], rolling_months)
    
    if not drivers_retail:
        return pd.DataFrame()
    
    # Obtener SKUs desde el DataFrame de drivers
    if 'kg_producidos' in drivers_retail:
        skus = drivers_retail['kg_producidos'].columns.tolist()
    else:
        return pd.DataFrame()
    
    
    # DataFrame resultado
    resultado = pd.DataFrame(index=skus)

    # Crear cuenta de materiales indirectos en la hoja MAYOR
    # Paso 1, sumar todos los materiales directos de retail y granel
    retail_denominador_acumulado = pd.Series(0.0, index=skus)
    materiales_directos_acumulado_retail = pd.Series(0.0, index=skus)
    
    for mes in rolling_months:
        # Acumular denominador desde DataFrame de drivers
        if mes in drivers_retail['kg_producidos'].index:
            retail_denominador_acumulado += drivers_retail['kg_producidos'].loc[mes]
        
        # Calcular materiales directos específicos por SKU para retail
        materiales_directos_retail = pd.Series(0.0, index=skus)
        df_retail_mes = dfs['INDICADORES_RETAIL'][dfs['INDICADORES_RETAIL']['mes'] == mes]
        
        for idx, fila in df_retail_mes.iterrows():
            sku = fila['SKU']
            if sku in skus:
                materiales_sku = 0
                if 'cajas' in fila and 'costo_unit_caja' in fila:
                    cajas_val = fila['cajas'] if pd.notna(fila['cajas']) else 0
                    costo_caja_val = fila['costo_unit_caja'] if pd.notna(fila['costo_unit_caja']) else 0
                    materiales_sku += cajas_val * costo_caja_val
                if 'bolsas' in fila and 'costo_unit_bolsa' in fila:
                    bolsas_val = fila['bolsas'] if pd.notna(fila['bolsas']) else 0
                    costo_bolsa_val = fila['costo_unit_bolsa'] if pd.notna(fila['costo_unit_bolsa']) else 0
                    materiales_sku += bolsas_val * costo_bolsa_val
                
                materiales_directos_retail[sku] = materiales_sku
        
        materiales_directos_acumulado_retail += materiales_directos_retail

        # Obtener el monto de materiales del mes
        materiales_mes = dfs['MAYOR'][
            (dfs['MAYOR']['familia_cc'] == 'Materiales') & 
            (dfs['MAYOR']['mes'] == mes)
        ]['monto'].sum()

        # Crear nueva fila para Materiales_Indirectos
        nueva_fila = pd.DataFrame({
            'familia_cc': ['Materiales_Indirectos'],
            'mes': [mes],
            'monto': [materiales_mes - materiales_directos_retail.sum()]
        })

        # Agregar la nueva fila al DataFrame MAYOR
        dfs['MAYOR'] = pd.concat([dfs['MAYOR'], nueva_fila], ignore_index=True)
    # Asegurar que ambos tengan el mismo índice antes de dividir
    materiales_directos_acumulado_retail = materiales_directos_acumulado_retail.reindex(skus, fill_value=0)
    retail_denominador_acumulado = retail_denominador_acumulado.reindex(skus, fill_value=1)
    resultado['Materiales_Directos'] = -(materiales_directos_acumulado_retail / retail_denominador_acumulado)


    # Procesar conceptos desde CONFIG_SPLITS usando promedio móvil de costos
    for _, config_row in dfs['CONFIG_SPLITS'].iterrows():
        concepto = config_row['concepto']
        
        # Conceptos estándar desde MAYOR usando promedio ponderado móvil
        df_mayor_concepto = dfs['MAYOR'][
            (dfs['MAYOR']['familia_cc'] == concepto) & 
            (dfs['MAYOR']['mes'].isin(rolling_months))
        ]
        
        if not df_mayor_concepto.empty:            
            # Usar configuración para distribución, manejando valores NaN
            split_retail_pct = config_row.get('split_retail_pct', 0.5)
            if pd.isna(split_retail_pct):
                split_retail_pct = 0.5  # Valor por defecto si es NaN
                
            driver_interno = config_row.get('driver_interno', 'kg_producidos')
            if pd.isna(driver_interno):
                driver_interno = 'kg_producidos'  # Valor por defecto si es NaN
                
            denominador = config_row.get('denominador', 'kg_producidos')
            resultado[concepto] = pd.Series(0.0, index=skus)

            retail_costos_acumulado = pd.Series(0.0, index=skus)
            retail_denominador_acumulado = pd.Series(0.0, index=skus)
            # Distribuir internamente
            for mes in rolling_months:
                if concepto == 'MO_Directa' or concepto == 'MO_Indirecta':
                    # Buscar split_hh_retail para este mes específico
                    split_hh_data = dfs['MAYOR'][
                        (dfs['MAYOR']['familia_cc'] == 'split_hh_retail') & 
                        (dfs['MAYOR']['mes'] == mes)
                    ]
                    
                    if not split_hh_data.empty:
                        split_hh_value = split_hh_data['monto'].iloc[0]
                        split_hh_retail = split_hh_value
                    else:
                        # Si no hay datos de split_hh_retail para este mes, usar 0.5 como default
                        split_hh_retail = 0.5
                    
                    monto_retail_mensual = df_mayor_concepto[df_mayor_concepto['mes'] == mes]['monto'] * split_hh_retail
                else:
                    monto_retail_mensual = df_mayor_concepto[df_mayor_concepto['mes'] == mes]['monto'] * split_retail_pct
                
                # Obtener drivers desde DataFrames
                pct_driver_key = f'pct_{driver_interno}'
                denominador_key = denominador
                
                if pct_driver_key in drivers_retail and mes in drivers_retail[pct_driver_key].index:
                    retail_pct_driver = drivers_retail[pct_driver_key].loc[mes]
                else:
                    retail_pct_driver = pd.Series(0, index=skus)
                
                if denominador_key in drivers_retail and mes in drivers_retail[denominador_key].index:
                    retail_denominador = drivers_retail[denominador_key].loc[mes]
                else:
                    retail_denominador = pd.Series(0, index=skus)
                
                # Distribuir costos usando DataFrames
                if retail_pct_driver.sum() > 0:
                    # Obtener un único valor por mes
                    if isinstance(monto_retail_mensual, pd.Series):
                        monto_escalar = monto_retail_mensual.iloc[0] if len(monto_retail_mensual) > 0 else 0
                    else:
                        monto_escalar = monto_retail_mensual
                    
                    # Distribuir el monto usando los porcentajes del driver
                    costos_mes = retail_pct_driver * monto_escalar
                    retail_costos_acumulado += costos_mes.fillna(0)
                
                # Acumular denominador
                retail_denominador_acumulado += retail_denominador.fillna(0)
            
            resultado[concepto] = retail_costos_acumulado / retail_denominador_acumulado
        
    # Renombrar columnas para compatibilidad con recalculate_totals
    column_renames = {
        'MO_Directa': 'MO Directa',
        'MO_Indirecta': 'MO Indirecta',
        'Materiales_Directos': 'Materiales Directos',
        'Materiales_Indirectos': 'Materiales Indirectos', 
        'Mantención': 'Mantención',  # Mantener nombre que espera recalculate_totals
        'Fletes_Internos': 'Fletes Internos',
        'MMPP_Fruta': 'MMPP (Fruta) (USD/kg)'  # Agregar renombrado para MMPP
    }
    
    for old_name, new_name in column_renames.items():
        if old_name in resultado.columns:
            resultado = resultado.rename(columns={old_name: new_name})
    
    # Calcular totales
    conceptos_directos = ['MMPP (Fruta) (USD/kg)', 'Materiales Directos', 'MO Directa', 'Laboratorio', 'Mantención']
    conceptos_indirectos = [col for col in resultado.columns if col not in conceptos_directos]
    
    if conceptos_directos:
        cols_directos_existentes = [col for col in conceptos_directos if col in resultado.columns]
        if cols_directos_existentes:
            resultado['Costos_Directos'] = resultado[cols_directos_existentes].sum(axis=1)
        else:
            resultado['Costos_Directos'] = 0
    
    if conceptos_indirectos:
        resultado['Costos_Indirectos'] = resultado[conceptos_indirectos].sum(axis=1)
    
    # Calcular Costos_Totales de manera segura
    costos_directos = resultado.get('Costos_Directos', 0)
    costos_indirectos = resultado.get('Costos_Indirectos', 0)
    
    # Asegurar que ambos sean Series o escalares compatibles
    if isinstance(costos_directos, pd.Series) or isinstance(costos_indirectos, pd.Series):
        if not isinstance(costos_directos, pd.Series):
            costos_directos = pd.Series(costos_directos, index=resultado.index)
        if not isinstance(costos_indirectos, pd.Series):
            costos_indirectos = pd.Series(costos_indirectos, index=resultado.index)
    
    resultado['Costos_Totales'] = costos_directos + costos_indirectos
    
    return resultado


def compute_costos_granel(dfs: Dict[str, pd.DataFrame], rolling_months: list = None) -> pd.DataFrame:
    """
    Calcula todos los costos Granel por Especie usando promedio móvil.
    
    Args:
        dfs: Diccionario de DataFrames
        rolling_months: Lista de meses para promedio móvil (si None, usa últimos 12 meses)
        
    Returns:
        DataFrame con costos por Especie y conceptos
    """
    # Obtener meses para promedio móvil
    if rolling_months is None:
        rolling_months = get_rolling_months(dfs, max_months=12)
    
    
    # Construir drivers usando promedio móvil
    drivers_granel = build_drivers_granel(dfs['INDICADORES_GRANEL'], rolling_months)
    
    if not drivers_granel:
        return pd.DataFrame()
    
    # Obtener Especies desde el DataFrame de drivers
    if 'kg_producidos' in drivers_granel:
        especies = drivers_granel['kg_producidos'].columns.tolist()
    else:
        return pd.DataFrame()
    
    
    # DataFrame resultado
    resultado = pd.DataFrame(index=especies)

    # Crear cuenta de materiales indirectos en la hoja MAYOR
    # Paso 1, sumar todos los materiales directos de retail y granel
    granel_denominador_acumulado = pd.Series(0.0, index=especies)
    materiales_directos_acumulado_granel = pd.Series(0.0, index=especies)
    
    for mes in rolling_months:
        # Acumular denominador desde DataFrame de drivers
        if mes in drivers_granel['kg_producidos'].index:
            granel_denominador_acumulado += drivers_granel['kg_producidos'].loc[mes]
        
        # Calcular materiales directos específicos por fruta para granel
        materiales_directos_granel = pd.Series(0.0, index=especies)
        df_granel_mes = dfs['INDICADORES_GRANEL'][dfs['INDICADORES_GRANEL']['mes'] == mes]
        
        for idx, fila in df_granel_mes.iterrows():
            fruta_id = fila['Fruta_id']
            if fruta_id in especies:
                materiales_fruta = 0
                if 'cajas' in fila and 'costo_unit_caja' in fila:
                    cajas_val = fila['cajas'] if pd.notna(fila['cajas']) else 0
                    costo_caja_val = fila['costo_unit_caja'] if pd.notna(fila['costo_unit_caja']) else 0
                    materiales_fruta += cajas_val * costo_caja_val
                if 'bolsas' in fila and 'costo_unit_bolsa' in fila:
                    bolsas_val = fila['bolsas'] if pd.notna(fila['bolsas']) else 0
                    costo_bolsa_val = fila['costo_unit_bolsa'] if pd.notna(fila['costo_unit_bolsa']) else 0
                    materiales_fruta += bolsas_val * costo_bolsa_val
                
                materiales_directos_granel[fruta_id] = materiales_fruta
        
        materiales_directos_acumulado_granel += materiales_directos_granel

        # Obtener el monto de materiales del mes
        materiales_mes = dfs['MAYOR'][
            (dfs['MAYOR']['familia_cc'] == 'Materiales') & 
            (dfs['MAYOR']['mes'] == mes)
        ]['monto'].sum()

        # Crear nueva fila para Materiales_Indirectos
        nueva_fila = pd.DataFrame({
            'familia_cc': ['Materiales_Indirectos'],
            'mes': [mes],
            'monto': [materiales_mes - materiales_directos_granel.sum()]
        })

        # Agregar la nueva fila al DataFrame MAYOR
        dfs['MAYOR'] = pd.concat([dfs['MAYOR'], nueva_fila], ignore_index=True)
    
    # Asegurar que ambos tengan el mismo índice antes de dividir
    materiales_directos_acumulado_granel = materiales_directos_acumulado_granel.reindex(especies, fill_value=0)
    granel_denominador_acumulado = granel_denominador_acumulado.reindex(especies, fill_value=1)
    resultado['Materiales_Directos'] = -(materiales_directos_acumulado_granel / granel_denominador_acumulado)

    # Procesar conceptos desde CONFIG_SPLITS usando promedio móvil de costos
    for _, config_row in dfs['CONFIG_SPLITS'].iterrows():
        concepto = config_row['concepto']
        
        # Conceptos estándar desde MAYOR usando promedio ponderado móvil
        df_mayor_concepto = dfs['MAYOR'][
            (dfs['MAYOR']['familia_cc'] == concepto) & 
            (dfs['MAYOR']['mes'].isin(rolling_months))
        ]
        
        if not df_mayor_concepto.empty:            
            # Usar configuración para distribución, manejando valores NaN
            split_granel_pct = config_row.get('split_granel_pct', 0.5)
            if pd.isna(split_granel_pct):
                split_granel_pct = 0.5  # Valor por defecto si es NaN
                
            driver_interno = config_row.get('driver_interno', 'kg_producidos')
            if pd.isna(driver_interno):
                driver_interno = 'kg_producidos'  # Valor por defecto si es NaN
                
            denominador = config_row.get('denominador', 'kg_producidos')
            resultado[concepto] = pd.Series(0.0, index=especies)

            granel_costos_acumulado = pd.Series(0.0, index=especies)
            granel_denominador_acumulado = pd.Series(0.0, index=especies)
            # Distribuir internamente
            for mes in rolling_months:
                if concepto == 'MO_Directa' or concepto == 'MO_Indirecta':
                    # Buscar split_hh_retail para este mes específico
                    split_hh_data = dfs['MAYOR'][
                        (dfs['MAYOR']['familia_cc'] == 'split_hh_retail') & 
                        (dfs['MAYOR']['mes'] == mes)
                    ]
                    
                    if not split_hh_data.empty:
                        split_hh_value = split_hh_data['monto'].iloc[0]
                        split_hh_granel = 1 - split_hh_value
                    else:
                        # Si no hay datos de split_hh_retail para este mes, usar 0.5 como default
                        split_hh_granel = 0.5
                    
                    monto_granel_mensual = df_mayor_concepto[df_mayor_concepto['mes'] == mes]['monto'] * split_hh_granel
                else:
                    monto_granel_mensual = df_mayor_concepto[df_mayor_concepto['mes'] == mes]['monto'] * split_granel_pct
                # Obtener drivers desde DataFrames
                pct_driver_key = f'pct_{driver_interno}'
                denominador_key = denominador
                
                if pct_driver_key in drivers_granel and mes in drivers_granel[pct_driver_key].index:
                    granel_pct_driver = drivers_granel[pct_driver_key].loc[mes]
                else:
                    granel_pct_driver = pd.Series(0, index=especies)
                
                if denominador_key in drivers_granel and mes in drivers_granel[denominador_key].index:
                    granel_denominador = drivers_granel[denominador_key].loc[mes]
                else:
                    granel_denominador = pd.Series(0, index=especies)
                
                # Distribuir costos usando DataFrames
                if granel_pct_driver.sum() > 0:
                    # Obtener un único valor por mes
                    if isinstance(monto_granel_mensual, pd.Series):
                        monto_escalar = monto_granel_mensual.iloc[0] if len(monto_granel_mensual) > 0 else 0
                    else:
                        monto_escalar = monto_granel_mensual
                    
                    # Distribuir el monto usando los porcentajes del driver
                    costos_mes = granel_pct_driver * monto_escalar
                    granel_costos_acumulado += costos_mes.fillna(0)
                
                # Acumular denominador
                granel_denominador_acumulado += granel_denominador.fillna(0)
            
            resultado[concepto] = granel_costos_acumulado / granel_denominador_acumulado
    # Renombrar columnas para compatibilidad con recalculate_totals
    column_renames = {
        'MO_Directa': 'MO Directa',
        'MO_Indirecta': 'MO Indirecta',
        'Materiales_Directos': 'Materiales Directos',
        'Materiales_Indirectos': 'Materiales Indirectos',
        'Mantención': 'Mantención',  # Mantener nombre que espera recalculate_totals
        'Fletes_Internos': 'Fletes Internos',
    }
    
    for old_name, new_name in column_renames.items():
        if old_name in resultado.columns:
            resultado = resultado.rename(columns={old_name: new_name})
    
    # Calcular totales
    conceptos_directos = ['Materiales Directos', 'MO Directa', 'Laboratorio', 'Mantención']
    conceptos_indirectos = [col for col in resultado.columns if col not in conceptos_directos]
    
    if conceptos_directos:
        cols_directos_existentes = [col for col in conceptos_directos if col in resultado.columns]
        if cols_directos_existentes:
            resultado['Costos_Directos'] = resultado[cols_directos_existentes].sum(axis=1)
        else:
            resultado['Costos_Directos'] = 0
    
    if conceptos_indirectos:
        resultado['Costos_Indirectos'] = resultado[conceptos_indirectos].sum(axis=1)
    
    # Calcular Costos_Totales de manera segura
    costos_directos = resultado.get('Costos_Directos', 0)
    costos_indirectos = resultado.get('Costos_Indirectos', 0)
    
    # Asegurar que ambos sean Series o escalares compatibles
    if isinstance(costos_directos, pd.Series) or isinstance(costos_indirectos, pd.Series):
        if not isinstance(costos_directos, pd.Series):
            costos_directos = pd.Series(costos_directos, index=resultado.index)
        if not isinstance(costos_indirectos, pd.Series):
            costos_indirectos = pd.Series(costos_indirectos, index=resultado.index)
    
    resultado['Costos_Totales'] = costos_directos + costos_indirectos
    
    return resultado


# Funciones de utilidad adicionales

def get_available_months(dfs: Dict[str, pd.DataFrame]) -> list:
    """
    Obtiene la lista de meses disponibles en los datos.
    
    Args:
        dfs: Diccionario de DataFrames
        
    Returns:
        Lista de meses únicos disponibles
    """
    meses = set()
    
    for sheet_name in ['MAYOR', 'INDICADORES_RETAIL', 'INDICADORES_GRANEL']:
        if sheet_name in dfs and 'mes' in dfs[sheet_name].columns:
            meses.update(dfs[sheet_name]['mes'].dropna().unique())
    
    return sorted(list(meses))


def get_rolling_months(dfs: Dict[str, pd.DataFrame], max_months: int = 12) -> list:
    """
    Obtiene los últimos N meses disponibles para promedio móvil.
    
    Args:
        dfs: Diccionario de DataFrames
        max_months: Número máximo de meses a incluir (default: 12)
        
    Returns:
        Lista de los últimos N meses disponibles
    """
    all_months = get_available_months(dfs)
    
    if not all_months:
        return []
    
    # Tomar los últimos max_months meses
    return all_months[-max_months:]


def compute_full_cost_analysis(uploaded_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Función principal que ejecuta el análisis completo de costos usando promedio móvil.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        Diccionario con resultados de Retail y Granel
        
    Raises:
        ValueError: Si hay problemas con los datos o validaciones
    """
    # Leer datos
    dfs = read_source(uploaded_bytes)
    
    # Validar estructura
    validate_inputs(dfs)
    
    # Obtener meses para promedio móvil
    rolling_months = get_rolling_months(dfs, max_months=12)
    if not rolling_months:
        raise ValueError("No hay datos disponibles para calcular promedio móvil")
    
    # Calcular costos usando promedio móvil
    resultados = {}
    
    # try:  # Comentado para debugging
    resultados['retail'] = compute_costos_retail(dfs, rolling_months)
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error calculando costos Retail: {str(e)}")
    
    # try:  # Comentado para debugging
    resultados['granel'] = compute_costos_granel(dfs, rolling_months)
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error calculando costos Granel: {str(e)}")
    
    # Agregar información de meses utilizados
    resultados['rolling_months'] = rolling_months
    resultados['months_count'] = len(rolling_months)
    
    return resultados


def build_granel_from_cost_engine(uploaded_bytes: bytes) -> tuple:
    """
    Construye datos de granel usando el cost_engine con promedio móvil.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        Tuple (df_granel, df_granel_ponderado) compatible con la aplicación
    """
    # try:  # Comentado para debugging
    # Importar funciones de data_io
    from src.data_io import read_workbook, build_fact_granel, build_fact_granel_ponderado, load_info_fruta
    
    # Leer hojas usando data_io
    sheets = read_workbook(uploaded_bytes)
    
    # Intentar usar las funciones probadas de data_io primero
    # try:  # Comentado para debugging
    # Usar build_granel de data_io si las hojas existen
    if 'FACT_GRANEL' in sheets and 'FACT_GRANEL_POND' in sheets:
        df_granel = build_fact_granel(sheets['FACT_GRANEL'], fill_before_first=True)
        
        # Cargar info_fruta
        if 'FRUTA' in sheets:
            info_fruta = load_info_fruta(sheets['FRUTA'])
        elif 'INFO_FRUTA' in sheets:
            info_fruta = load_info_fruta(sheets['INFO_FRUTA'])
        else:
            info_fruta = pd.DataFrame()
        
        df_granel_ponderado = build_fact_granel_ponderado(sheets['FACT_GRANEL_POND'], info_fruta=info_fruta)
        
        return df_granel, df_granel_ponderado
    
    # except Exception:  # Comentado para debugging
    #     # Si falla, usar cost_engine como fallback
    #     pass
    
    # Fallback: usar cost_engine
    resultados = compute_full_cost_analysis(uploaded_bytes)
    
    if 'granel' not in resultados:
        # Crear DataFrames vacíos si no hay datos de granel
        df_granel = pd.DataFrame()
        df_granel_ponderado = pd.DataFrame()
        return df_granel, df_granel_ponderado
    
    df_granel = resultados['granel']
    
    # Construir granel ponderado usando INFO_FRUTA
    if 'FRUTA' in sheets:
        # try:  # Comentado para debugging
        info_fruta = load_info_fruta(sheets['FRUTA'])
        # except:
        #     info_fruta = sheets['FRUTA']
    elif 'INFO_FRUTA' in sheets:
        # try:  # Comentado para debugging
        info_fruta = load_info_fruta(sheets['INFO_FRUTA'])
        # except:
        #     info_fruta = sheets['INFO_FRUTA']
    else:
        info_fruta = pd.DataFrame()
    
    # Crear df_granel_ponderado básico
    df_granel_ponderado = df_granel.reset_index()
    df_granel_ponderado = df_granel_ponderado.rename(columns={'index': 'Fruta_id'})
    
    # Crear columnas faltantes específicas para granel
    # 1. MO Total
    mo_components = ["MO Directa", "MO Indirecta"]
    if all(col in df_granel_ponderado.columns for col in mo_components):
        df_granel_ponderado["MO Total"] = df_granel_ponderado[mo_components].sum(axis=1)
    
    # 2. Materiales Total
    materiales_components = ["Materiales Directos", "Materiales Indirectos"]
    if all(col in df_granel_ponderado.columns for col in materiales_components):
        df_granel_ponderado["Materiales Total"] = df_granel_ponderado[materiales_components].sum(axis=1)
    
    # 3. Proceso Granel (USD/kg)
    proceso_granel_components = ["MO Directa", "MO Indirecta", "Materiales Directos", "Materiales Indirectos", "Laboratorio", "Mantención", "Servicios Generales", "Utilities"]
    if all(col in df_granel_ponderado.columns for col in proceso_granel_components):
        df_granel_ponderado["Proceso Granel (USD/kg)"] = df_granel_ponderado[proceso_granel_components].sum(axis=1)
    
    # Merge con info_fruta para obtener nombres y datos adicionales
    if not info_fruta.empty:
        # Asegurar que Fruta_id sea string en ambos DataFrames
        df_granel_ponderado['Fruta_id'] = df_granel_ponderado['Fruta_id'].astype(str)
        info_fruta['Fruta_id'] = info_fruta['Fruta_id'].astype(str)
        
        # Seleccionar columnas disponibles para el merge
        merge_cols = ['Fruta_id']
        if 'Name' in info_fruta.columns:
            merge_cols.append('Name')
        if 'Nombre' in info_fruta.columns:
            merge_cols.append('Nombre')
        
        df_granel_ponderado = df_granel_ponderado.merge(
            info_fruta[merge_cols], 
            on='Fruta_id', how='left'
        )
        
        # Asegurar que hay una columna Name
        if 'Name' not in df_granel_ponderado.columns:
            if 'Nombre' in df_granel_ponderado.columns:
                df_granel_ponderado['Name'] = df_granel_ponderado['Nombre']
            else:
                df_granel_ponderado['Name'] = df_granel_ponderado['Fruta_id']
    else:
        df_granel_ponderado['Name'] = df_granel_ponderado['Fruta_id']
    
    return df_granel, df_granel_ponderado
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error construyendo granel desde cost_engine: {str(e)}")


def get_available_months_from_excel(uploaded_bytes: bytes) -> list:
    """
    Obtiene los meses disponibles en el archivo Excel.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        Lista de meses disponibles
    """
    # try:  # Comentado para debugging
    dfs = read_source(uploaded_bytes)
    return get_available_months(dfs)
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error obteniendo meses disponibles: {str(e)}")


def build_cost_engine_pipeline(uploaded_bytes: bytes) -> dict:
    """
    Pipeline completo que integra cost_engine con data_io para construir todos los datos necesarios.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        
    Returns:
        Dict con todos los DataFrames necesarios para la aplicación
    """
    # try:  # Comentado para debugging
    # Obtener meses disponibles
    meses_disponibles = get_available_months_from_excel(uploaded_bytes)
    
    if not meses_disponibles:
        raise ValueError("No hay meses disponibles en los datos")
    
    # Obtener meses para promedio móvil
    rolling_months = get_rolling_months(read_source(uploaded_bytes), max_months=12)
    
    # Construir todos los componentes usando promedio móvil
    df_granel, df_granel_ponderado = build_granel_from_cost_engine(uploaded_bytes)
    detalle = build_detalle_from_cost_engine(uploaded_bytes)


    
    # Leer datos adicionales
    dfs = read_source(uploaded_bytes)
    
    # Preparar receta e info_fruta si existen
    receta_df = None
    info_df = None
    
    if 'RECETAS' in dfs:
        receta_df = dfs['RECETAS']
    
    if 'FRUTA' in dfs:
        info_df = dfs['FRUTA']
    
    return {
        'detalle': detalle,
        'df_granel': df_granel,
        'df_granel_ponderado': df_granel_ponderado,
        'receta_df': receta_df,
        'info_df': info_df,
        'rolling_months': rolling_months,
        'months_count': len(rolling_months),
        'meses_disponibles': meses_disponibles,
        'cost_engine_results': compute_full_cost_analysis(uploaded_bytes)
    }
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error en pipeline cost_engine: {str(e)}")
