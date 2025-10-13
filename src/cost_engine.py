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

key_map = {
    # ARÁNDANOS
    "fruta_1": "ARANDANOS",
    "fruta_2": "ARANDANOS",
    "fruta_3": "ARANDANOS",         # Arsub
    "fruta_4": "ARANDANOS",         # Arsub_organic

    # PALTA (AVOCADO)
    "fruta_67": "AVOCADO",
    "fruta_68": "AVOCADO",

    # BANANA
    "fruta_5": "BANANA",
    "fruta_6": "BANANA",

    # CACAO
    "fruta_7": "CACAO",
    "fruta_8": "CACAO",

    # CAÑAMO
    "fruta_9": "CAÑAMO",
    "fruta_10": "CAÑAMO",

    # CEREZA (todas las variantes)
    "fruta_11": "CEREZA",  # Cereza_acida
    "fruta_12": "CEREZA",
    "fruta_13": "CEREZA",  # Cereza_oscura
    "fruta_14": "CEREZA",
    "fruta_15": "CEREZA",  # Cereza_roja
    "fruta_16": "CEREZA",

    # CHIA
    "fruta_17": "CHIA",
    "fruta_18": "CHIA",

    # CHIRIMOYA
    "fruta_19": "CHIRIMOYA",
    "fruta_20": "CHIRIMOYA",

    # CRANBERRIES
    "fruta_23": "CRANBERRIES",
    "fruta_24": "CRANBERRIES",

    # DÁTILES
    "fruta_25": "DATILES",
    "fruta_26": "DATILES",

    # DRAGON FRUIT
    "fruta_27": "FRUTOS DEL DRAGON",
    "fruta_28": "FRUTOS DEL DRAGON",

    # DURAZNO
    "fruta_29": "DURAZNO",
    "fruta_30": "DURAZNO",

    # ESPINACA
    "fruta_31": "ESPINACA",
    "fruta_32": "ESPINACA",

    # FRAMBUESA (incluye variantes)
    "fruta_33": "FRAMBUESA",
    "fruta_34": "FRAMBUESA",
    "fruta_35": "FRAMBUESA",  # Framsub
    "fruta_36": "FRAMBUESA",
    "fruta_85": "FRAMBUESA",  # frambAA
    "fruta_86": "FRAMBUESA",

    # FRUTILLA (incluye diced/sliced/sub)
    "fruta_37": "FRUTILLAS",
    "fruta_38": "FRUTILLAS",
    "fruta_81": "FRUTILLAS",   # diced_frutilla
    "fruta_82": "FRUTILLAS",
    "fruta_111": "FRUTILLAS",  # sliced_frutilla
    "fruta_112": "FRUTILLAS",
    "fruta_87": "FRUTILLAS",   # frutsub
    "fruta_88": "FRUTILLAS",

    # GOJI
    "fruta_39": "GOJI",
    "fruta_40": "GOJI",

    # GRANADA
    "fruta_41": "GRANADA",
    "fruta_42": "GRANADA",

    # KALE
    "fruta_43": "KALE",
    "fruta_44": "KALE",

    # KIWI
    "fruta_45": "KIWI",
    "fruta_46": "KIWI",

    # LIMÓN
    "fruta_47": "LIMÓN",
    "fruta_48": "LIMÓN",

    # LÚCUMA
    "fruta_49": "LÚCUMA",
    "fruta_50": "LÚCUMA",

    # MANGO (incluye Mangosub)
    "fruta_51": "MANGO",
    "fruta_52": "MANGO",
    "fruta_53": "MANGO",  # Mangosub
    "fruta_54": "MANGO",

    # MANZANA
    "fruta_89": "MANZANA",
    "fruta_90": "MANZANA",

    # MARACUYÁ
    "fruta_55": "MARACUYÁ",
    "fruta_56": "MARACUYÁ",

    # MELÓN (incluye pure/sub)
    "fruta_57": "MELÓN",
    "fruta_58": "MELÓN",
    "fruta_59": "MELÓN",  # Melon_pure
    "fruta_60": "MELÓN",
    "fruta_61": "MELÓN",  # Melonsub
    "fruta_62": "MELÓN",

    # MORA
    "fruta_63": "MORAS",
    "fruta_64": "MORAS",

    # NARANJA
    "fruta_65": "NARANJA",
    "fruta_66": "NARANJA",

    # PAPAYA
    "fruta_69": "PAPAYA",
    "fruta_70": "PAPAYA",

    # PIÑA (incluye Piñasub)
    "fruta_71": "PIÑA",
    "fruta_72": "PIÑA",
    "fruta_73": "PIÑA",  # Piñasub
    "fruta_74": "PIÑA",

    # UVA
    "fruta_75": "UVAS",
    "fruta_76": "UVAS",

    # ZARZAPARRILLA
    "fruta_77": "ZARZAPARRILLA",
    "fruta_78": "ZARZAPARRILLA",

    # AÇAÍ
    "fruta_79": "ACAI",
    "fruta_80": "ACAI",

    # ZAPALLO / Semilla de calabaza
    "fruta_109": "ZAPALLO",
    "fruta_110": "ZAPALLO",
}

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

def compute_almacenaje_mmpp_por_fruta(
    dfs: Dict[str, pd.DataFrame],
    rolling_months: list,
    guardakey_map: Union[pd.DataFrame, Dict[str, str]]
    ) -> pd.DataFrame:
    """
    Calcula 'Almacenaje' (USD/kg, negativo) por Fruta_id.

    Parámetros:
      - dfs: diccionario con hojas 'MAYOR', 'INDICADORES_GRANEL', 'INDICADORES_RETAIL', 'RECETAS', 'FRUTA' (FRUTA opcional)
      - rolling_months: lista de meses a considerar
      - guardakey_map:
          * DataFrame con columnas ['Fruta_id','GuardaKey'], o
          * dict {Fruta_id: GuardaKey}

    Lógica:
      1) Pool mensual de 'Guarda' = MAYOR['Guarda']_mes * (1 - split_pt_mes)
      2) Prorrateo mensual por GuardaKey usando kg_guardados de INDICADORES_GRANEL
      3) Suma USD en el período por GuardaKey
      4) Denominador: kg despachados del período por GuardaKey (Retail+Recetas)
      5) Unitario por GuardaKey = - USD_tot / kg_desp_tot  → propagar a Fruta_id
    """
    import pandas as pd
    import numpy as np

    # -------- Validaciones mínimas --------
    for req in ("MAYOR", "INDICADORES_GRANEL", "INDICADORES_RETAIL", "RECETAS"):
        if req not in dfs:
            return pd.DataFrame(columns=["Fruta_id","Almacenaje"])
    if not rolling_months:
        return pd.DataFrame(columns=["Fruta_id","Almacenaje"])

    mayor = dfs["MAYOR"].copy()
    gra   = dfs["INDICADORES_GRANEL"].copy()
    ret   = dfs["INDICADORES_RETAIL"].copy()
    rec   = dfs["RECETAS"].copy()

    # -------- Mapa Fruta_id -> GuardaKey (proporcionado por el usuario) --------
    if isinstance(guardakey_map, dict):
        map_df = pd.DataFrame(
            {"Fruta_id": list(guardakey_map.keys()), "GuardaKey": list(guardakey_map.values())}
        )
    else:
        map_df = guardakey_map.copy()
    if not {"Fruta_id","GuardaKey"}.issubset(map_df.columns):
        raise ValueError("guardakey_map debe tener columnas ['Fruta_id','GuardaKey'] o ser dict {Fruta_id: GuardaKey}")

    map_df["Fruta_id"]  = map_df["Fruta_id"].astype(str)
    map_df["GuardaKey"] = map_df["GuardaKey"].astype(str)
    fruta_to_key = map_df.groupby("GuardaKey")["Fruta_id"].apply(list).to_dict()

    # -------- Filtrar período --------
    mayor_per = mayor.loc[mayor["mes"].isin(rolling_months)].copy()
    gra_per   = gra.loc[gra["mes"].isin(rolling_months)].copy()
    ret_per   = ret.loc[ret["mes"].isin(rolling_months)].copy()

    # -------- (1) Pool mensual de Guarda (USD) --------
    guarda_mes = (mayor_per.loc[mayor_per["familia_cc"].eq("Guarda")]
                           .groupby("mes")["monto"].sum()
                           .reindex(rolling_months).fillna(0.0))
    split_pt_mes = (mayor.loc[mayor["familia_cc"].eq("split_pt")]
                           .groupby("mes")["monto"].sum()
                           .reindex(rolling_months).fillna(0.0))
    split_factor_mes = (1.0 - split_pt_mes).clip(lower=0.0, upper=1.0)  # parte MMPP
    pool_mes_usd = guarda_mes * split_factor_mes  # Serie por mes (USD; normalmente negativa)

    # -------- (2) kg_guardados por GuardaKey y mes (para prorrateo del pool) --------
    # detectar columna de kg_guardados; proxy: kg_producidos
    col_kg_g = None
    for c in ["kg_guardados","kilos_guardados","kg_almacenados","kilos_almacenados","kg_producidos"]:
        if c in gra_per.columns:
            col_kg_g = c; break

    gra_per["Fruta_id"]  = gra_per["Fruta_id"].astype(str)
    gra_per[col_kg_g]    = pd.to_numeric(gra_per[col_kg_g], errors="coerce").fillna(0.0)
    gra_g = gra_per.merge(map_df, on="Fruta_id", how="left")
    gra_g["GuardaKey"] = gra_g["GuardaKey"].fillna(gra_g["Fruta_id"])  # fallback mínimo

    kg_guardakey_mes = gra_g.groupby(["mes","GuardaKey"])[col_kg_g].sum()

    # prorrateo del pool mensual
    usd_guardakey_period = {}
    for mes in rolling_months:
        pool = float(pool_mes_usd.get(mes, 0.0))
        if pool == 0.0:
            continue
        serie_mes = kg_guardakey_mes.xs(mes, level="mes") if mes in kg_guardakey_mes.index.get_level_values(0) else None
        if serie_mes is None or serie_mes.empty:
            continue
        tot_mes = float(serie_mes.sum())
        if tot_mes <= 0.0:
            continue
        share = serie_mes / tot_mes
        for gk, usd in (share * pool).items():
            usd_guardakey_period[gk] = usd_guardakey_period.get(gk, 0.0) + float(usd)


    # -------- (3) kg despachados del PERÍODO por GuardaKey (Retail + Recetas) --------
    if "kg_despachados" not in ret_per.columns:
        ret_per["kg_despachados"] = 0.0
    ret_per["kg_despachados"] = pd.to_numeric(ret_per["kg_despachados"], errors="coerce").fillna(0.0)

    # --- expandir kg de SKU a kg de fruta del período y mapear a GuardaKey ---
    df_join = ret_per.merge(rec[["SKU","Fruta_id","Porcentaje"]].drop_duplicates(), on="SKU", how="left")

    df_join["Fruta_id"]  = df_join["Fruta_id"].astype(str)
    df_join["kg_fruta"]  = df_join["kg_despachados"] * df_join["Porcentaje"]/100

    df_join = df_join.merge(map_df, on="Fruta_id", how="left")
    df_join["GuardaKey"] = df_join["GuardaKey"].fillna(df_join["Fruta_id"])
    kg_desp_guardakey_period = df_join.groupby("GuardaKey")["kg_fruta"].sum()

    # -------- (4) Unitario por GuardaKey y propagación a Fruta_id --------
    rows = []
    for gk, usd_total in usd_guardakey_period.items():
        kg_norm = float(kg_desp_guardakey_period.get(gk, 0.0))
        unit = -(usd_total / kg_norm) if kg_norm > 0 else 0.0  # negativo por convención
        for fruta_id in fruta_to_key.get(gk, []):
            rows.append({"Fruta_id": str(fruta_id), "Almacenaje": unit})

    df_out = pd.DataFrame(rows).drop_duplicates(subset=["Fruta_id"])
    df_out["Fruta_id"]  = df_out["Fruta_id"].astype(str)
    df_out["Almacenaje"] = pd.to_numeric(df_out["Almacenaje"], errors="coerce").fillna(0.0)
    return df_out

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
        compute_latest_price, build_dim_sku,
        compute_mmpp_unified, correct_species_from_recipes, 
        ensure_list_species, recalculate_totals, load_receta_sku
    )
    
    # 1. Leer todas las hojas usando la función probada de data_io
    sheets = read_workbook(uploaded_bytes)
    
    # 2. Intentar usar cost_engine para costos con promedio móvil (si hay datos suficientes)
    resultados_cost_engine = None
    # try:  # Comentado para debugging
    resultados_cost_engine = compute_full_cost_analysis(uploaded_bytes)

    df_granel, info_fruta, df_granel_optimo = build_granel_from_cost_engine(uploaded_bytes)

    # Optimos
    costos_detalle_optimo = build_tbl_costos_pond(sheets['OPTIMOS_RETAIL'])

    # except Exception:
    #     # Si falla el cost_engine, continuar con data_io solamente
    #     pass
    
    # 3. Construir costos base usando data_io (funcionalidad probada)
    costos_detalle = None
    
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
    else:
        latest_prices = pd.DataFrame(columns=['SKU-Cliente', 'PrecioVenta (USD/kg)'])
    
    # 5. Procesar dimensiones usando data_io
    if 'DIM_SKU' in sheets:
        dim = build_dim_sku(sheets["DIM_SKU"])

    # 6. Procesar volúmenes y producción desde INDICADORES_RETAIL
    if 'INDICADORES_RETAIL' in sheets:
        retail_data = sheets['INDICADORES_RETAIL'].copy()
        
        if not retail_data.empty and 'SKU' in retail_data.columns:
            # Determinar columnas de agrupación (usar SKU-Cliente si existe para evitar duplicación)
            group_cols = ['SKU', 'SKU-Cliente'] if 'SKU-Cliente' in retail_data.columns else ['SKU']
            
            # Preparar datos de volúmenes (kg_despachados -> KgEmbarcados)
            if 'kg_despachados' in retail_data.columns:
                # Convertir a numérico antes de sumar
                retail_data['kg_despachados'] = pd.to_numeric(retail_data['kg_despachados'], errors='coerce')
                volumenes_recientes = retail_data.groupby(group_cols)['kg_despachados'].sum().reset_index()
                volumenes_recientes = volumenes_recientes.rename(columns={'kg_despachados': 'KgEmbarcados'})
                # Asegurar que sea numérico
                volumenes_recientes['KgEmbarcados'] = pd.to_numeric(volumenes_recientes['KgEmbarcados'], errors='coerce').fillna(0)
        else:
            volumenes_recientes = pd.DataFrame(columns=group_cols + ['KgEmbarcados'])
        
        # Preparar datos de producción (kg_producidos)
        if 'kg_producidos' in retail_data.columns:
            # Convertir a numérico antes de sumar
            retail_data['kg_producidos'] = pd.to_numeric(retail_data['kg_producidos'], errors='coerce')
            produccion_reciente = retail_data.groupby(group_cols)['kg_producidos'].sum().reset_index()
            produccion_reciente = produccion_reciente.rename(columns={'kg_producidos': 'KgProducidos'})
            # Asegurar que sea numérico
            produccion_reciente['KgProducidos'] = pd.to_numeric(produccion_reciente['KgProducidos'], errors='coerce').fillna(0)
        else:
            produccion_reciente = pd.DataFrame(columns=group_cols + ['KgProducidos'])
    else:
        volumenes_recientes = pd.DataFrame(columns=['SKU', 'KgEmbarcados'])
        produccion_reciente = pd.DataFrame(columns=['SKU', 'KgProducidos'])
    
    # 7. Calcular MMPP usando la función probada de data_io
    if 'RECETAS' in sheets:
        receta_df = load_receta_sku(sheets["RECETAS"]) if not sheets["RECETAS"].empty else pd.DataFrame()
    elif 'RECETA_SKU' in sheets:
        receta_df = load_receta_sku(sheets["RECETA_SKU"]) if not sheets["RECETA_SKU"].empty else pd.DataFrame()
    else:
        receta_df = pd.DataFrame()
    
    
    # === 7. Calcular MMPP usando la función unificada de data_io (normal y óptimo) ===
    mmpp_almacenaje = pd.DataFrame()
    mmpp_almacenaje_optimo = pd.DataFrame()

    if not receta_df.empty and not info_fruta.empty:
        # Normal
        mmpp_almacenaje = compute_mmpp_unified(receta_df, info_fruta, df_granel).rename(columns={
            "MMPP (Fruta) (USD/kg)": "MMPP (Fruta) (USD/kg) (Calculado)",
            "Almacenaje": "Almacenaje (Calculado)", 
            "Proceso Granel (USD/kg)": "Proceso Granel (USD/kg) (Calculado)"
        })
        # Óptimo
        if 'df_granel_optimo' in locals() and df_granel_optimo is not None and not df_granel_optimo.empty:
            mmpp_almacenaje_optimo = compute_mmpp_unified(receta_df, info_fruta, df_granel_optimo).rename(columns={
                "MMPP (Fruta) (USD/kg)": "MMPP (Fruta) (USD/kg) (Calculado)",
                "Almacenaje": "Almacenaje (Calculado)", 
                "Proceso Granel (USD/kg)": "Proceso Granel (USD/kg) (Calculado)"
            })

    # Helper: aplica el pipeline de uniones y correcciones de forma DRY
    def _build_detalle_variant(_costos_detalle, _mmpp_calc):
        # 8) Merge con MMPP calculado
        if not _mmpp_calc.empty:
            if 'SKU' in _costos_detalle.columns and 'SKU' in _mmpp_calc.columns:
                _costos_detalle = _costos_detalle.copy()
                _mmpp_calc = _mmpp_calc.copy()
                _costos_detalle['SKU'] = _costos_detalle['SKU'].astype(str)
                _mmpp_calc['SKU'] = _mmpp_calc['SKU'].astype(str)

            _detalle = _costos_detalle.merge(_mmpp_calc, on="SKU", how="left")

            # Usar valores calculados si están; si no, dejar los existentes
            if "MMPP (Fruta) (USD/kg) (Calculado)" in _detalle.columns:
                _detalle["MMPP (Fruta) (USD/kg)"] = _detalle["MMPP (Fruta) (USD/kg) (Calculado)"].fillna(
                    _detalle.get("MMPP (Fruta) (USD/kg)", 0)
                )
                _detalle = _detalle.drop(columns=["MMPP (Fruta) (USD/kg) (Calculado)"])

            if "Almacenaje (Calculado)" in _detalle.columns:
                _detalle["Almacenaje MMPP"] = _detalle["Almacenaje (Calculado)"].fillna(
                    _detalle.get("Almacenaje MMPP", 0)
                )
                _detalle = _detalle.drop(columns=["Almacenaje (Calculado)"])

            if "Proceso Granel (USD/kg) (Calculado)" in _detalle.columns:
                _detalle["Proceso Granel (USD/kg)"] = _detalle["Proceso Granel (USD/kg) (Calculado)"].fillna(
                    _detalle.get("Proceso Granel (USD/kg)", 0)
                )
                _detalle = _detalle.drop(columns=["Proceso Granel (USD/kg) (Calculado)"])
        else:
            _detalle = _costos_detalle.copy()

        # 9) Merge con dimensiones
        if not dim.empty:
            if 'SKU' in _detalle.columns and 'SKU' in dim.columns:
                _detalle['SKU'] = _detalle['SKU'].astype(str)
                dim['SKU'] = dim['SKU'].astype(str)
            _detalle = _detalle.merge(dim, on="SKU", how="left")

        # 10) Corregir especies
        if not receta_df.empty and not info_fruta.empty:
            _detalle = correct_species_from_recipes(_detalle, receta_df, info_fruta)
            _detalle = ensure_list_species(_detalle, "Especie")

        # 11) Merge con precios
        if not latest_prices.empty:
            if "SKU-Cliente" in dim.columns and "SKU-Cliente" in latest_prices.columns and "SKU-Cliente" in _detalle.columns:
                _detalle = _detalle.merge(latest_prices, on="SKU-Cliente", how="left")
            else:
                if "SKU" in latest_prices.columns and "SKU" in _detalle.columns:
                    latest_by_sku = latest_prices.groupby("SKU")["PrecioVenta (USD/kg)"].last().reset_index()
                    _detalle['SKU'] = _detalle['SKU'].astype(str)
                    latest_by_sku['SKU'] = latest_by_sku['SKU'].astype(str)
                    _detalle = _detalle.merge(latest_by_sku, on="SKU", how="left")

        # 12) Volúmenes (kg_despachados)
        if not volumenes_recientes.empty and "SKU" in volumenes_recientes.columns and "SKU" in _detalle.columns:
            merge_cols = ['SKU', 'SKU-Cliente'] if ('SKU-Cliente' in _detalle.columns and 'SKU-Cliente' in volumenes_recientes.columns) else ['SKU']
            for col in merge_cols:
                if col in _detalle.columns and col in volumenes_recientes.columns:
                    _detalle[col] = _detalle[col].astype(str)
                    volumenes_recientes[col] = volumenes_recientes[col].astype(str)
            _detalle = _detalle.merge(volumenes_recientes[merge_cols + ["KgEmbarcados"]], on=merge_cols, how="left")

        # 13) Producción (kg_producidos)
        if not produccion_reciente.empty and "SKU" in produccion_reciente.columns and "SKU" in _detalle.columns:
            merge_cols = ['SKU', 'SKU-Cliente'] if ('SKU-Cliente' in _detalle.columns and 'SKU-Cliente' in produccion_reciente.columns) else ['SKU']
            for col in merge_cols:
                if col in _detalle.columns and col in produccion_reciente.columns:
                    _detalle[col] = _detalle[col].astype(str)
                    produccion_reciente[col] = produccion_reciente[col].astype(str)
            _detalle = _detalle.merge(produccion_reciente[merge_cols + ["KgProducidos"]], on=merge_cols, how="left")

        # Cast numérico seguro
        if 'KgEmbarcados' in _detalle.columns:
            _detalle['KgEmbarcados'] = pd.to_numeric(_detalle['KgEmbarcados'], errors='coerce').fillna(0)
        if 'KgProducidos' in _detalle.columns:
            _detalle['KgProducidos'] = pd.to_numeric(_detalle['KgProducidos'], errors='coerce').fillna(0)

        # 14) Recalcular totales (esto aplica signos y KPI de la app)
        _detalle = recalculate_totals(_detalle)
        return _detalle

    # Construir ambas variantes
    detalle        = _build_detalle_variant(costos_detalle,        mmpp_almacenaje)
    detalle_optimo = _build_detalle_variant(costos_detalle_optimo, mmpp_almacenaje_optimo)

    # (Opcional) Alinear SKUs entre ambas (por si difieren)
    if 'SKU' in detalle.columns and 'SKU' in detalle_optimo.columns:
        detalle['SKU'] = detalle['SKU'].astype(str)
        detalle_optimo['SKU'] = detalle_optimo['SKU'].astype(str)

        # 🔧 NUEVO: forzar unicidad por SKU para poder reindexar
        # Si prefieres otra política, ver alternativas más abajo
        detalle = detalle.sort_index().drop_duplicates(subset=['SKU'], keep='last')
        detalle_optimo = detalle_optimo.sort_index().drop_duplicates(subset=['SKU'], keep='last')

        sku_union = pd.Index(detalle['SKU']).union(pd.Index(detalle_optimo['SKU']))
        detalle        = detalle.set_index('SKU').reindex(sku_union).reset_index()
        detalle_optimo = detalle_optimo.set_index('SKU').reindex(sku_union).reset_index()

    return detalle, detalle_optimo, df_granel, df_granel_optimo, info_fruta
    
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

# ------------------------------
# Helpers
# ------------------------------
def _sum_mat_dir_por_mes_y_clave(df: pd.DataFrame, key_col: str, rolling_months: list) -> tuple[pd.Series, pd.Series]:
    """
    Retorna:
      - tot_por_mes: Serie indexada por mes con el total $ de materiales directos del DF
      - tot_por_clave_periodo: Serie indexada por key_col (SKU o Fruta_id) con el total $ en TODO el período
    """
    if df.empty or 'mes' not in df.columns or key_col not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    dfx = df.loc[df['mes'].isin(rolling_months)].copy()
    if dfx.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Asegurar columnas y tipos
    for c in ['cajas', 'costo_unit_caja', 'bolsas', 'costo_unit_bolsa']:
        if c not in dfx.columns:
            dfx[c] = 0
        dfx[c] = pd.to_numeric(dfx[c], errors='coerce').fillna(0)

    dfx['mat_dir_val'] = dfx['cajas'] * dfx['costo_unit_caja'] * -1 + dfx['bolsas'] * dfx['costo_unit_bolsa'] * -1

    # Total $ por mes
    tot_por_mes = dfx.groupby('mes')['mat_dir_val'].sum()

    # Total $ por clave en el período (ya sumado todos los meses)
    tot_por_clave_periodo = dfx.groupby(key_col)['mat_dir_val'].sum()

    return tot_por_mes, tot_por_clave_periodo

def calc_materiales_indirectos_y_directos(dfs: Dict[str, pd.DataFrame], rolling_months: list):
    """
    Devuelve un dict con:
      - 'mat_ind_mes': Serie por mes (Materiales totales - Directos Retail - Directos Granel)
      - 'retail': {'por_mes': Serie por mes, 'por_sku_periodo': Serie por SKU en el período}
      - 'granel': {'por_mes': Serie por mes, 'por_especie_periodo': Serie por Fruta_id en el período}
    """
    mayor = dfs['MAYOR']
    mat_total_mes = mayor.loc[
        (mayor['familia_cc'] == 'Materiales') & (mayor['mes'].isin(rolling_months))
    ].groupby('mes')['monto'].sum()

    # Directos
    ret_por_mes, ret_por_sku_periodo = _sum_mat_dir_por_mes_y_clave(
        dfs['INDICADORES_RETAIL'], 'SKU', rolling_months
    )
    gra_por_mes, gra_por_especie_periodo = _sum_mat_dir_por_mes_y_clave(
        dfs['INDICADORES_GRANEL'], 'Fruta_id', rolling_months
    )

    # Indirectos (residuo) por mes
    mat_ind_mes = (mat_total_mes
                   .sub(ret_por_mes, fill_value=0)
                   .sub(gra_por_mes, fill_value=0)).clip(upper=0)

    return {
        'mat_ind_mes': mat_ind_mes,
        'retail': {
            'por_mes': ret_por_mes,
            'por_sku_periodo': ret_por_sku_periodo
        },
        'granel': {
            'por_mes': gra_por_mes,
            'por_especie_periodo': gra_por_especie_periodo
        }
    }

def _clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.5
    if not np.isfinite(x):
        return 0.5
    return max(0.0, min(1.0, x))

# ------------------------------
# RETAIL
# ------------------------------
def compute_costos_retail(dfs: Dict[str, pd.DataFrame], rolling_months: list = None) -> pd.DataFrame:
    """
    Calcula todos los costos Retail por SKU usando promedio móvil (idempotente, sin mutar MAYOR).
    """
    if rolling_months is None:
        rolling_months = get_rolling_months(dfs, max_months=12)
    
    drivers_retail = build_drivers_retail(dfs['INDICADORES_RETAIL'], rolling_months)
    if not drivers_retail or 'kg_producidos' not in drivers_retail:
        return pd.DataFrame()
    
    skus = drivers_retail['kg_producidos'].columns.tolist()
    resultado = pd.DataFrame(index=skus)

    # ---- Materiales (directos e indirectos) precomputados ----
    mat_pack = calc_materiales_indirectos_y_directos(dfs, rolling_months)
    mat_ind_mes = mat_pack['mat_ind_mes']                           # Serie por mes
    mat_dir_por_sku_periodo = mat_pack['retail']['por_sku_periodo'] # Serie $ por SKU (todo el período)

    # ---- Denominador (kg) acumulado del período ----
    retail_den_acum = pd.Series(0.0, index=skus)
    retail_den_acum_desp = pd.Series(0.0, index=skus)
    for mes in rolling_months:
        if mes in drivers_retail['kg_producidos'].index:
            retail_den_acum += drivers_retail['kg_producidos'].loc[mes].reindex(skus).fillna(0)
            retail_den_acum_desp += drivers_retail['kg_despachados'].loc[mes].reindex(skus).fillna(0)

    # ---- Materiales Directos unitarios (USD/kg) ----
    denom_safe = retail_den_acum.replace(0, np.nan)
    resultado['Materiales_Directos'] = (
        mat_dir_por_sku_periodo.reindex(skus).fillna(0) / denom_safe
    ).fillna(0)

    # Calcular Guarda PT
    for mes in rolling_months:
        guarda = dfs['MAYOR'].loc[
            (dfs['MAYOR']['familia_cc'] == 'Guarda') &
            (dfs['MAYOR']['mes'] == mes), 'monto'
        ].sum()
        split_pt = dfs['MAYOR'].loc[
            (dfs['MAYOR']['familia_cc'] == 'split_pt') &
            (dfs['MAYOR']['mes'] == mes), 'monto'
        ].sum()
        split = _clamp01(split_pt)
        monto_escalar = guarda * split

        nueva_fila = {
            'familia_cc': 'Guarda PT',
            'mes': mes,
            'monto': monto_escalar,
        }
        dfs['MAYOR'] = pd.concat([dfs['MAYOR'], pd.DataFrame([nueva_fila])], ignore_index=True)

    # === Comex mixto (por MES): Directo + Indirecto ===
    # 1) Datos retail del período (mes, SKU, kg despachados + columnas de comex directo)
    df_retail = dfs['INDICADORES_RETAIL']
    df_ret = df_retail[df_retail['mes'].isin(rolling_months)].copy()

    # Asegurar tipos numéricos
    if 'kg_despachados' in df_ret.columns:
        df_ret['kg_despachados'] = pd.to_numeric(df_ret['kg_despachados'], errors='coerce').fillna(0)
    else:
        df_ret['kg_despachados'] = 0.0

    cols_comex_directo = [
        'flete_terrestre_usd', 'flete_maritimo_usd',
        'flete_terrestre', 'flete_maritimo',
        'costo_flete_terrestre', 'costo_flete_maritimo'
    ]
    presentes = [c for c in cols_comex_directo if c in df_ret.columns]
    for c in presentes:
        df_ret[c] = pd.to_numeric(df_ret[c], errors='coerce').fillna(0)
    if not presentes:
        df_ret['__comex_directo_usd'] = 0.0
        presentes = ['__comex_directo_usd']

    # USD directos por fila y luego agrupar por (mes, SKU)
    df_ret['__comex_directo_total_usd'] = df_ret[presentes].sum(axis=1)
    dir_sku_mes = (
        df_ret.groupby(['mes', 'SKU'], as_index=False)
            .agg(comex_dir_usd=('__comex_directo_total_usd','sum'),
                kg_desp=('kg_despachados','sum'))
    )

    # 2) Total kg por mes (para porcentajes)
    kg_mes = (
        dir_sku_mes.groupby('mes', as_index=False)['kg_desp']
                .sum().rename(columns={'kg_desp':'kg_mes_total'})
    )

    # 3) Total Comex en MAYOR (por mes) y split retail
    df_mayor_comex = dfs['MAYOR'][
        (dfs['MAYOR']['familia_cc'] == 'Comex') &
        (dfs['MAYOR']['mes'].isin(rolling_months))
    ].copy()
    split_retail_pct = 1.0
    cfg_split = dfs['CONFIG_SPLITS'][dfs['CONFIG_SPLITS']['concepto'] == 'Comex']
    if not cfg_split.empty and pd.notna(cfg_split['split_retail_pct'].iloc[0]):
        split_retail_pct = float(cfg_split['split_retail_pct'].iloc[0])

    comex_mayor_mes = (df_mayor_comex.groupby('mes', as_index=False)['monto']
                                .sum())
    comex_mayor_mes['comex_mayor_retail_usd'] = comex_mayor_mes['monto'] * split_retail_pct
    comex_mayor_mes = comex_mayor_mes[['mes','comex_mayor_retail_usd']]

    # 4) Directo total por mes (USD) para compararlo con el MAYOR
    dir_mes = (dir_sku_mes.groupby('mes', as_index=False)['comex_dir_usd']
                        .sum().rename(columns={'comex_dir_usd':'comex_dir_mes_usd'}))

    # 5) Pool indirecto por mes (USD, negativo para costos)
    pool = comex_mayor_mes.merge(dir_mes, on='mes', how='outer').fillna(0)
    # Diferencia en valor absoluto para evitar signos cruzados; el pool es costo (negativo)
    pool['comex_ind_pool_mes_usd'] = -1.0 * (pool['comex_mayor_retail_usd'].abs() - pool['comex_dir_mes_usd'].abs()).clip(lower=0)

    # 6) Reparto mensual del pool según % de kg del mes; calcular USD indirectos por (mes,SKU)
    dir_sku_mes = dir_sku_mes.merge(kg_mes, on='mes', how='left').fillna({'kg_mes_total':0})
    dir_sku_mes = dir_sku_mes.merge(pool[['mes','comex_ind_pool_mes_usd']], on='mes', how='left').fillna({'comex_ind_pool_mes_usd':0})

    # % de cada SKU ese mes (si kg_mes_total==0, todo queda 0)
    share = pd.Series(0.0, index=dir_sku_mes.index)
    mask = dir_sku_mes['kg_mes_total'] > 0
    share.loc[mask] = dir_sku_mes.loc[mask, 'kg_desp'] / dir_sku_mes.loc[mask, 'kg_mes_total']

    dir_sku_mes['comex_ind_usd'] = dir_sku_mes['comex_ind_pool_mes_usd'] * share

    # 7) Sumar USD directos + indirectos a nivel SKU (en TODO el período)
    tot_por_sku = (
        dir_sku_mes.groupby('SKU', as_index=False)
                .agg(comex_dir_usd_sku=('comex_dir_usd','sum'),
                        comex_ind_usd_sku=('comex_ind_usd','sum'),
                        kg_desp_sku=('kg_desp','sum'))
    )

    # 8) Llevar a USD/kg por SKU del período (denominador: kg despachados del período por SKU)
    tot_por_sku['Comex_unit'] = 0.0
    den0 = tot_por_sku['kg_desp_sku'] > 0
    tot_por_sku.loc[den0, 'Comex_unit'] = (
        (tot_por_sku.loc[den0, 'comex_dir_usd_sku'] + tot_por_sku.loc[den0, 'comex_ind_usd_sku'])
        / tot_por_sku.loc[den0, 'kg_desp_sku']
    )

    # 9) Escribir columna final en resultado (alineando SKUs)
    resultado['Comex'] = (
        tot_por_sku.set_index('SKU')['Comex_unit']
                .reindex(skus).astype(float).fillna(0)
    )

    # ---- Resto de conceptos (incluye Materiales_Indirectos) ----
    for _, cfg in dfs['CONFIG_SPLITS'].iterrows():
        concepto = cfg['concepto']
        split_retail_pct = _clamp01(cfg.get('split_retail_pct', 0.5))
        driver_interno = cfg.get('driver_interno', 'kg_producidos') or 'kg_producidos'
        denominador = cfg.get('denominador', 'kg_producidos') or 'kg_producidos'

        serie_costos_acum = pd.Series(0.0, index=skus)
        serie_den_acum = pd.Series(0.0, index=skus)

        if concepto in ['Materiales_Directos', 'Comex']:
            break

        # datos del MAYOR del concepto (cuando aplica)
        if concepto != 'Materiales_Indirectos':
            df_mc = dfs['MAYOR'].loc[
                (dfs['MAYOR']['familia_cc'] == concepto) & (dfs['MAYOR']['mes'].isin(rolling_months))
            ]
        else:
            df_mc = None  # no usamos MAYOR directo; usamos mat_ind_mes

        for mes in rolling_months:
            # Monto mensual del concepto para Retail (escalar)
            if concepto in ['MO_Directa', 'MO_Indirecta']:
                # split por HH
                hh = dfs['MAYOR'].loc[
                    (dfs['MAYOR']['familia_cc'] == 'split_hh_retail') &
                    (dfs['MAYOR']['mes'] == mes), 'monto'
                ]
                split = _clamp01(hh.iloc[0]) if len(hh) else 0.5

                monto_base = 0.0
                if df_mc is not None:
                    monto_base = df_mc.loc[df_mc['mes'].eq(mes), 'monto'].sum()
                monto_escalar = monto_base * split

            elif concepto == 'Materiales_Indirectos':
                monto_base = float(mat_ind_mes.get(mes, 0.0))
                monto_escalar = monto_base * split_retail_pct

            else:
                monto_base = df_mc.loc[df_mc['mes'].eq(mes), 'monto'].sum() if df_mc is not None else 0.0
                monto_escalar = monto_base * split_retail_pct

            # Drivers de distribución
            pct_key = f'pct_{driver_interno}'
            den_key = denominador

            pct_row = drivers_retail.get(pct_key)
            den_row = drivers_retail.get(den_key)

            pct = pct_row.loc[mes].reindex(skus).fillna(0) if (pct_row is not None and mes in pct_row.index) else pd.Series(0.0, index=skus)
            den = den_row.loc[mes].reindex(skus).fillna(0) if (den_row is not None and mes in den_row.index) else pd.Series(0.0, index=skus)

            if pct.sum() > 0 and monto_escalar != 0:
                serie_costos_acum += (pct * monto_escalar)
            # Denominador representa los kg del período (promedio móvil)
            serie_den_acum += den

        resultado[concepto] = (serie_costos_acum / serie_den_acum.replace(0, np.nan)).fillna(0)

    # Renombrados y totales (compatibilidad)
    column_renames = {
        'MO_Directa': 'MO Directa',
        'MO_Indirecta': 'MO Indirecta',
        'Materiales_Directos': 'Materiales Directos',
        'Materiales_Indirectos': 'Materiales Indirectos', 
        'Mantención': 'Mantención',
        'Fletes_Internos': 'Fletes Internos',
        'MMPP_Fruta': 'MMPP (Fruta) (USD/kg)',
    }
    for a, b in column_renames.items():
        if a in resultado.columns:
            resultado = resultado.rename(columns={a: b})
    
    conceptos_directos = ['MMPP (Fruta) (USD/kg)', 'Materiales Directos', 'MO Directa', 'Laboratorio', 'Mantención']
    cols_dir = [c for c in conceptos_directos if c in resultado.columns]
    resultado['Costos_Directos'] = resultado[cols_dir].sum(axis=1) if cols_dir else 0

    # Todo lo demás se considera indirecto (excepto totales)
    excl = set(cols_dir + ['Costos_Directos', 'Costos_Indirectos', 'Costos_Totales'])
    cols_ind = [c for c in resultado.columns if c not in excl]
    resultado['Costos_Indirectos'] = resultado[cols_ind].sum(axis=1) if cols_ind else 0

    resultado['Costos_Totales'] = resultado['Costos_Directos'] + resultado['Costos_Indirectos']
    return resultado


# ------------------------------
# GRANEL
# ------------------------------
def compute_costos_granel(dfs: Dict[str, pd.DataFrame], rolling_months: list = None, especie_key_map: Dict[str, str] = key_map) -> pd.DataFrame:
    """
    Calcula todos los costos Granel por Especie usando promedio móvil (idempotente, sin mutar MAYOR).
    """
    if rolling_months is None:
        rolling_months = get_rolling_months(dfs, max_months=12)
    
    drivers_granel = build_drivers_granel(dfs['INDICADORES_GRANEL'], rolling_months)
    if not drivers_granel or 'kg_producidos' not in drivers_granel:
        return pd.DataFrame()
    
    especies = drivers_granel['kg_producidos'].columns.tolist()
    resultado = pd.DataFrame(index=especies)

    # ---- Materiales (directos e indirectos) precomputados ----
    mat_pack = calc_materiales_indirectos_y_directos(dfs, rolling_months)
    mat_ind_mes = mat_pack['mat_ind_mes']                                 # Serie por mes
    mat_dir_por_especie_periodo = mat_pack['granel']['por_especie_periodo'] # Serie $ por Fruta_id

    # ---- Denominador (kg) acumulado del período ----
    den_acum = pd.Series(0.0, index=especies)
    for mes in rolling_months:
        if mes in drivers_granel['kg_producidos'].index:
            den_acum += drivers_granel['kg_producidos'].loc[mes].reindex(especies).fillna(0)

    # ---- Materiales Directos unitarios (USD/kg) ----
    resultado['Materiales_Directos'] = (
        mat_dir_por_especie_periodo.reindex(especies).fillna(0) / den_acum.replace(0, np.nan)
    ).fillna(0)

    # ---- Resto de conceptos (incluye Materiales_Indirectos) ----
    for _, cfg in dfs['CONFIG_SPLITS'].iterrows():
        concepto = cfg['concepto']
        split_granel_pct = _clamp01(cfg.get('split_granel_pct', 0.5))
        driver_interno = cfg.get('driver_interno', 'kg_producidos') or 'kg_producidos'
        denominador = cfg.get('denominador', 'kg_producidos') or 'kg_producidos'

        serie_costos_acum = pd.Series(0.0, index=especies)
        serie_den_acum = pd.Series(0.0, index=especies)

        if concepto in ['Materiales_Directos', 'Comex']:
            continue

        if concepto != 'Materiales_Indirectos':
            df_mc = dfs['MAYOR'].loc[
                (dfs['MAYOR']['familia_cc'] == concepto) & (dfs['MAYOR']['mes'].isin(rolling_months))
            ]
        else:
            df_mc = None

        for mes in rolling_months:
            # Monto mensual del concepto para Granel (escalar)
            if concepto in ['MO_Directa', 'MO_Indirecta']:
                # split HH: para granel usamos (1 - split_hh_retail)
                hh = dfs['MAYOR'].loc[
                    (dfs['MAYOR']['familia_cc'] == 'split_hh_retail') &
                    (dfs['MAYOR']['mes'] == mes), 'monto'
                ]
                split = 1.0 - _clamp01(hh.iloc[0]) if len(hh) else 0.5
                monto_base = 0.0
                if df_mc is not None:
                    monto_base = df_mc.loc[df_mc['mes'].eq(mes), 'monto'].sum()
                monto_escalar = monto_base * split

            elif concepto == 'Materiales_Indirectos':
                monto_base = float(mat_ind_mes.get(mes, 0.0))
                monto_escalar = monto_base * split_granel_pct

            else:
                monto_base = df_mc.loc[df_mc['mes'].eq(mes), 'monto'].sum() if df_mc is not None else 0.0
                monto_escalar = monto_base * split_granel_pct

            # Drivers
            pct_key = f'pct_{driver_interno}'
            den_key = denominador

            pct_row = drivers_granel.get(pct_key)
            den_row = drivers_granel.get(den_key)

            pct = pct_row.loc[mes].reindex(especies).fillna(0) if (pct_row is not None and mes in pct_row.index) \
                else pd.Series(0.0, index=especies)
            den = den_row.loc[mes].reindex(especies).fillna(0) if (den_row is not None and mes in den_row.index) \
                else pd.Series(0.0, index=especies)

            if pct.sum() > 0 and monto_escalar != 0:
                serie_costos_acum += (pct * monto_escalar)
            serie_den_acum += den

        resultado[concepto] = (serie_costos_acum / serie_den_acum.replace(0, np.nan)).fillna(0)

    # Renombrados y totales
    column_renames = {
        'MO_Directa': 'MO Directa',
        'MO_Indirecta': 'MO Indirecta',
        'Materiales_Directos': 'Materiales Directos',
        'Materiales_Indirectos': 'Materiales Indirectos',
        'Mantención': 'Mantención',
        'Fletes_Internos': 'Fletes Internos',
    }
    for a, b in column_renames.items():
        if a in resultado.columns:
            resultado = resultado.rename(columns={a: b})
    
    conceptos_directos = ['Materiales Directos', 'MO Directa', 'Laboratorio', 'Mantención']
    cols_dir = [c for c in conceptos_directos if c in resultado.columns]
    resultado['Costos_Directos'] = resultado[cols_dir].sum(axis=1) if cols_dir else 0

    excl = set(cols_dir + ['Costos_Directos', 'Costos_Indirectos', 'Costos_Totales'])
    cols_ind = [c for c in resultado.columns if c not in excl]
    resultado['Costos_Indirectos'] = resultado[cols_ind].sum(axis=1) if cols_ind else 0

    resultado['Costos_Totales'] = resultado['Costos_Directos'] + resultado['Costos_Indirectos']

    def _replicar_variantes_por_peer(resultado, key_map):
        import pandas as pd
        if resultado.empty or key_map is None:
            return resultado

        m = pd.DataFrame({"Fruta_id": list(key_map.keys()),
                            "GuardaKey": list(key_map.values())})
        m["Fruta_id"] = m["Fruta_id"].astype(str)
        m["GuardaKey"] = m["GuardaKey"].astype(str)

        # Agrupa variantes por GuardaKey
        gk_to_frutas = m.groupby("GuardaKey")["Fruta_id"].apply(list)

        to_add_rows = []
        for gk, frutas in gk_to_frutas.items():
            # ¿Qué variantes ya existen en resultado?
            presentes = [f for f in frutas if f in resultado.index]
            if not presentes:
                # Si ninguna variante de este grupo existe, no hay fuente para copiar.
                continue
            fuente_idx = presentes[0]        # toma la primera presente como "peer" fuente
            source_row = resultado.loc[fuente_idx]

            # 🔹 NUEVO BLOQUE corregido: sobreescribir filas existentes con valores 0
            for f in presentes:
                if f == fuente_idx:
                    continue

                fila = resultado.loc[f]
                # Detectar columnas numéricas una sola vez, desde el DataFrame
                numeric_cols = resultado.select_dtypes(include=[np.number]).columns

                # Si todos los valores numéricos son 0 o NaN → sobrescribir
                if (fila[numeric_cols].fillna(0).abs() <= 1e-12).all():
                    resultado.loc[f, numeric_cols] = source_row[numeric_cols]

            # Para cada faltante, clona la fila fuente
            faltantes = [f for f in frutas if f not in resultado.index]
            for f in faltantes:
                r = source_row.copy()
                r.name = f
                to_add_rows.append(r)

        if not to_add_rows:
            return resultado

        nuevos = pd.DataFrame(to_add_rows)
        nuevos.index.name = resultado.index.name
        # Concatena sin sobre-escribir existentes
        resultado = pd.concat([resultado, nuevos], axis=0)
        return resultado

    # --- al final de tu compute_costos_granel, JUSTO antes de retornar:
    resultado = _replicar_variantes_por_peer(resultado, especie_key_map)  # <- 1 línea
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




def get_rolling_months(dfs: Dict[str, pd.DataFrame], max_months: int = None) -> list:
    """
    Obtiene los últimos N meses disponibles para promedio móvil.
    
    Args:
        dfs: Diccionario de DataFrames
        max_months: Número máximo de meses a incluir (None = todos los meses)
        
    Returns:
        Lista de los últimos N meses disponibles
    """
    all_months = get_available_months(dfs)
    
    if not all_months:
        return []
    
    # Si max_months es None, usar todos los meses
    if max_months is None:
        return all_months
    
    # Tomar los últimos max_months meses
    return all_months[-max_months:]




def compute_full_cost_analysis(uploaded_bytes: bytes, max_months: int = None) -> Dict[str, pd.DataFrame]:
    """
    Función principal que ejecuta el análisis completo de costos usando promedio móvil.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        max_months: Número máximo de meses para el periodo (None = todos)
        
    Returns:
        Diccionario con resultados de Retail y Granel
        
    Raises:
        ValueError: Si hay problemas con los datos o validaciones
    """
    # Leer datos
    dfs = read_source(uploaded_bytes)
    
    # Validar estructura
    validate_inputs(dfs)
    
    # Obtener meses para el periodo
    rolling_months = get_rolling_months(dfs, max_months=max_months)
    if not rolling_months:
        raise ValueError("No hay datos disponibles para calcular costos")
    
    # Calcular costos usando promedio móvil
    resultados = {}
    
    # try:  # Comentado para debugging
    resultados['retail'] = compute_costos_retail(dfs, rolling_months)
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error calculando costos Retail: {str(e)}")
    
    # try:  # Comentado para debugging
    resultados['granel'] = compute_costos_granel(dfs, rolling_months, key_map)
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error calculando costos Granel: {str(e)}")
        
    resultados['method'] = 'complex'
    
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
    from src.data_io import read_workbook, load_info_fruta, build_fact_granel_ponderado
    
    # Leer hojas usando data_io
    sheets = read_workbook(uploaded_bytes)
    
    # Fallback: usar cost_engine
    resultados = compute_full_cost_analysis(uploaded_bytes)
    
    if 'granel' not in resultados:
        # Crear DataFrames vacíos si no hay datos de granel
        df_granel = pd.DataFrame()
        df_granel_ponderado = pd.DataFrame()
        return df_granel, df_granel_ponderado        
    
    df_granel = resultados['granel']
    
    # Procesar kg_producidos desde INDICADORES_GRANEL
    if 'INDICADORES_GRANEL' in sheets:
        granel_data = sheets['INDICADORES_GRANEL'].copy()
        
        if not granel_data.empty and 'Fruta_id' in granel_data.columns and 'kg_producidos' in granel_data.columns:
            # Convertir a numérico antes de sumar
            granel_data['kg_producidos'] = pd.to_numeric(granel_data['kg_producidos'], errors='coerce')
            # Sumar kg_producidos por Fruta_id
            kg_producidos_granel = granel_data.groupby('Fruta_id')['kg_producidos'].sum().reset_index()
            kg_producidos_granel = kg_producidos_granel.rename(columns={'kg_producidos': 'KgProducidos'})
            # Asegurar que sea numérico
            kg_producidos_granel['KgProducidos'] = pd.to_numeric(kg_producidos_granel['KgProducidos'], errors='coerce').fillna(0)
        else:
            kg_producidos_granel = pd.DataFrame(columns=['Fruta_id', 'KgProducidos'])
    else:
        kg_producidos_granel = pd.DataFrame(columns=['Fruta_id', 'KgProducidos'])
    
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

    # Enriquecer hoja FRUTA con Almacenaje calculado (en memoria)
    dfs_all = read_source(uploaded_bytes)
    rolling_months = get_rolling_months(dfs_all)
    alm_por_fruta = compute_almacenaje_mmpp_por_fruta(dfs_all, rolling_months, key_map)
    if not alm_por_fruta.empty and not info_fruta.empty and 'Fruta_id' in info_fruta.columns:
        info_fruta['Fruta_id'] = info_fruta['Fruta_id'].astype(str)
        alm_por_fruta['Fruta_id'] = alm_por_fruta['Fruta_id'].astype(str)
        # Sobrescribe/crea la columna 'Almacenaje' en FRUTA
        info_fruta = info_fruta.drop(columns=['Almacenaje'], errors='ignore')
        info_fruta = info_fruta.merge(alm_por_fruta, on='Fruta_id', how='left')
    
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
    
    # Merge con kg_producidos
    if not kg_producidos_granel.empty and 'Fruta_id' in kg_producidos_granel.columns:
        # Asegurar que Fruta_id sea string en ambos DataFrames
        df_granel_ponderado['Fruta_id'] = df_granel_ponderado['Fruta_id'].astype(str)
        kg_producidos_granel['Fruta_id'] = kg_producidos_granel['Fruta_id'].astype(str)
        
        df_granel_ponderado = df_granel_ponderado.merge(
            kg_producidos_granel[['Fruta_id', 'KgProducidos']], 
            on='Fruta_id', how='left'
        )
    
    # Asegurar que la columna de producción sea numérica
    if 'KgProducidos' in df_granel_ponderado.columns:
        df_granel_ponderado['KgProducidos'] = pd.to_numeric(df_granel_ponderado['KgProducidos'], errors='coerce').fillna(0)

    # Optimos
    df_granel_optimo = build_fact_granel_ponderado(sheets["OPTIMOS_GRANEL"], info_fruta)
    
    return df_granel_ponderado, info_fruta, df_granel_optimo
    
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


def build_cost_engine_pipeline(uploaded_bytes: bytes, max_months: int = None) -> dict:
    """
    Pipeline completo que integra cost_engine con data_io para construir todos los datos necesarios.
    
    Args:
        uploaded_bytes: Bytes del archivo Excel
        max_months: Número máximo de meses para el periodo (None = todos)
        
    Returns:
        Dict con todos los DataFrames necesarios para la aplicación
    """
    # try:  # Comentado para debugging
    # Obtener meses disponibles
    meses_disponibles = get_available_months_from_excel(uploaded_bytes)
    
    if not meses_disponibles:
        raise ValueError("No hay meses disponibles en los datos")
    
    # Obtener meses para el periodo
    rolling_months = get_rolling_months(read_source(uploaded_bytes), max_months=max_months)
    
    # Construir todos los componentes usando promedio móvil
    detalle, detalle_optimo, df_granel, df_granel_optimo, info_fruta = build_detalle_from_cost_engine(uploaded_bytes)
    
    # Leer datos adicionales
    dfs = read_source(uploaded_bytes)
    
    # Preparar receta e info_fruta si existen
    receta_df = None
    
    if 'RECETAS' in dfs:
        receta_df = dfs['RECETAS']
    
    return {
        'detalle': detalle,
        'detalle_optimo': detalle_optimo,
        'df_granel_ponderado': df_granel,
        'df_granel_optimo': df_granel_optimo,
        'receta_df': receta_df,
        'info_df': info_fruta,
        'rolling_months': rolling_months,
        'months_count': len(rolling_months),
        'meses_disponibles': meses_disponibles,
        'cost_engine_results': compute_full_cost_analysis(uploaded_bytes, max_months=max_months),
        'method': 'complex'
    }
    
    # except Exception as e:  # Comentado para debugging
    #     raise ValueError(f"Error en pipeline cost_engine: {str(e)}")
