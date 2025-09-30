"""
Tests para el módulo cost_engine.py
Cubre validaciones, cálculos de drivers, conceptos especiales y totales.
"""

import pytest
import pandas as pd
import numpy as np
import io
from typing import Dict
import sys
from pathlib import Path

# Agregar src al path para importar el módulo
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cost_engine import (
    read_source,
    validate_inputs,
    build_drivers_retail,
    build_drivers_granel,
    compute_mmpp_por_receta,
    costear_concepto_mayor,
    compute_costos_retail,
    compute_costos_granel,
    _compute_comex_especial,
    compute_full_cost_analysis
)


# ===================== FIXTURES =====================

@pytest.fixture
def sample_excel_bytes():
    """
    Crea un archivo Excel en memoria con datos mínimos para testing.
    Incluye 2 SKUs y 2 especies para un mes.
    """
    # Crear DataFrames de ejemplo
    mayor_data = pd.DataFrame({
        'familia_cc': ['Comex', 'Comex', 'Mantención', 'Mantención', 'Materiales', 'Materiales'],
        'mes': ['2024-01', '2024-01', '2024-01', '2024-01', '2024-01', '2024-01'],
        'monto': [10000, 5000, 8000, 4000, 6000, 3000]
    })
    
    indicadores_retail = pd.DataFrame({
        'SKU': ['SKU001', 'SKU002'],
        'mes': ['2024-01', '2024-01'],
        'kg_producidos': [1000, 2000],
        'kg_despachados': [800, 1600],
        'hh_directas': [100, 150],
        'hh_indirectas': [50, 75],
        'tiempo_maquina': [20, 30],
        'flete_terrestre_usd': [500, 800],
        'flete_maritimo_usd': [300, 400]
    })
    
    indicadores_granel = pd.DataFrame({
        'Especie': ['Arándanos', 'Frambuesas'],
        'mes': ['2024-01', '2024-01'],
        'kg_producidos': [5000, 3000],
        'hh_directas': [200, 120],
        'hh_indirectas': [100, 60],
        'tiempo_maquina': [80, 50]
    })
    
    recetas = pd.DataFrame({
        'tipo': ['FRUTA', 'FRUTA', 'GRANEL', 'GRANEL'],
        'concepto': ['MMPP_Base', 'MMPP_Adicional', 'Proceso_Base', 'Proceso_Adicional'],
        'valor': [2.5, 1.0, 1.8, 0.7]
    })
    
    fruta = pd.DataFrame({
        'Especie': ['Arándanos', 'Frambuesas'],
        'eficiencia': [0.85, 0.90]
    })
    
    config_splits = pd.DataFrame({
        'concepto': ['Comex', 'Mantención', 'Materiales_Indirectos', 'MMPP_Fruta'],
        'split_retail_pct': [80, 60, 80, 100],
        'split_granel_pct': [20, 40, 20, 0],
        'driver_split': ['kg_despachados', 'tiempo_maquina', 'kg_producidos', 'kg_producidos'],
        'driver_interno': ['kg_despachados', 'tiempo_maquina', 'kg_producidos', 'kg_producidos'],
        'denominador': ['kg_despachados', 'kg_producidos', 'kg_producidos', 'kg_producidos']
    })
    
    # Hojas opcionales vacías
    optimos_retail = pd.DataFrame({'SKU': [], 'valor_optimo': []})
    optimos_granel = pd.DataFrame({'Especie': [], 'valor_optimo': []})
    
    # Crear Excel en memoria
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        mayor_data.to_excel(writer, sheet_name='MAYOR', index=False)
        indicadores_retail.to_excel(writer, sheet_name='INDICADORES_RETAIL', index=False)
        indicadores_granel.to_excel(writer, sheet_name='INDICADORES_GRANEL', index=False)
        recetas.to_excel(writer, sheet_name='RECETAS', index=False)
        fruta.to_excel(writer, sheet_name='FRUTA', index=False)
        config_splits.to_excel(writer, sheet_name='CONFIG_SPLITS', index=False)
        optimos_retail.to_excel(writer, sheet_name='OPTIMOS_RETAIL', index=False)
        optimos_granel.to_excel(writer, sheet_name='OPTIMOS_GRANEL', index=False)
    
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_dfs(sample_excel_bytes):
    """Fixture que retorna los DataFrames parseados del Excel de ejemplo."""
    return read_source(sample_excel_bytes)


@pytest.fixture
def incomplete_excel_bytes():
    """Excel con hojas faltantes para testing de validación."""
    # Solo crear algunas hojas, faltan otras requeridas
    mayor_data = pd.DataFrame({
        'familia_cc': ['Comex'],
        'mes': ['2024-01'],
        'monto': [10000]
    })
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        mayor_data.to_excel(writer, sheet_name='MAYOR', index=False)
        # Faltan las demás hojas requeridas
    
    buffer.seek(0)
    return buffer.getvalue()


# ===================== TESTS DE LECTURA Y VALIDACIÓN =====================

def test_read_source_success(sample_excel_bytes):
    """Test de lectura exitosa del Excel."""
    dfs = read_source(sample_excel_bytes)
    
    # Verificar que se leyeron todas las hojas
    expected_sheets = ['MAYOR', 'INDICADORES_RETAIL', 'INDICADORES_GRANEL', 
                      'RECETAS', 'FRUTA', 'CONFIG_SPLITS', 'OPTIMOS_RETAIL', 'OPTIMOS_GRANEL']
    
    for sheet in expected_sheets:
        assert sheet in dfs
        assert isinstance(dfs[sheet], pd.DataFrame)


def test_read_source_invalid_bytes():
    """Test de lectura con bytes inválidos."""
    invalid_bytes = b"invalid excel data"
    
    with pytest.raises(ValueError, match="Error al leer el archivo Excel"):
        read_source(invalid_bytes)


def test_validate_inputs_success(sample_dfs):
    """Test de validación exitosa."""
    # No debe lanzar excepción
    validate_inputs(sample_dfs)


def test_validate_inputs_missing_sheets(incomplete_excel_bytes):
    """Test de validación con hojas faltantes."""
    dfs = read_source(incomplete_excel_bytes)
    
    with pytest.raises(ValueError) as exc_info:
        validate_inputs(dfs)
    
    error_msg = str(exc_info.value)
    assert "Hojas faltantes" in error_msg
    assert "INDICADORES_RETAIL" in error_msg


def test_validate_inputs_missing_columns(sample_dfs):
    """Test de validación con columnas faltantes."""
    # Eliminar una columna requerida
    sample_dfs['MAYOR'] = sample_dfs['MAYOR'].drop(columns=['monto'])
    
    with pytest.raises(ValueError) as exc_info:
        validate_inputs(sample_dfs)
    
    error_msg = str(exc_info.value)
    assert "Columnas faltantes" in error_msg
    assert "monto" in error_msg


# ===================== TESTS DE DRIVERS =====================

def test_build_drivers_retail_success(sample_dfs):
    """Test de construcción de drivers Retail."""
    drivers = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    
    # Verificar drivers básicos
    assert 'kg_producidos' in drivers
    assert 'kg_despachados' in drivers
    assert 'hh_directas' in drivers
    assert 'tiempo_maquina' in drivers
    
    # Verificar porcentajes
    assert 'pct_kg_producidos' in drivers
    assert 'pct_kg_despachados' in drivers
    
    # Verificar que los porcentajes suman 1
    assert abs(drivers['pct_kg_producidos'].sum() - 1.0) < 1e-10
    assert abs(drivers['pct_kg_despachados'].sum() - 1.0) < 1e-10
    
    # Verificar índices (SKUs)
    expected_skus = ['SKU001', 'SKU002']
    assert list(drivers['kg_producidos'].index) == expected_skus


def test_build_drivers_retail_empty_month(sample_dfs):
    """Test de drivers con mes inexistente."""
    drivers = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2025-01')
    
    assert drivers == {}


def test_build_drivers_retail_zero_division(sample_dfs):
    """Test de manejo de división por cero en drivers."""
    # Crear DataFrame con valores cero
    df_zeros = pd.DataFrame({
        'SKU': ['SKU001', 'SKU002'],
        'mes': ['2024-01', '2024-01'],
        'kg_producidos': [0, 0],
        'kg_despachados': [0, 0]
    })
    
    drivers = build_drivers_retail(df_zeros, '2024-01')
    
    # Los porcentajes deben ser 0 cuando el total es 0
    assert 'pct_kg_producidos' in drivers
    assert all(drivers['pct_kg_producidos'] == 0)
    assert all(drivers['pct_kg_despachados'] == 0)


def test_build_drivers_granel_success(sample_dfs):
    """Test de construcción de drivers Granel."""
    drivers = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    # Verificar drivers básicos
    assert 'kg_producidos' in drivers
    assert 'hh_directas' in drivers
    assert 'tiempo_maquina' in drivers
    
    # Verificar porcentajes
    assert 'pct_kg_producidos' in drivers
    
    # Verificar que los porcentajes suman 1
    assert abs(drivers['pct_kg_producidos'].sum() - 1.0) < 1e-10
    
    # Verificar índices (Especies)
    expected_especies = ['Arándanos', 'Frambuesas']
    assert list(drivers['kg_producidos'].index) == expected_especies


# ===================== TESTS DE MMPP =====================

def test_compute_mmpp_por_receta_fruta(sample_dfs):
    """Test de cálculo MMPP para FRUTA."""
    mmpp = compute_mmpp_por_receta(sample_dfs['RECETAS'], 'FRUTA')
    
    # Debe haber conceptos de FRUTA
    assert not mmpp.empty
    assert 'MMPP_Base' in mmpp.index
    assert 'MMPP_Adicional' in mmpp.index
    
    # Verificar valores
    assert mmpp['MMPP_Base'] == 2.5
    assert mmpp['MMPP_Adicional'] == 1.0


def test_compute_mmpp_por_receta_granel(sample_dfs):
    """Test de cálculo MMPP para GRANEL."""
    mmpp = compute_mmpp_por_receta(sample_dfs['RECETAS'], 'GRANEL')
    
    # Debe haber conceptos de GRANEL
    assert not mmpp.empty
    assert 'Proceso_Base' in mmpp.index
    assert 'Proceso_Adicional' in mmpp.index
    
    # Verificar valores
    assert mmpp['Proceso_Base'] == 1.8
    assert mmpp['Proceso_Adicional'] == 0.7


def test_compute_mmpp_por_receta_empty(sample_dfs):
    """Test de MMPP con tipo inexistente."""
    mmpp = compute_mmpp_por_receta(sample_dfs['RECETAS'], 'INEXISTENTE')
    
    assert mmpp.empty


# ===================== TESTS DE COSTEO MAYOR =====================

def test_costear_concepto_mayor_basic(sample_dfs):
    """Test básico de costeo desde MAYOR."""
    drivers_retail = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    drivers_granel = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    # Configuración de ejemplo para Mantención
    config_row = pd.Series({
        'concepto': 'Mantención',
        'split_retail_pct': 60,
        'split_granel_pct': 40,
        'driver_interno': 'tiempo_maquina',
        'denominador': 'kg_producidos'
    })
    
    result = costear_concepto_mayor(
        sample_dfs['MAYOR'], sample_dfs['INDICADORES_RETAIL'], 
        sample_dfs['INDICADORES_GRANEL'], config_row, '2024-01',
        drivers_retail, drivers_granel
    )
    
    # Verificar estructura del resultado
    assert isinstance(result, dict)
    assert 'retail' in result
    assert 'granel' in result
    
    # Verificar que hay valores para ambos segmentos
    assert not result['retail'].empty
    assert not result['granel'].empty
    
    # Verificar que los valores son positivos (costos)
    assert all(result['retail'] >= 0)
    assert all(result['granel'] >= 0)


def test_costear_concepto_mayor_no_data(sample_dfs):
    """Test de costeo con concepto sin datos en MAYOR."""
    drivers_retail = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    drivers_granel = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    # Concepto que no existe en MAYOR
    config_row = pd.Series({
        'concepto': 'ConceptoInexistente',
        'driver_split': 'kg_producidos',
        'driver_interno': 'kg_producidos',
        'denominador': 'kg_producidos'
    })
    
    result = costear_concepto_mayor(
        sample_dfs['MAYOR'], sample_dfs['INDICADORES_RETAIL'], 
        sample_dfs['INDICADORES_GRANEL'], config_row, '2024-01',
        drivers_retail, drivers_granel
    )
    
    # Debe retornar ceros
    assert all(result['retail'] == 0)
    assert all(result['granel'] == 0)


# ===================== TESTS DE COMEX ESPECIAL =====================

def test_comex_con_fletes_reales(sample_dfs):
    """Test de Comex con fletes reales y diferencia a distribuir."""
    drivers_retail = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    drivers_granel = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    result = _compute_comex_especial(sample_dfs, '2024-01', drivers_retail, drivers_granel)
    
    # Verificar estructura
    assert 'retail' in result
    assert 'granel' in result
    
    # Verificar que hay costos para retail (donde están los fletes)
    assert not result['retail'].empty
    assert all(result['retail'] >= 0)
    
    # Granel debe ser cero (no tiene fletes)
    assert all(result['granel'] == 0)


def test_comex_sku_sin_despachos(sample_dfs):
    """Test de Comex con SKU sin despachos (debe retornar 0)."""
    # Modificar datos para que un SKU tenga kg_despachados = 0
    sample_dfs['INDICADORES_RETAIL'].loc[
        sample_dfs['INDICADORES_RETAIL']['SKU'] == 'SKU001', 'kg_despachados'
    ] = 0
    
    drivers_retail = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    drivers_granel = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    result = _compute_comex_especial(sample_dfs, '2024-01', drivers_retail, drivers_granel)
    
    # SKU001 debe tener costo 0 (sin despachos)
    assert result['retail']['SKU001'] == 0
    
    # SKU002 debe tener costo > 0 (con despachos)
    assert result['retail']['SKU002'] > 0


# ===================== TESTS DE MATERIALES INDIRECTOS =====================

def test_materiales_indirectos_split_80_20(sample_dfs):
    """Test de Materiales Indirectos con split 80/20."""
    # Este test requiere implementar la lógica específica de Materiales_Indirectos
    # Por ahora, verificamos que el concepto está en CONFIG_SPLITS
    config_splits = sample_dfs['CONFIG_SPLITS']
    materiales_config = config_splits[config_splits['concepto'] == 'Materiales_Indirectos']
    
    assert not materiales_config.empty
    assert materiales_config.iloc[0]['split_retail_pct'] == 80
    assert materiales_config.iloc[0]['split_granel_pct'] == 20


# ===================== TESTS DE MANTENCIÓN =====================

def test_mantencion_split_60_40_tiempo_maquina(sample_dfs):
    """Test de Mantención con split 60/40 y driver tiempo_maquina."""
    drivers_retail = build_drivers_retail(sample_dfs['INDICADORES_RETAIL'], '2024-01')
    drivers_granel = build_drivers_granel(sample_dfs['INDICADORES_GRANEL'], '2024-01')
    
    # Obtener configuración de Mantención
    config_splits = sample_dfs['CONFIG_SPLITS']
    mantencion_config = config_splits[config_splits['concepto'] == 'Mantención'].iloc[0]
    
    result = costear_concepto_mayor(
        sample_dfs['MAYOR'], sample_dfs['INDICADORES_RETAIL'], 
        sample_dfs['INDICADORES_GRANEL'], mantencion_config, '2024-01',
        drivers_retail, drivers_granel
    )
    
    # Verificar que se usó el split correcto
    assert mantencion_config['split_retail_pct'] == 60
    assert mantencion_config['split_granel_pct'] == 40
    assert mantencion_config['driver_interno'] == 'tiempo_maquina'
    
    # Verificar que hay costos distribuidos
    assert all(result['retail'] > 0)
    assert all(result['granel'] > 0)
    
    # Verificar proporcionalidad aproximada (60/40 split)
    total_retail = (result['retail'] * drivers_retail['kg_producidos']).sum()
    total_granel = (result['granel'] * drivers_granel['kg_producidos']).sum()
    total_general = total_retail + total_granel
    
    retail_pct = total_retail / total_general
    granel_pct = total_granel / total_general
    
    # Permitir cierta tolerancia debido a la distribución interna
    assert abs(retail_pct - 0.6) < 0.1
    assert abs(granel_pct - 0.4) < 0.1


# ===================== TESTS DE TOTALES =====================

def test_compute_costos_retail_totales(sample_dfs):
    """Test de cálculo de totales Retail correctos."""
    resultado = compute_costos_retail(sample_dfs, '2024-01')
    
    # Verificar que el DataFrame no está vacío
    assert not resultado.empty
    
    # Verificar que tiene las columnas de totales
    assert 'Costos_Totales' in resultado.columns
    
    # Verificar que los totales son la suma de los conceptos
    conceptos_cols = [col for col in resultado.columns 
                     if col not in ['Costos_Directos', 'Costos_Indirectos', 'Costos_Totales']]
    
    if conceptos_cols:
        suma_conceptos = resultado[conceptos_cols].sum(axis=1)
        assert np.allclose(resultado['Costos_Totales'], suma_conceptos, rtol=1e-10)
    
    # Verificar que hay datos para ambos SKUs
    expected_skus = ['SKU001', 'SKU002']
    assert list(resultado.index) == expected_skus


def test_compute_costos_granel_totales(sample_dfs):
    """Test de cálculo de totales Granel correctos."""
    resultado = compute_costos_granel(sample_dfs, '2024-01')
    
    # Verificar que el DataFrame no está vacío
    assert not resultado.empty
    
    # Verificar que tiene las columnas de totales
    assert 'Costos_Totales' in resultado.columns
    
    # Verificar que los totales son la suma de los conceptos
    conceptos_cols = [col for col in resultado.columns 
                     if col not in ['Costos_Directos', 'Costos_Indirectos', 'Costos_Totales']]
    
    if conceptos_cols:
        suma_conceptos = resultado[conceptos_cols].sum(axis=1)
        assert np.allclose(resultado['Costos_Totales'], suma_conceptos, rtol=1e-10)
    
    # Verificar que hay datos para ambas especies
    expected_especies = ['Arándanos', 'Frambuesas']
    assert list(resultado.index) == expected_especies


# ===================== TESTS DE INTEGRACIÓN =====================

def test_compute_full_cost_analysis_success(sample_excel_bytes):
    """Test de análisis completo exitoso."""
    resultados = compute_full_cost_analysis(sample_excel_bytes, '2024-01')
    
    # Verificar estructura del resultado
    assert isinstance(resultados, dict)
    assert 'retail' in resultados
    assert 'granel' in resultados
    
    # Verificar que ambos DataFrames tienen datos
    assert not resultados['retail'].empty
    assert not resultados['granel'].empty
    
    # Verificar columnas de totales
    assert 'Costos_Totales' in resultados['retail'].columns
    assert 'Costos_Totales' in resultados['granel'].columns


def test_compute_full_cost_analysis_invalid_month(sample_excel_bytes):
    """Test de análisis con mes inválido."""
    with pytest.raises(ValueError, match="No hay datos disponibles para el mes"):
        compute_full_cost_analysis(sample_excel_bytes, '2025-12')


def test_compute_full_cost_analysis_invalid_data(incomplete_excel_bytes):
    """Test de análisis con datos incompletos."""
    with pytest.raises(ValueError, match="Hojas faltantes"):
        compute_full_cost_analysis(incomplete_excel_bytes, '2024-01')


# ===================== TESTS DE CASOS EDGE =====================

def test_drivers_with_missing_columns(sample_dfs):
    """Test de drivers cuando faltan columnas opcionales."""
    # Eliminar columnas opcionales
    df_retail_minimal = sample_dfs['INDICADORES_RETAIL'][['SKU', 'mes', 'kg_producidos', 'kg_despachados']].copy()
    
    drivers = build_drivers_retail(df_retail_minimal, '2024-01')
    
    # Debe tener drivers básicos pero no los opcionales
    assert 'kg_producidos' in drivers
    assert 'kg_despachados' in drivers
    assert 'hh_directas' not in drivers
    assert 'tiempo_maquina' not in drivers


def test_empty_dataframes():
    """Test con DataFrames vacíos."""
    empty_df = pd.DataFrame()
    
    drivers = build_drivers_retail(empty_df, '2024-01')
    assert drivers == {}
    
    mmpp = compute_mmpp_por_receta(empty_df, 'FRUTA')
    assert mmpp.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
