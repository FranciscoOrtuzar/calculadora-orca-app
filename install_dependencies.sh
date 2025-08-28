#!/bin/bash

# =============================================================================
# SCRIPT DE INSTALACIÓN - CALCULADORA VF
# =============================================================================
# Este script instala todas las dependencias necesarias para el proyecto
# =============================================================================

echo "🚀 Iniciando instalación de dependencias para Calculadora VF..."
echo ""

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    echo "   Por favor, instala Python 3.9+ desde https://python.org"
    exit 1
fi

# Verificar versión de Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detectado"

# Verificar si pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 no está instalado"
    echo "   Por favor, instala pip3 o actualiza Python"
    exit 1
fi

echo "✅ pip3 detectado"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "🔧 Creando entorno virtual..."
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate
echo "✅ Entorno virtual activado"

# Actualizar pip
echo "🔧 Actualizando pip..."
pip install --upgrade pip
echo "✅ pip actualizado"

# Instalar dependencias de desarrollo
echo "🔧 Instalando dependencias de desarrollo..."
pip install -r requirements-dev.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencias instaladas correctamente"
else
    echo "❌ Error al instalar dependencias"
    echo "   Intentando con requirements.txt..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Dependencias básicas instaladas"
    else
        echo "❌ Error crítico en la instalación"
        exit 1
    fi
fi

echo ""
echo "🎉 ¡Instalación completada!"
echo ""
echo "📋 Para activar el entorno virtual en el futuro:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 Para ejecutar la aplicación:"
echo "   streamlit run Inicio.py"
echo ""
echo "📚 Para ver las dependencias instaladas:"
echo "   pip list"
echo ""
echo "🔧 Para desactivar el entorno virtual:"
echo "   deactivate"
