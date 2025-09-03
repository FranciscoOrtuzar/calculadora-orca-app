#!/bin/bash

# Setup script para Calculadora VF App
# Este script se ejecuta automáticamente en el despliegue

echo "🚀 Configurando Calculadora VF App..."

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios necesarios si no existen
mkdir -p outputs
mkdir -p data

echo "✅ Configuración completada!"
