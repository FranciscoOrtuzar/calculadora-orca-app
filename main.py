# main.py - Punto de entrada para el despliegue web
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # Configuración para despliegue web
    port = "8501"
    sys.argv = [
        "streamlit", "run", "Histórico de Datos.py",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        f"--server.port={port}",
    ]
    sys.exit(stcli.main())