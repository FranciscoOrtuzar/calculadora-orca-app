# main.py
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    port = "8501"  # cámbialo si choca con otro
    sys.argv = [
        "streamlit", "run", "app.py",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        f"--server.port={port}",
    ]
    sys.exit(stcli.main())