"""Conexion a Azure SQL leyendo credenciales de .env.

Repo autocontenido: no depende del paquete `etl` del repo `deli`.
"""
import os
from pathlib import Path

import pyodbc

_ENV = Path(__file__).resolve().parent / ".env"
_CLAVES = ("SQL_SERVER", "SQL_DATABASE", "SQL_USERNAME", "SQL_PASSWORD")


def _cargar_env() -> None:
    """Parser minimo de .env (sin python-dotenv), igual criterio que deli."""
    if not _ENV.exists():
        return
    for linea in _ENV.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave, valor = clave.strip(), valor.strip()
        if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in "\"'":
            valor = valor[1:-1]
        os.environ.setdefault(clave, valor)


_cargar_env()


def _conn_str() -> str:
    faltan = [k for k in _CLAVES if not os.getenv(k)]
    if faltan:
        raise SystemExit(f"[conexion] faltan variables en .env: {', '.join(faltan)}")
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.environ['SQL_SERVER']};"
        f"DATABASE={os.environ['SQL_DATABASE']};"
        f"UID={os.environ['SQL_USERNAME']};"
        f"PWD={os.environ['SQL_PASSWORD']};"
        "TrustServerCertificate=yes;"
    )


def get_connection():
    """Conexion pyodbc nueva. Solo lectura: el proyecto nunca escribe en la base."""
    return pyodbc.connect(_conn_str())
