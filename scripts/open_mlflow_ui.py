from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "notebooks" / "mlflow.db"

if not MLFLOW_DB.exists():
    raise FileNotFoundError(
        f"Não encontrei o banco do MLflow em: {MLFLOW_DB}\n"
        "Rode primeiro os notebooks que registram os experimentos."
    )

backend_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

cmd = [
    sys.executable,
    "-m",
    "mlflow",
    "ui",
    "--backend-store-uri",
    backend_uri,
]

print("Abrindo MLflow UI...")
print(f"Backend: {backend_uri}")
print("Acesse: http://127.0.0.1:5000")

subprocess.run(cmd, check=True)
