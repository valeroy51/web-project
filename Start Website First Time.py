import subprocess
import os
import socket
import time

def wait_for_db(host: str, port: int, timeout: int = 60):
    print(f"Waiting for DB at {host}:{port} ...")
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("DB is reachable.")
                return
        except OSError:
            if time.time() - start > timeout:
                raise TimeoutError("DB not reachable after timeout")
            time.sleep(1)

DB_HOST = os.getenv("DB_HOST", "database")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

wait_for_db(DB_HOST, DB_PORT)

print("Running migrate...")
subprocess.run(["python", "manage.py", "migrate"], check=True)

print("Creating superuser if not exists...")

env = os.environ.copy()
env["DJANGO_SUPERUSER_USERNAME"] = "Admin"
env["DJANGO_SUPERUSER_EMAIL"] = "admin@gmail.com"
env["DJANGO_SUPERUSER_PASSWORD"] = "Admin1234"

subprocess.run(
    ["python", "manage.py", "createsuperuser", "--noinput"],
    env=env,
    check=False
)

print("Collecting static files...")
subprocess.run(["python", "manage.py", "collectstatic", "--noinput"], check=True)

print("Running MSSA Training (Run_All.py)...")
subprocess.run(["python", "API/Utils/Run_All.py"])

print("Run Web...")
subprocess.run(["python", "manage.py", "runserver", "0.0.0.0:8000"])