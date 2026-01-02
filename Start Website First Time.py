import subprocess
import os
import time

def run(cmd):
    return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)

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

print("Huey worker...")
run(["python", "manage.py", "run_huey"])

print("Running MSSA Training (Run_All.py)...")
run(["python", "API/Utils/Run_All.py"])

print("Tailwind...")
run(["python", "manage.py", "tailwind", "dev"])