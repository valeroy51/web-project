import subprocess, time, os

def run(cmd):
    return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)

print("Tailwind")
run(["python", "manage.py", "runserver"])

print("Huey worker")
run(["python", "manage.py", "run_huey"])