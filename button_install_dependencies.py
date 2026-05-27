import subprocess
import sys
import os

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

print("=" * 60)
print("Installing Imitation Lab dependencies")
print(f"Python:       {sys.executable}")
print(f"Requirements: {requirements_path}")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", requirements_path],
)

if result.returncode == 0:
    print("\nAll dependencies installed successfully.")
else:
    print(f"\nInstallation failed with return code {result.returncode}.")
    sys.exit(result.returncode)
