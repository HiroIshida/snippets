import subprocess
from pathlib import Path
import pkg_resources

def get_git_commit_hash(path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).strip().decode()
    except Exception as e:
        return f"Error: {e}"

def get_git_diff(path):
    try:
        diff_output = subprocess.check_output(["git", "diff"], cwd=path)
        return diff_output.decode()
    except subprocess.CalledProcessError as e:
        return f"Git command failed: {e}"
    except Exception as e:
        return f"Error: {e}"

installed_packages = [(d.project_name, d.version, d.location) for d in pkg_resources.working_set]
installed_packages.sort()

off_the_shelf_packages = []
manual_packages = []
for pkg in installed_packages:
    name, version, location = pkg
    if Path(location).name in ("site-packages", "dist-packages"):
        off_the_shelf_packages.append(pkg)
        print(pkg)
    else:
        manual_packages.append(pkg)

print("Off the shelf packages:")
for pkg in off_the_shelf_packages:
    print(pkg)
print("Manualy installed packages:")
for pkg in manual_packages:
    print(pkg)

print("detail of off the shelf packages:")
for pkg in manual_packages:
    commit_hash = get_git_commit_hash(pkg[2])
    print(f"{pkg[0]}=={pkg[1]} ({commit_hash})")

    diff = get_git_diff(pkg[2])
    print(diff)

