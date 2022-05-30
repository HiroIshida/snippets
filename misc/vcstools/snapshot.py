from dataclasses import dataclass
import copy
from typing import List
import os
import subprocess
import yaml

base_dir = '/home/h-ishida/pilot-auto'
filename = os.path.join(base_dir, 'autoware.repos')
filename_out = os.path.join(base_dir, 'autoware.repos-snapshot')

with open(filename, 'r') as f:
    data = yaml.safe_load(f)

excludes = ['autoware/universe', 'autoware/launcher']
removes = [
    'simulator/scenario_simulator',
    'simulator/tier4_autoware_msgs',
    'simulator/logsim',
    'simulator/awml_evaluation',
    'simulator/ndt_convergence_evaluation']

repos = data['repositories']

for key in removes:
    repos.pop(key)

for repo_name, repo in repos.items():
    if repo_name in excludes:
        continue

    assert repo['type'] == 'git'
    repo_localition = os.path.join(base_dir, 'src', repo_name)
    cmd = ['git', '-C', repo_localition, 'rev-parse', '--verify', 'HEAD']
    print('processing {}'.format(repo_name))
    out = subprocess.run(cmd, capture_output=True)
    repo['version'] = out.stdout.decode().strip()

with open(filename_out, 'w') as f:
    yaml.dump(data, f)
