import subprocess

def run_commnad(cmd: str):
    print("\033[0;36m" + "[command] => {}\33[0m".format(cmd))
    p = subprocess.run(cmd, shell=True)
    return p

versions = []
versions.extend(["0.3.{}".format(i) for i in range(2, 18)])
#versions.extend(["0.4.2", "0.4.1"])

for version in versions:
    run_commnad("pip3 uninstall mohou -y")
    run_commnad("pip3 install mohou=={} -q".format(version))
    run_commnad("python3 test.py -post {}".format(version))
