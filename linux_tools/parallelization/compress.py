import subprocess
import path

#cmd = "xargs -n 1 -P 12 pigz --keep --force"
cmd = "xargs -n 1 -P 12 unpigz --keep --force"

p = path.Path("./tmp")
#ps = [str(pp) for pp in p.listdir() if pp.name.endswith(".pkl")]
ps = [str(pp) for pp in p.listdir() if pp.name.endswith(".pkl.gz")]
string = "\n".join(ps)
subprocess.run(cmd, shell=True, input=string, text=True)
