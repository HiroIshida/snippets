import numpy as np
import skrobot
from skrobot.model import MeshLink

m = MeshLink(visual_mesh=skrobot.data.bunny_objpath(), with_sdf=True)
m.translate([0, 0.1, 0])
pts = np.random.randn(100, 3)

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
for i in range(100):
    m.sdf(pts)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
