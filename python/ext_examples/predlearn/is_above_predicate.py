import time
import os
import pybullet
from predlearn.files import get_dataset_path
from predlearn.predicates import IsAbove
from predlearn.utils import pb_slide_point, temporary_slide

client = pybullet.connect(pybullet.GUI)
ycb_path = get_dataset_path('ycb') 
urdf1 = os.path.join(ycb_path, "002_master_chef_can.urdf")
urdf2 = os.path.join(ycb_path, "003_cracker_box.urdf")
id1 = pybullet.loadURDF(urdf1)
id2 = pybullet.loadURDF(urdf2)

with temporary_slide(id2, [0.0, 0, 0.3]):
    with temporary_slide(id1, [0.08, 0, 1.0]):
        ts = time.time()
        pred = IsAbove.from_ids(id1, id2) 
        print(pred.value)
        print(time.time() - ts)
        time.sleep(10)
