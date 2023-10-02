import numpy as np
from skrobot.coordinates import Coordinates

co_obj = Coordinates([1.0, 1.0, 0.0], rot=[np.pi * 0.25, 0.0, 0.0])
co_hand = Coordinates([0.0, 0.0, 0.0])

co_hand.transform(co_obj.inverse_transformation())
print(co_hand)


