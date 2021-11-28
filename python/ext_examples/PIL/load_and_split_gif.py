import os
from PIL import Image
import matplotlib.pyplot as plt
im = Image.open(os.path.expanduser('~/.kyozi/replay_cache/fed_images-20211128003304.gif'))
im.seek(0) # move to frame I want to access
frame = im.copy()
frame.save('sample.png', 'PNG')
