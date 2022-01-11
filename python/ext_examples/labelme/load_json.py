import os
import json
import labelme
from labelme.utils import img_b64_to_arr
from labelme.utils import shapes_to_label
import matplotlib.pyplot as plt

json_file = os.path.expanduser('~/.kyozi/scene/scene-20220112034112.json')
data = json.load(open(json_file))
imageData = data.get("imageData")
img = img_b64_to_arr(imageData)

label_name_to_value = {"_background_": 0}
for shape in sorted(data["shapes"], key=lambda x: x["label"]):
    label_name = shape["label"]
    if label_name in label_name_to_value:
        label_value = label_name_to_value[label_name]
    else:
        label_value = len(label_name_to_value)
        label_name_to_value[label_name] = label_value
lbl, _ = shapes_to_label(
    img.shape, data["shapes"], label_name_to_value)

# labelme then convert lbl to image. But I need only lbl not image

plt.imshow(lbl)
plt.show()

