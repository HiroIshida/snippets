from mimic.trainer import TrainCache
from mimic.models import ImageAutoEncoder
from mimic.datatype import ImageCommandDataChunk
from mimic.scripts.train_propagator import prepare_trained_image_chunk
import torch
import torchvision
import matplotlib.pyplot as plt

project_name = "dish_demo"
try:
    print(chunk)
except:
    chunk: ImageCommandDataChunk = prepare_trained_image_chunk(project_name)
    tcache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    n_intact = 5
    chunk_intact, _ = chunk.split(n_intact)
    imgseq, cmdseq = chunk_intact[0]

tf = torchvision.transforms.ToTensor()
img_list = [tf(img).float() for img in imgseq.data]
data_torch = torch.stack(img_list)

model = tcache.best_model
out = model(data_torch)

for i in range(100):
    plt.imshow(torchvision.transforms.ToPILImage()(out[i]), interpolation="bicubic")
    plt.show()
