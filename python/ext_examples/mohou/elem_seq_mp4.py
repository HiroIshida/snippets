import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from mohou.types import EpisodeBundle
from mohou.file import get_project_path
from mohou.types import RGBImage

pp = get_project_path("bunsetsu_test_project")
bundle = EpisodeBundle.load(pp)
episode = bundle[0]
rgb_seq = episode.get_sequence_by_type(RGBImage)
seq = [e.numpy() for e in rgb_seq.elem_list]

fps = 20
clip = ImageSequenceClip(seq, fps=fps)

use_mp4 = True
if use_mp4:
    clip.write_videofile("hoge.mp4", audio=False)
    clip_again = VideoFileClip("hoge.mp4")
else:
    clip.write_gif("hoge.gif")
    clip_again = VideoFileClip("hoge.gif")

rgb_seq_again = [RGBImage(f) for f in clip_again.iter_frames()]

for a, b in zip(rgb_seq, rgb_seq_again):
    size = np.prod(a.shape[:2])
    diff = np.linalg.norm(a.numpy() - b.numpy()) / size
    print(diff)
