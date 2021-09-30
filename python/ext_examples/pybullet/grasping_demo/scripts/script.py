import os.path as osp
import gdown

download_dir = osp.expanduser('pr2_description')
#download_dir = osp.expanduser('tmp')
gdown.cached_download(
    url='https://drive.google.com/uc?id=1OXyxBEqamCg7cVnLmLj8WvPWRKFEldsC',
    path=osp.join(download_dir, 'meshes.zip'),
    md5='1d504ebcae17d79c5cc05593c37eb447',
    postprocess=gdown.extractall,
    quiet=True,
)
