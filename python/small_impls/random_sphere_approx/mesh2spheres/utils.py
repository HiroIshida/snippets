import logging
import pickle
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pysdfgen
from scipy.interpolate import RegularGridInterpolator
from trimesh import Trimesh

logger = logging.getLogger(__name__)


class GridSDF:
    def __init__(
        self, mesh: Trimesh, n_grid: int = 100, n_padding: int = 5, fill_value: float = np.inf
    ):

        hash_val = md5(pickle.dumps(mesh)).hexdigest()
        cache_dir = Path("/tmp/mesh2spheres_sdf_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{hash_val}-{n_grid}-{n_padding}.sdf"
        if not cache_file.exists():
            logger.info("sdf cache not found, generating sdf. This may take a while.")
            # save mesh to tmp file as currently pysdfgen required file path rather than mesh object
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                tmpfile = tmpdir / "tmp.obj"
                mesh.export(tmpfile)
                pysdfgen.mesh2sdf(tmpfile, n_grid, n_padding, cache_file)
                logger.info(f"sdf cache is saved to {cache_file}")

        sdf_data = None
        with open(cache_file, "r") as f:
            nx, ny, nz = [int(i) for i in f.readline().split()]
            ox, oy, oz = [float(i) for i in f.readline().split()]
            dims = np.array([nx, ny, nz])
            origin = np.array([ox, oy, oz])

            resolution = float(f.readline())
            sdf_data = (
                np.fromstring(f.read(), dtype=float, sep="\n").reshape(*dims).transpose(2, 1, 0)
            )
        assert sdf_data is not None

        xlin, ylin, zlin = [np.array(range(d)) * resolution for d in sdf_data.shape]

        self.itp = RegularGridInterpolator(
            (xlin, ylin, zlin), sdf_data, bounds_error=False, fill_value=fill_value
        )
        self.origin = origin

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.itp(points - self.origin[None, :])
