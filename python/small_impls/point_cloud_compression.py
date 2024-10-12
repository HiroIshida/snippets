import time
import struct
import pickle
from dataclasses import dataclass
import numpy as np
from skrobot.model.primitives import Box, Sphere, PointCloudLink
from skrobot.coordinates import Transform
from typing import Tuple, Union, Callable, Optional, ClassVar
import zlib
import lzma
import brotli
import lz4.frame



class SerializableTransform(Transform):
    n_bytes: ClassVar[int] = 96  # 8 * (3 + 9)

    def serialize(self) -> bytes:
        trans_bytes = self.translation.tobytes()
        rot_bytes = self.rotation.tobytes()
        return trans_bytes + rot_bytes

    @classmethod
    def deserialize(cls, serialized: bytes) -> "SerializableTransform":
        assert len(serialized) == cls.n_bytes
        translation = np.frombuffer(serialized[:24], dtype=np.float64)
        rotation = np.frombuffer(serialized[24:], dtype=np.float64)
        return cls(translation, rotation)


@dataclass
class VoxelGridSkelton:
    tf_local_to_world: Transform
    extents: Tuple[float, float, float]
    resols: Tuple[int, int, int]
    n_bytes_decomp: ClassVar[int] = 120  # 96 for tf, 12 for extents, 12 for resols

    @property
    def intervals(self) -> np.ndarray:
        return np.array(self.extents) / np.array(self.resols)

    @classmethod
    def from_box(cls, box: Box, resols: Tuple[int, int, int]) -> 'VoxelGridSkelton':
        tf_local_to_world = SerializableTransform(box.worldpos(), box.worldrot())
        return cls(tf_local_to_world, box.extents, resols)

    def get_eval_points(self) -> np.ndarray:
        half_ext = np.array(self.extents) * 0.5
        lins = [np.linspace(-half_ext[i], half_ext[i], self.resols[i]) for i in range(3)]
        X, Y, Z = np.meshgrid(*lins)
        points_wrt_local = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        points_wrt_world = self.tf_local_to_world.transform_vector(points_wrt_local)
        return points_wrt_world

    def points_to_indices(self, points_wrt_world: np.ndarray) -> np.ndarray:
        tf_world_to_local = self.tf_local_to_world.inverse_transformation()
        points_wrt_local = tf_world_to_local.transform_vector(points_wrt_world)
        lb = np.array(self.extents) * -0.5
        indices = np.floor((points_wrt_local - lb) / self.intervals).astype(int)
        indices_flat = (indices[:, 0] + indices[:, 1] * self.resols[0] + indices[:, 2] * self.resols[0] * self.resols[1]).astype(int)
        indices_flat.sort()
        return indices_flat.astype(np.uint32)

    def indices_to_points(self, indices_flat: np.ndarray) -> np.ndarray:
        indices_z = indices_flat // (self.resols[0] * self.resols[1])
        indices_y = (indices_flat % (self.resols[0] * self.resols[1])) // self.resols[0]
        indices_x = indices_flat % self.resols[0]
        indices = np.stack([indices_x, indices_y, indices_z], axis=-1)

        lb = np.array(self.extents) * -0.5
        points_wrt_local = np.array(indices) * self.intervals + lb
        points_wrt_world = self.tf_local_to_world.transform_vector(points_wrt_local)
        return points_wrt_world

    def serialize(self) -> bytes:
        extents_bytes = struct.pack('fff', *self.extents)
        resols_bytes = struct.pack('iii', *self.resols)
        tf_bytes = self.tf_local_to_world.serialize()
        serialized = extents_bytes + resols_bytes + tf_bytes
        return zlib.compress(serialized, 0)

    @classmethod
    def deserialize(cls, serialized: bytes) -> 'VoxelGridSkelton':
        unziped = zlib.decompress(serialized)
        assert len(unziped) == cls.n_bytes_decomp
        extents = struct.unpack('fff', unziped[:12])
        resols = struct.unpack('iii', unziped[12:24])
        tf_local_to_world = SerializableTransform.deserialize(unziped[24:])
        return cls(tf_local_to_world, extents, resols)


class ZlibCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return zlib.compress(data, 1)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return zlib.decompress(data)


class LzmaCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return lzma.compress(data, preset=1)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return lzma.decompress(data)


class BrotliCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return brotli.compress(data, quality=2)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return brotli.decompress(data)


class LZ4Compressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return lz4.frame.compress(data)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class CombinedCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return zlib.compress(lzma.compress(data, preset=1), 1)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return lzma.decompress(zlib.decompress(data))



@dataclass
class VoxelGrid(LzmaCompressor):
    skelton: VoxelGridSkelton
    indices: np.ndarray

    @classmethod
    def from_points(cls,
                    points_world: np.ndarray,
                    skelton: VoxelGridSkelton,
                    np_type: Union[None, np.uint8, np.uint16, np.uint32, np.uint64] = None,
                    ) -> 'VoxelGrid':
        tf_world_to_local = skelton.tf_local_to_world.inverse_transformation()
        points_local = tf_world_to_local.transform_vector(points_world)
        indices = skelton.points_to_indices(points_local)
        return cls(skelton, indices)

    @classmethod
    def from_sdf(cls, sdf: Callable[[np.ndarray], np.ndarray], skelton: VoxelGridSkelton) -> 'VoxelGrid':
        points = skelton.get_eval_points()
        sdf_values = sdf(points)
        width = np.max(skelton.intervals)
        surface_indices = np.logical_and(sdf_values < 0, sdf_values > - width)
        return cls.from_points(points[surface_indices], skelton)

    def serialize(self) -> bytes:
        skelton_bytes = self.skelton.serialize()
        indices_bytes = self.indices.tobytes()
        indices_comp_bytes = self.compress(indices_bytes)
        skelton_bytes_size_bytes = struct.pack('I', len(skelton_bytes))
        return skelton_bytes_size_bytes + skelton_bytes + indices_comp_bytes

    @classmethod
    def deserialize(cls, serialized: bytes) -> 'VoxelGrid':
        skelton_bytes_size = struct.unpack('I', serialized[:4])[0]
        bytes_skelton = serialized[4:4 + skelton_bytes_size]
        skelton = VoxelGridSkelton.deserialize(bytes_skelton)
        bytes_other = serialized[4 + skelton_bytes_size:]
        unziped = cls.decompress(bytes_other)
        indices = np.frombuffer(unziped, dtype=int)
        return cls(skelton, indices)

    def to_points(self) -> np.ndarray:
        points_wrt_world = self.skelton.indices_to_points(self.indices)
        return points_wrt_world


if __name__ == "__main__":
    from skrobot.viewers import TrimeshSceneViewer
    import subprocess
    import sys
    region = Box([1, 1, 0.5])
    sphere = Sphere(0.2, with_sdf=True)
    skelton = VoxelGridSkelton.from_box(region, (56, 56, 28))
    voxel_grid = VoxelGrid.from_sdf(sphere.sdf, skelton)

    points = skelton.get_eval_points()
    indices = skelton.points_to_indices(points)
    points_again = skelton.indices_to_points(indices)
    
    # voxel_grid_bytes_round_trip = VoxelGrid.deserialize(voxel_grid.serialize()).serialize()
    # assert voxel_grid.serialize() == voxel_grid_bytes_round_trip

    ts = time.time()
    for _ in range(1000):
        serialized = voxel_grid.serialize()
    print(f"time elapsed average: {(time.time() - ts) / 1000} sec")
    print(f"Serialized size: {len(serialized)} bytes")

    # points = voxel_grid.to_points()
    # plink = PointCloudLink(points)

    # viewer = TrimeshSceneViewer()
    # viewer.add(plink)
    # viewer.show()
    # import time; time.sleep(1000)
