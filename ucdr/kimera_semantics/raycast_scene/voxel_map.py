from google.protobuf.internal.decoder import _DecodeVarint32
import numpy as np

from ucdr.kimera_semantics.raycast_scene.proto.semantic_map_pb2 import SemanticMapProto
from ucdr.kimera_semantics.raycast_scene import get_grid_index_from_point, get_a_b_c_from_linear


__all__ = ["VoxelMap"]

EPS = 10 ** (-4)


def parse_protobug_msg_into_accessiable_np_array(vox_map):
    """
    Assumes constant voxel size and grid.
    Will allocate the full memory in a cube
    :param origins:
    :param vox_map:
    :return:
    """
    voxels_per_side = vox_map.semantic_blocks[0].voxels_per_side
    voxel_size = vox_map.semantic_blocks[0].voxel_size

    origins = np.zeros((len(vox_map.semantic_blocks), 3))
    for i in range(len(vox_map.semantic_blocks)):
        origins[i, 0] = vox_map.semantic_blocks[i].origin.x
        origins[i, 1] = vox_map.semantic_blocks[i].origin.y
        origins[i, 2] = vox_map.semantic_blocks[i].origin.z

    # Real mi and ma value of voxel_grid
    mi = np.min(origins, axis=0)
    ma = np.max(origins, axis=0) + (voxel_size * voxels_per_side)

    large_grid = voxel_size * voxels_per_side
    large_grid_inv = 1 / large_grid
    elements = np.floor(((ma - mi) + EPS) / large_grid).astype(np.uint32)

    # Store all voxels in a grid volume
    voxels = np.zeros((*tuple((elements) * voxels_per_side), 41))

    for j, block in enumerate(vox_map.semantic_blocks):
        block_idx = get_grid_index_from_point(origins[j] - mi, large_grid_inv)
        block_idx = block_idx * voxels_per_side
        for sem_voxel in block.semantic_voxels:
            abc = get_a_b_c_from_linear(sem_voxel.linear_index, voxels_per_side)
            voxel_idx = block_idx + abc
            voxels[tuple(voxel_idx)] = sem_voxel.semantic_labels
    # dont return the orgin of the first block which is center
    # return orging point of new voxel_grid
    return voxels, mi


def parse(file_handle, msg):
    buf = file_handle.read()
    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n : n + msg_len]
        n += msg_len
        read_metric = msg
        read_metric.ParseFromString(msg_buf)
    return msg


def get_semantic_map(p):
    msg = SemanticMapProto()
    with open(p, "rb") as f:
        msg = parse(f, msg)
    return msg


from scipy.interpolate import griddata


class VoxelMap:
    def __init__(self, map_serialized_path, size, r_sub):
        H, W = size
        vox_map = get_semantic_map(map_serialized_path)
        self._voxels, self._mi = parse_protobug_msg_into_accessiable_np_array(vox_map)
        self._voxel_size = vox_map.semantic_blocks[0].voxel_size

        self._probs = np.zeros((H, W, self._voxels.shape[3]))
        v, u = np.mgrid[0:H, 0:W]
        self._vo = np.copy(v)
        self._uo = np.copy(u)
        self._vr = v[::r_sub, ::r_sub].reshape(-1)
        self._ur = u[::r_sub, ::r_sub].reshape(-1)
        self._v = v.reshape(-1)
        self._u = u.reshape(-1)
        self._r_sub = r_sub

        points = np.zeros_like(self._vo)
        points[::r_sub, ::r_sub] = 1
        self.m = points == 1
        self.points = np.stack(np.where(points), axis=1)

    def ray_cast_results_to_probs(self, locations, index_ray):
        self._probs = np.zeros(self._probs.shape)

        if locations.shape[0] == 0:
            return self._probs

        idx_tmp = np.floor(((locations - self._mi + EPS) / self._voxel_size)).astype(np.uint32)

        for j in range(locations.shape[0]):
            _v, _u = self._vr[index_ray[j]], self._ur[index_ray[j]]
            self._probs[_v, _u, :] = self._voxels[tuple(idx_tmp[j])]

        self._probs = self._probs - (np.min(self._probs, axis=2)[..., None]).repeat(self._probs.shape[2], 2)

        m = (np.sum(self._probs, axis=2)[..., None]).repeat(self._probs.shape[2], 2) > EPS
        self._probs[m] = self._probs[m] / (np.sum(self._probs, axis=2)[..., None]).repeat(self._probs.shape[2], 2)[m]
        inv_m = m == False
        self._probs[inv_m] = 0
        inv_m[:, :, 1:] = False
        self._probs[inv_m] = 1
        self._probs = griddata(self.points, self._probs[self.m], (self._vo, self._uo), method="nearest")
        return self._probs
