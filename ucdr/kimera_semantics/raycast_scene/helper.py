import numpy as np

__all__ = [
    "get_rays",
    "transform_points",
    "get_a_b_c_from_linear",
    "get_linear_from_a_b_c",
    "get_x_y_z",
    "get_grid_index_from_point",
    "get_semantic_voxel_from_point",
    "get_voxels_grid_idx_from_point",
]

EPS = 10 ** (-4)

# GEOMETRIC HELPER FUNCTIONS


def get_rays(k, size, extrinsic, d_min=0.1, d_max=5):
    h, w = size
    v, u = np.mgrid[0:h, 0:w]
    n = np.ones_like(u)
    image_coordinates = np.stack([u, v, n], axis=2).reshape((-1, 3))
    k_inv = np.linalg.inv(k)
    points = (k_inv @ image_coordinates.T).T

    start = np.copy(points) * d_min
    stop = np.copy(points) * d_max
    dir = stop - start

    start = start.reshape((h, w, 3))
    stop = stop.reshape((h, w, 3))
    dir = dir.reshape((h, w, 3))
    return start, stop, dir


def transform_points(points, H):
    return points @ H[:3, :3].T + H[:3, 3]


# VOXEL HELPER FUNCTIONS


def get_a_b_c_from_linear(index, voxels_per_side):
    a = index % (voxels_per_side)
    b = ((index - a) / (voxels_per_side)) % (voxels_per_side)
    c = ((index - a) - (voxels_per_side * b)) / (voxels_per_side ** 2)
    return (int(a), int(b), int(c))


def get_linear_from_a_b_c(a, b, c, voxels_per_side):
    return a + (b * voxels_per_side) + (c * voxels_per_side * voxels_per_side)


def get_x_y_z(index, origin, voxels_per_side, voxel_size):
    abc = get_a_b_c_from_linear(index, voxels_per_side)
    print("abc", abc)
    add = np.full((3), voxel_size) * (abc - np.array([16, 16, 16]))
    return origin + add


def get_grid_index_from_point(point, grid_size_inv):
    return np.floor(point * grid_size_inv + EPS).astype(np.uint32)


def get_semantic_voxel_from_point(point, voxel_size, voxels_per_side):
    grid_size = voxel_size * voxels_per_side
    grid_size_inv = 1 / grid_size
    block_coordinate = get_grid_index_from_point(point, grid_size_inv)
    point_local = point - block_coordinate * np.fill(grid_size, 3)
    local_coordinate = get_grid_index_from_point(point_local, 1 / voxel_size)
    return block_coordinate, local_coordinate


def get_voxels_grid_idx_from_point(point, mi, voxel_size):
    idx = np.floor((point - mi) / (voxel_size))
    return idx.astype(np.uint32)
