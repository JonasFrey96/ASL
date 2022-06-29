import open3d as o3d

__all__ = ["draw_cube", "Visualizer3D"]


def draw_cube(vis, translation, size, color):
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.scale(size, center=cube.get_center())
    cube.translate(translation, relative=False)
    cube.paint_uniform_color(color / 255)
    vis.add_geometry(cube)


class Visualizer3D:
    def __init__(self, size, k_image, mesh_path):
        self._size = size
        self._k_image = k_image

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self._size[1], height=self._size[0], visible=True)

        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        self.vis.add_geometry(mesh_o3d)

    def __del__(self):
        self.vis.destroy_window()

    def visu(self, locations, ray_origins, sub=1000, sub2=8):
        if locations is not None:
            for j in range(0, locations.shape[0], sub):
                # Draw detected mesh intersection points
                sphere_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(locations[j, :])
                self.vis.add_geometry(sphere_o3d)

        if ray_origins is not None:
            # Draw camera rays start and end
            for j in range(0, ray_origins.shape[0], sub):
                sphere_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(ray_origins[j, :])
                sphere_o3d.paint_uniform_color([1, 0, 0])
                self.vis.add_geometry(sphere_o3d)

        # if False:
        #   for block in range(len(map.semantic_blocks)):
        #     large_size = (map.semantic_blocks[block].voxel_size * map.semantic_blocks[block].voxels_per_side)  # in meters
        #     cube = o3d.geometry.TriangleMesh.create_box()
        #     cube.scale(1 * large_size / 5, center=cube.get_center())
        #     cube.translate(origins[block])
        #     vis.add_geometry(cube)
        #     print(large_size, origins[block], map.semantic_blocks[block].voxel_size)

        # if False:
        #   for block in range(4):  # len(map.semantic_blocks)):
        #     voxel_size = map.semantic_blocks[block].voxel_size  # in meters
        #     voxels_per_side = map.semantic_blocks[block].voxels_per_side

        #     for j in range(0, len(map.semantic_blocks[block].semantic_voxels), 1):
        #       cube = o3d.geometry.TriangleMesh.create_box()
        #       cube.scale(1 * voxel_size * 2, center=cube.get_center())
        #       index = map.semantic_blocks[block].semantic_voxels[j].linear_index
        #       trans = get_x_y_z(index, origins[block], voxels_per_side, voxel_size)
        #       cube.translate(trans)
        #       rgb = [
        #         map.semantic_blocks[block].semantic_voxels[j].color.r / 255,
        #         map.semantic_blocks[block].semantic_voxels[j].color.g / 255,
        #         map.semantic_blocks[block].semantic_voxels[j].color.b / 255]

        #       rgb = [0, 0, 1]
        #       cube.paint_uniform_color(rgb)
        #       vis.add_geometry(cube)
        # if True:
        #   for j in range(0, self._voxels.shape[0], sub2):
        #     for k in range(0, self._voxels.shape[1], sub2):
        #       for l in range(0, self._voxels.shape[2], sub2):
        #         translation = np.array([j, k, l], dtype=np.float)
        #         translation *= self._voxel_size
        #         translation += self._mi

        #         # check if voxel is valid
        #         if np.sum(self._voxels[j, k, l, :]) != 0:
        #           col_index = np.argmax(self._voxels[j, k, l, :])
        #           draw_cube(vis, translation, self._voxel_size, self._rgb[col_index])

        # if locations is not None:
        #   idx_tmp = np.floor(((locations - self._mi + eps) / self._voxel_size)).astype(np.uint32)

        #   for j in range(0, locations.shape[0], sub):
        #     col_index = np.argmax(self._voxels[idx_tmp[j, 0],
        #                           idx_tmp[j, 1],
        #                           idx_tmp[j, 2], :])

        #     translation = np.copy(idx_tmp[j]).astype(np.float)
        #     translation *= self._voxel_size
        #     translation += self._mi
        #     draw_cube(vis, translation, self._voxel_size * 2, self._rgb[col_index])

        self.vis.run()
