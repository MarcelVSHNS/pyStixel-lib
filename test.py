import numpy as np
import open3d as o3d
from stixel import StixelWorld, CameraInfo
from stixel.utils import draw_stixels_on_image
from PIL import Image
import yaml


def main():
    # install pyyaml
    with open('sample/waymo_calib.yaml') as yaml_file:
        calib = yaml.load(yaml_file, Loader=yaml.FullLoader)
    cam_info = CameraInfo(cam_mtx_k=calib['K'])
    stixel_world: StixelWorld = StixelWorld.read("sample/10084636266401282188_1120_000_1140_000_0.csv", camera_info=cam_info)

    """ packing test """
    stixel_world.save(binary=True)
    stixel_world = StixelWorld.read("10084636266401282188_1120_000_1140_000_0.stx1")

    """ visual 2d test """
    stixel_img = draw_stixels_on_image(stixel_world.image, stixel_world.stixel)
    stixel_img.show()
    print(stixel_world.camera_info.K)
    print(stixel_world.camera_info.img_size)

    """ 3d test """
    # install open3d
    stxl_wrld_pts, colors = stixel_world.get_pseudo_coordinates()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_wrld_pts)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualise point cloud
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
