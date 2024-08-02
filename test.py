import numpy as np
import open3d as o3d
from stixel import StixelWorld
from stixel.utils import draw_stixels_on_image
from PIL import Image
import yaml


def main():
    stixel_world: StixelWorld = StixelWorld.read("sample/10084636266401282188_1120_000_1140_000_0.csv")
    stixel_world.image = Image.open("sample/10084636266401282188_1120_000_1140_000_0.png")
    with open('sample/waymo_calib.yaml') as yaml_file:
        calib = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # stixel_img = draw_stixels_on_image(stixel_world.image, stixel_world.stixel)
    # stixel_img.show()
    stixel_world.camera_mtx = np.array(calib['K'])
    stxl_wrld_pts, colors = stixel_world.get_pseudo_coordinates()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_wrld_pts)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualise point cloud
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
