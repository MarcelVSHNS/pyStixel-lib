import stixel as stx


if __name__ == "__main__":
    """ Convert 
    stixel_world: stx.StixelWorld = stx.read_csv("sample/17065833287841703_2980_000_3000_000_165_FRONT.csv",
                                                 camera_calib_file="sample/waymo_calib.yaml")
    stx.save(stixel_world, "sample/result/")"""
    # Read
    stxl_wrld = stx.read("sample/result/17065833287841703_2980_000_3000_000_165_FRONT.stx1")
    # stx.draw_stixels_in_3d(stxl_wrld)
    """ Use
    stxl_pt_cld, pt_cld_colors = stx.convert_to_point_cloud(stxl_wrld, return_rgb_values=True)"""

    """ Visual in 2D """
    img = stx.decode_img(stxl_wrld)
    img_stxl = stx.draw_stixels_on_image(stxl_wrld)
    img_stxl.show()

    """ Visualize in 3D 
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(stxl_pt_cld)
    point_cloud.colors = o3d.utility.Vector3dVector(pt_cld_colors)
    o3d.visualization.draw_geometries([point_cloud])"""
