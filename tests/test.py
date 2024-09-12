import stixel as stx


if __name__ == "__main__":
    """ Convert 
    stixel_world: stx.StixelWorld = stx.read_csv("sample/17065833287841703_2980_000_3000_000_165_FRONT.csv",
                                                 camera_calib_file="sample/waymo_calib.yaml")
    stx.save(stixel_world, "sample/result/") """
    # Read & Functions
    stxl_wrld = stx.read("sample/result/17065833287841703_2980_000_3000_000_165_FRONT.stx1")
    stx_mtx = stx.convert_to_matrix(stxl_wrld)
    print(stx_mtx[0])
    stxl_pt_cld = stx.convert_to_point_cloud(stxl_wrld)
    # img = stx.decode_img(stxl_wrld)

    """ Visual in 2D 
    img_stxl = stx.draw_stixels_on_image(stxl_wrld)
    img_stxl.show() """

    """ Visualize in 3D 
    stx.draw_stixels_in_3d(stxl_wrld) """
