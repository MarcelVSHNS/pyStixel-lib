import stixel as stx
from datetime import datetime


if __name__ == "__main__":
    """ Convert  """
    stixel_world: stx.StixelWorld = stx.read_csv("samples/17065833287841703_2980_000_3000_000_165_FRONT.csv",
                                                 camera_calib_file="samples/waymo_calib.yaml")
    stx.save(stixel_world)
    # Read & Functions
    stxl_wrld = stx.read("/home/marcel/workspace/datasets/waymo-od/validation/Stixel_bbox/10289507859301986274_4200_000_4220_000_24_FRONT.stx1")
    # stx_mtx = stx.convert_to_matrix(stxl_wrld)
    # stxl_pt_cld = stx.convert_to_point_cloud(stxl_wrld)
    # img = stx.decode_img(stxl_wrld)
    # img.show()

    """ Visual in 2D 
    img_stxl = stx.draw_stixels_on_image(stxl_wrld)
    img_stxl.show() """

    """ Visualize in 3D 
    stx.draw_stixels_in_3d(stxl_wrld) """
