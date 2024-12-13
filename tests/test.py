import stixel as stx
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    """ Convert 
    stixel_world: stx.StixelWorld = stx.read_csv("samples/17065833287841703_2980_000_3000_000_165_FRONT.csv",
                                                 camera_calib_file="samples/waymo_calib.yaml")
    stx.save(stixel_world) """
    # Read & Functions
    # stxl_wrld = stx.read("samples/17065833287841703_2980_000_3000_000_165_FRONT.stx1")
    stxl_wrld = stx.read("samples/set_0_2011_09_29_0026_3.stx1")
    # stx_mtx = stx.convert_to_matrix(stxl_wrld)
    # stxl_pt_cld = stx.convert_to_point_cloud(stxl_wrld)
    # img = stx.decode_img(stxl_wrld)
    # img.show()
    k = np.array(stxl_wrld.context.calibration.K).reshape(3,3)

    # stxl_wrld = stx.attach_dbscan_clustering(stxl_wrld)

    # stxl_wrld.context.calibration.K[:] = []
    # stxl_wrld.context.calibration.K.extend(np.array(K.flatten().tolist()))

    """ Visualize in 2D """
    img_stxl = stx.draw_stixels_on_image(stxl_wrld, instances=False)
    img_stxl.show()

    """ Visualize in 3D """
    stx.draw_stixels_in_3d(stxl_wrld, instances=False)
