import stixel as stx
import numpy as np
from datetime import datetime
from PIL import Image

if __name__ == "__main__":
    """ Convert 
    stixel_world: stx.StixelWorld = stx.read_csv("samples/17065833287841703_2980_000_3000_000_165_FRONT.csv",
                                                 camera_calib_file="samples/waymo_calib.yaml")
    stx.save(stixel_world) """
    # Read & Functions
    stxl_wrld = stx.read("samples/10289507859301986274_4200_000_4220_000_1557847409972029.stx1")
    # stx.add_image(stxl_wrld, Image.open("samples/17065833287841703_2980_000_3000_000_1553628864248893.png").convert("RGB"))
    # stx_mtx = stx.convert_to_matrix(stxl_wrld)
    # stxl_pt_cld = stx.convert_to_point_cloud(stxl_wrld)
    # img = stx.decode_img(stxl_wrld)
    # img.show()
    # img = Image.open("samples/10203656353524179475_7625_000_7645_000_1522688014970187.png")
    # img.show()
    #stxl_wrld_w_img = stx.add_image(stxl_wrld, Image.open("samples/10203656353524179475_7625_000_7645_000_1522688017167470.png"))
    # print(len(stxl_wrld.stixel))
    stxl_wrld = stx.filter_stixels_by_confidence(stxl_wrld, 0.2)
    stxl_wrld = stx.attach_dbscan_clustering(stxl_wrld, eps=1.05, min_samples=3)
    bev = stx.draw_stixels_birds_eye_view(stxl_wrld, instances=True, prob=0.0)
    bev.show()
    stx_img = stx.draw_stixels_on_image(stxl_wrld, prob=0.0, instances=True)
    stx_img.show()
    bboxes = stx.derive_3d_bounding_boxes_from_clusters(stxl_wrld, min_cluster_size=2)
    img_bbox = stx.draw_bbox_on_image(stxl_wrld=stxl_wrld, bboxes=bboxes)
    img_bbox.show()
    """
    stx_img = stx.draw_stixels_on_image(stxl_wrld, prob=0.1)
    stx_img.show()
    for stxl in stxl_wrld.stixel:
        print(stxl.label)
    k = np.array(stxl_wrld.context.calibration.K).reshape(3,3)
    print(k)
    startzeit = datetime.now()
    stxl_wrld = stx.attach_dbscan_clustering(stxl_wrld, eps=0.36, min_samples=2)
    endzeit = datetime.now()
    dm = stx.derive_depth_map_from_stixel_world(stxl_wrld)
    img_dp = stx.draw_depth_map_from_stixel_world(stxl_wrld)
    img_dp.show()
    bboxes = stx.derive_3d_bounding_boxes_from_clusters(stxl_wrld, min_cluster_size=2)
    img_bbox = stx.draw_bbox_on_image(stxl_wrld=stxl_wrld, bboxes=bboxes)
    img_bbox.show()
    print(bboxes)
    # Berechnung der Dauer
    dauer = endzeit - startzeit
    print(f"Die Funktion dauerte: {dauer}")
    # stxl_wrld.context.calibration.K[:] = []
    # stxl_wrld.context.calibration.K.extend(np.array(K.flatten().tolist()))
    """
