import importlib.util
from ..stixel_world_pb2 import StixelWorld
from .transformation import convert_to_3d_stixel

def attach_dbscan_clustering(stxl_wrld: StixelWorld, eps: float = 1.42, min_samples: int = 2) -> StixelWorld:
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("Install 'sklearn' in your Python environment with: 'python -m pip install sklearn'. ")
    from sklearn.cluster import DBSCAN
    points = convert_to_3d_stixel(stxl_wrld)
    # BEV view
    bev_points = points[:, :2]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(bev_points)
    for i in range(len(stxl_wrld.stixel)):
        stxl_wrld.stixel[i].cluster = labels[i]
    stxl_wrld.context.clusters = labels.max()
    return stxl_wrld
