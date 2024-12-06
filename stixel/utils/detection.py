import importlib.util
from ..stixel_world_pb2 import StixelWorld
from .transformation import convert_to_3d_stixel

def attach_dbscan_clustering(stxl_wrld: StixelWorld, eps: float = 1.42, min_samples: int = 2) -> StixelWorld:
    """
    Attaches DBSCAN clustering information to a `StixelWorld` object.

    This function performs clustering on the 3D stixels of a `StixelWorld` object in
    bird's-eye view (BEV) using the DBSCAN algorithm. Each stixel is assigned to a
    cluster, and the cluster labels are stored in the `cluster` field of each stixel.
    The total number of clusters is saved in the `context.clusters` attribute.

    Args:
        stxl_wrld (StixelWorld): A `StixelWorld` object containing stixels to cluster.
        eps (float, optional): The maximum distance between two samples for them
            to be considered as part of the same cluster. Defaults to 1.42.
        min_samples (int, optional): The number of samples in a neighborhood for
            a point to be considered a core point. Defaults to 2.

    Returns:
        StixelWorld: The modified `StixelWorld` object with updated cluster information.

    Raises:
        ImportError: If the `sklearn` library is not installed.
    """
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
