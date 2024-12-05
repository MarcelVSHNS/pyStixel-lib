from stixel.protos import segmentation_pb2 as _segmentation_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Stixel(_message.Message):
    __slots__ = ("u", "vT", "vB", "d", "label", "width", "confidence", "idx", "cluster")
    U_FIELD_NUMBER: _ClassVar[int]
    VT_FIELD_NUMBER: _ClassVar[int]
    VB_FIELD_NUMBER: _ClassVar[int]
    D_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    u: int
    vT: int
    vB: int
    d: float
    label: _segmentation_pb2.Segmentation.Type
    width: int
    confidence: float
    idx: int
    cluster: int
    def __init__(self, u: _Optional[int] = ..., vT: _Optional[int] = ..., vB: _Optional[int] = ..., d: _Optional[float] = ..., label: _Optional[_Union[_segmentation_pb2.Segmentation.Type, str]] = ..., width: _Optional[int] = ..., confidence: _Optional[float] = ..., idx: _Optional[int] = ..., cluster: _Optional[int] = ...) -> None: ...

class CameraInfo(_message.Message):
    __slots__ = ("K", "T", "R", "D", "DistortionModel", "reference", "img_name", "width", "height", "channels")
    class DistortionModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        MODEL_UNDEFINED: _ClassVar[CameraInfo.DistortionModelType]
        MODEL_PLUMB_BOB: _ClassVar[CameraInfo.DistortionModelType]
    MODEL_UNDEFINED: CameraInfo.DistortionModelType
    MODEL_PLUMB_BOB: CameraInfo.DistortionModelType
    K_FIELD_NUMBER: _ClassVar[int]
    T_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    D_FIELD_NUMBER: _ClassVar[int]
    DISTORTIONMODEL_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    IMG_NAME_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    K: _containers.RepeatedScalarFieldContainer[float]
    T: _containers.RepeatedScalarFieldContainer[float]
    R: _containers.RepeatedScalarFieldContainer[float]
    D: _containers.RepeatedScalarFieldContainer[float]
    DistortionModel: CameraInfo.DistortionModelType
    reference: str
    img_name: str
    width: int
    height: int
    channels: int
    def __init__(self, K: _Optional[_Iterable[float]] = ..., T: _Optional[_Iterable[float]] = ..., R: _Optional[_Iterable[float]] = ..., D: _Optional[_Iterable[float]] = ..., DistortionModel: _Optional[_Union[CameraInfo.DistortionModelType, str]] = ..., reference: _Optional[str] = ..., img_name: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., channels: _Optional[int] = ...) -> None: ...

class Context(_message.Message):
    __slots__ = ("name", "calibration", "clusters")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_FIELD_NUMBER: _ClassVar[int]
    CLUSTERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    calibration: CameraInfo
    clusters: int
    def __init__(self, name: _Optional[str] = ..., calibration: _Optional[_Union[CameraInfo, _Mapping]] = ..., clusters: _Optional[int] = ...) -> None: ...

class StixelWorld(_message.Message):
    __slots__ = ("stixel", "image", "context")
    STIXEL_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    stixel: _containers.RepeatedCompositeFieldContainer[Stixel]
    image: bytes
    context: Context
    def __init__(self, stixel: _Optional[_Iterable[_Union[Stixel, _Mapping]]] = ..., image: _Optional[bytes] = ..., context: _Optional[_Union[Context, _Mapping]] = ...) -> None: ...
