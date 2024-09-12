from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Stixel(_message.Message):
    __slots__ = ("u", "vT", "vB", "d", "label", "width", "confidence")
    U_FIELD_NUMBER: _ClassVar[int]
    VT_FIELD_NUMBER: _ClassVar[int]
    VB_FIELD_NUMBER: _ClassVar[int]
    D_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    u: int
    vT: int
    vB: int
    d: float
    label: Segmentation.Type
    width: int
    confidence: float
    def __init__(self, u: _Optional[int] = ..., vT: _Optional[int] = ..., vB: _Optional[int] = ..., d: _Optional[float] = ..., label: _Optional[_Union[Segmentation.Type, str]] = ..., width: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...

class Segmentation(_message.Message):
    __slots__ = ()
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_UNDEFINED: _ClassVar[Segmentation.Type]
        TYPE_CAR: _ClassVar[Segmentation.Type]
        TYPE_TRUCK: _ClassVar[Segmentation.Type]
        TYPE_BUS: _ClassVar[Segmentation.Type]
        TYPE_OTHER_VEHICLE: _ClassVar[Segmentation.Type]
        TYPE_MOTORCYCLIST: _ClassVar[Segmentation.Type]
        TYPE_BICYCLIST: _ClassVar[Segmentation.Type]
        TYPE_PEDESTRIAN: _ClassVar[Segmentation.Type]
        TYPE_SIGN: _ClassVar[Segmentation.Type]
        TYPE_TRAFFIC_LIGHT: _ClassVar[Segmentation.Type]
        TYPE_POLE: _ClassVar[Segmentation.Type]
        TYPE_CONSTRUCTION_CONE: _ClassVar[Segmentation.Type]
        TYPE_BICYCLE: _ClassVar[Segmentation.Type]
        TYPE_MOTORCYCLE: _ClassVar[Segmentation.Type]
        TYPE_BUILDING: _ClassVar[Segmentation.Type]
        TYPE_VEGETATION: _ClassVar[Segmentation.Type]
        TYPE_TREE_TRUNK: _ClassVar[Segmentation.Type]
        TYPE_CURB: _ClassVar[Segmentation.Type]
        TYPE_ROAD: _ClassVar[Segmentation.Type]
        TYPE_LANE_MARKER: _ClassVar[Segmentation.Type]
        TYPE_OTHER_GROUND: _ClassVar[Segmentation.Type]
        TYPE_WALKABLE: _ClassVar[Segmentation.Type]
        TYPE_SIDEWALK: _ClassVar[Segmentation.Type]
    TYPE_UNDEFINED: Segmentation.Type
    TYPE_CAR: Segmentation.Type
    TYPE_TRUCK: Segmentation.Type
    TYPE_BUS: Segmentation.Type
    TYPE_OTHER_VEHICLE: Segmentation.Type
    TYPE_MOTORCYCLIST: Segmentation.Type
    TYPE_BICYCLIST: Segmentation.Type
    TYPE_PEDESTRIAN: Segmentation.Type
    TYPE_SIGN: Segmentation.Type
    TYPE_TRAFFIC_LIGHT: Segmentation.Type
    TYPE_POLE: Segmentation.Type
    TYPE_CONSTRUCTION_CONE: Segmentation.Type
    TYPE_BICYCLE: Segmentation.Type
    TYPE_MOTORCYCLE: Segmentation.Type
    TYPE_BUILDING: Segmentation.Type
    TYPE_VEGETATION: Segmentation.Type
    TYPE_TREE_TRUNK: Segmentation.Type
    TYPE_CURB: Segmentation.Type
    TYPE_ROAD: Segmentation.Type
    TYPE_LANE_MARKER: Segmentation.Type
    TYPE_OTHER_GROUND: Segmentation.Type
    TYPE_WALKABLE: Segmentation.Type
    TYPE_SIDEWALK: Segmentation.Type
    def __init__(self) -> None: ...

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
    __slots__ = ("name", "calibration")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    calibration: CameraInfo
    def __init__(self, name: _Optional[str] = ..., calibration: _Optional[_Union[CameraInfo, _Mapping]] = ...) -> None: ...

class StixelWorld(_message.Message):
    __slots__ = ("stixel", "context")
    class Image(_message.Message):
        __slots__ = ("data", "encoding")
        DATA_FIELD_NUMBER: _ClassVar[int]
        ENCODING_FIELD_NUMBER: _ClassVar[int]
        data: bytes
        encoding: str
        def __init__(self, data: _Optional[bytes] = ..., encoding: _Optional[str] = ...) -> None: ...
    STIXEL_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    stixel: _containers.RepeatedCompositeFieldContainer[Stixel]
    context: Context
    def __init__(self, stixel: _Optional[_Iterable[_Union[Stixel, _Mapping]]] = ..., context: _Optional[_Union[Context, _Mapping]] = ...) -> None: ...