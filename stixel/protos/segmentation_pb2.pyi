from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

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

class BboxSegmentation(_message.Message):
    __slots__ = ()
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_UNKNOWN: _ClassVar[BboxSegmentation.Type]
        TYPE_VEHICLE: _ClassVar[BboxSegmentation.Type]
        TYPE_PEDESTRIAN: _ClassVar[BboxSegmentation.Type]
        TYPE_SIGN: _ClassVar[BboxSegmentation.Type]
        TYPE_CYCLIST: _ClassVar[BboxSegmentation.Type]
    TYPE_UNKNOWN: BboxSegmentation.Type
    TYPE_VEHICLE: BboxSegmentation.Type
    TYPE_PEDESTRIAN: BboxSegmentation.Type
    TYPE_SIGN: BboxSegmentation.Type
    TYPE_CYCLIST: BboxSegmentation.Type
    def __init__(self) -> None: ...
