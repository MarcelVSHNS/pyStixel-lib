# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: stixel/protos/segmentation.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n stixel/protos/segmentation.proto\x12\x0bstixelworld\"\xd6\x03\n\x0cSegmentation\"\xc5\x03\n\x04Type\x12\x12\n\x0eTYPE_UNDEFINED\x10\x00\x12\x0c\n\x08TYPE_CAR\x10\x01\x12\x0e\n\nTYPE_TRUCK\x10\x02\x12\x0c\n\x08TYPE_BUS\x10\x03\x12\x16\n\x12TYPE_OTHER_VEHICLE\x10\x04\x12\x15\n\x11TYPE_MOTORCYCLIST\x10\x05\x12\x12\n\x0eTYPE_BICYCLIST\x10\x06\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x07\x12\r\n\tTYPE_SIGN\x10\x08\x12\x16\n\x12TYPE_TRAFFIC_LIGHT\x10\t\x12\r\n\tTYPE_POLE\x10\n\x12\x1a\n\x16TYPE_CONSTRUCTION_CONE\x10\x0b\x12\x10\n\x0cTYPE_BICYCLE\x10\x0c\x12\x13\n\x0fTYPE_MOTORCYCLE\x10\r\x12\x11\n\rTYPE_BUILDING\x10\x0e\x12\x13\n\x0fTYPE_VEGETATION\x10\x0f\x12\x13\n\x0fTYPE_TREE_TRUNK\x10\x10\x12\r\n\tTYPE_CURB\x10\x11\x12\r\n\tTYPE_ROAD\x10\x12\x12\x14\n\x10TYPE_LANE_MARKER\x10\x13\x12\x15\n\x11TYPE_OTHER_GROUND\x10\x14\x12\x11\n\rTYPE_WALKABLE\x10\x15\x12\x11\n\rTYPE_SIDEWALK\x10\x16\"t\n\x10\x42\x62oxSegmentation\"`\n\x04Type\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x10\n\x0cTYPE_VEHICLE\x10\x01\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x02\x12\r\n\tTYPE_SIGN\x10\x03\x12\x10\n\x0cTYPE_CYCLIST\x10\x04\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'stixel.protos.segmentation_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SEGMENTATION']._serialized_start=50
  _globals['_SEGMENTATION']._serialized_end=520
  _globals['_SEGMENTATION_TYPE']._serialized_start=67
  _globals['_SEGMENTATION_TYPE']._serialized_end=520
  _globals['_BBOXSEGMENTATION']._serialized_start=522
  _globals['_BBOXSEGMENTATION']._serialized_end=638
  _globals['_BBOXSEGMENTATION_TYPE']._serialized_start=542
  _globals['_BBOXSEGMENTATION_TYPE']._serialized_end=638
# @@protoc_insertion_point(module_scope)