# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: stixel/stixel_world.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from stixel.protos import segmentation_pb2 as stixel_dot_protos_dot_segmentation__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19stixel/stixel_world.proto\x12\x0bstixelworld\x1a stixel/protos/segmentation.proto\"\x88\x01\n\x06Stixel\x12\t\n\x01u\x18\x01 \x01(\x05\x12\n\n\x02vT\x18\x02 \x01(\x05\x12\n\n\x02vB\x18\x03 \x01(\x05\x12\t\n\x01\x64\x18\x04 \x01(\x02\x12-\n\x05label\x18\x05 \x01(\x0e\x32\x1e.stixelworld.Segmentation.Type\x12\r\n\x05width\x18\x06 \x01(\x05\x12\x12\n\nconfidence\x18\x07 \x01(\x02\"\x95\x02\n\nCameraInfo\x12\t\n\x01K\x18\x01 \x03(\x02\x12\t\n\x01T\x18\x02 \x03(\x02\x12\t\n\x01R\x18\x03 \x03(\x02\x12\t\n\x01\x44\x18\x04 \x03(\x02\x12\x44\n\x0f\x44istortionModel\x18\x05 \x01(\x0e\x32+.stixelworld.CameraInfo.DistortionModelType\x12\x11\n\treference\x18\x06 \x01(\t\x12\x10\n\x08img_name\x18\x07 \x01(\t\x12\r\n\x05width\x18\x08 \x01(\x05\x12\x0e\n\x06height\x18\t \x01(\x05\x12\x10\n\x08\x63hannels\x18\n \x01(\x05\"?\n\x13\x44istortionModelType\x12\x13\n\x0fMODEL_UNDEFINED\x10\x00\x12\x13\n\x0fMODEL_PLUMB_BOB\x10\x01\"E\n\x07\x43ontext\x12\x0c\n\x04name\x18\x01 \x01(\t\x12,\n\x0b\x63\x61libration\x18\x02 \x01(\x0b\x32\x17.stixelworld.CameraInfo\"h\n\x0bStixelWorld\x12#\n\x06stixel\x18\x01 \x03(\x0b\x32\x13.stixelworld.Stixel\x12\r\n\x05image\x18\x02 \x01(\x0c\x12%\n\x07\x63ontext\x18\x03 \x01(\x0b\x32\x14.stixelworld.Contextb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'stixel.stixel_world_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_STIXEL']._serialized_start=77
  _globals['_STIXEL']._serialized_end=213
  _globals['_CAMERAINFO']._serialized_start=216
  _globals['_CAMERAINFO']._serialized_end=493
  _globals['_CAMERAINFO_DISTORTIONMODELTYPE']._serialized_start=430
  _globals['_CAMERAINFO_DISTORTIONMODELTYPE']._serialized_end=493
  _globals['_CONTEXT']._serialized_start=495
  _globals['_CONTEXT']._serialized_end=564
  _globals['_STIXELWORLD']._serialized_start=566
  _globals['_STIXELWORLD']._serialized_end=670
# @@protoc_insertion_point(module_scope)
