# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: schema.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0cschema.proto"p\n\x08Geometry\x12\x1f\n\x06points\x18\x01 \x03(\x0b\x32\x0f.Geometry.Point\x12\x11\n\tis_closed\x18\x02 \x01(\x08\x12\r\n\x05inner\x18\x03 \x01(\x08\x1a!\n\x05Point\x12\x0b\n\x03lat\x18\x01 \x01(\x02\x12\x0b\n\x03lon\x18\x02 \x01(\x02"u\n\x07\x46\x65\x61ture\x12 \n\x04tags\x18\x01 \x03(\x0b\x32\x12.Feature.TagsEntry\x12\x1b\n\x08geometry\x18\x02 \x01(\x0b\x32\t.Geometry\x1a+\n\tTagsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01"m\n\x05Group\x12\x1e\n\x04tags\x18\x01 \x03(\x0b\x32\x10.Group.TagsEntry\x12\x17\n\x0f\x66\x65\x61ture_indices\x18\x02 \x03(\x05\x1a+\n\tTagsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01"^\n\x04Tile\x12\x0c\n\x04zoom\x18\x01 \x01(\x05\x12\t\n\x01x\x18\x02 \x01(\x05\x12\t\n\x01y\x18\x03 \x01(\x05\x12\x1a\n\x08\x66\x65\x61tures\x18\x04 \x03(\x0b\x32\x08.Feature\x12\x16\n\x06groups\x18\x05 \x03(\x0b\x32\x06.Group"E\n\tTileGroup\x12\x0c\n\x04zoom\x18\x01 \x01(\x05\x12\t\n\x01x\x18\x02 \x01(\x05\x12\t\n\x01y\x18\x03 \x01(\x05\x12\x14\n\x05tiles\x18\x04 \x03(\x0b\x32\x05.Tileb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "schema_pb2", _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _globals["_FEATURE_TAGSENTRY"]._options = None
    _globals["_FEATURE_TAGSENTRY"]._serialized_options = b"8\001"
    _globals["_GROUP_TAGSENTRY"]._options = None
    _globals["_GROUP_TAGSENTRY"]._serialized_options = b"8\001"
    _globals["_GEOMETRY"]._serialized_start = 16
    _globals["_GEOMETRY"]._serialized_end = 128
    _globals["_GEOMETRY_POINT"]._serialized_start = 95
    _globals["_GEOMETRY_POINT"]._serialized_end = 128
    _globals["_FEATURE"]._serialized_start = 130
    _globals["_FEATURE"]._serialized_end = 247
    _globals["_FEATURE_TAGSENTRY"]._serialized_start = 204
    _globals["_FEATURE_TAGSENTRY"]._serialized_end = 247
    _globals["_GROUP"]._serialized_start = 249
    _globals["_GROUP"]._serialized_end = 358
    _globals["_GROUP_TAGSENTRY"]._serialized_start = 204
    _globals["_GROUP_TAGSENTRY"]._serialized_end = 247
    _globals["_TILE"]._serialized_start = 360
    _globals["_TILE"]._serialized_end = 454
    _globals["_TILEGROUP"]._serialized_start = 456
    _globals["_TILEGROUP"]._serialized_end = 525
# @@protoc_insertion_point(module_scope)
