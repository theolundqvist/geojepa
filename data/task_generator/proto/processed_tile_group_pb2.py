# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: processed_tile_group.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1aprocessed_tile_group.proto\x12\tprocessed"\xb7\x01\n\x07\x46\x65\x61ture\x12\x0c\n\x04tags\x18\x01 \x03(\t\x12\x0f\n\x07min_box\x18\x02 \x03(\x02\x12\x0c\n\x04\x61rea\x18\x03 \x01(\x02\x12\r\n\x05width\x18\x04 \x01(\x02\x12\x0e\n\x06height\x18\x05 \x01(\x02\x12\x10\n\x08rotation\x18\x06 \x01(\x02\x12\x10\n\x08is_point\x18\x07 \x01(\x08\x12\x13\n\x0bis_polyline\x18\x08 \x01(\x08\x12\x12\n\nis_polygon\x18\t \x01(\x08\x12\x13\n\x0bis_relation\x18\n \x01(\x08"\xb8\x01\n\x04Tile\x12\r\n\x05nodes\x18\x01 \x03(\x02\x12\x14\n\x0clocal_coords\x18\x02 \x03(\x02\x12\x13\n\x0binter_edges\x18\x03 \x03(\x05\x12\x13\n\x0bintra_edges\x18\x04 \x03(\x05\x12\x17\n\x0fnode_to_feature\x18\x05 \x03(\x05\x12$\n\x08\x66\x65\x61tures\x18\x06 \x03(\x0b\x32\x12.processed.Feature\x12\x0c\n\x04zoom\x18\x07 \x01(\x05\x12\t\n\x01x\x18\x08 \x01(\x05\x12\t\n\x01y\x18\t \x01(\x05"O\n\tTileGroup\x12\x0c\n\x04zoom\x18\x01 \x01(\x05\x12\t\n\x01x\x18\x02 \x01(\x05\x12\t\n\x01y\x18\x03 \x01(\x05\x12\x1e\n\x05tiles\x18\x04 \x03(\x0b\x32\x0f.processed.Tileb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "processed_tile_group_pb2", _globals
)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _globals["_FEATURE"]._serialized_start = 42
    _globals["_FEATURE"]._serialized_end = 225
    _globals["_TILE"]._serialized_start = 228
    _globals["_TILE"]._serialized_end = 412
    _globals["_TILEGROUP"]._serialized_start = 414
    _globals["_TILEGROUP"]._serialized_end = 493
# @@protoc_insertion_point(module_scope)
