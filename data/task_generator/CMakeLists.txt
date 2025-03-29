cmake_minimum_required(VERSION 3.16.3)
project(tile_task_generator)

set(CMAKE_CXX_STANDARD 17)

# Create the executable target
add_executable(task_generator generator.cpp
        generator.h
        utils.h)

add_executable(cooccurrence co-occurrence.cpp
        co-occurrence.h)

add_executable(ag_generator ag_csv_generator.cpp
        ag_csv_generator.h)

add_executable(tag_pruner ag_tag_pruner.cpp ag_tag_pruner.h)

# Find packages

find_package(Protobuf CONFIG REQUIRED)
find_package(boost_geometry CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS geometry)

# Link libraries
target_link_libraries(task_generator PRIVATE
        protobuf::libprotoc
        protobuf::libprotobuf
        protobuf::libprotobuf-lite
)
target_link_libraries(cooccurrence PRIVATE
        protobuf::libprotoc
        protobuf::libprotobuf
        protobuf::libprotobuf-lite
)
target_link_libraries(ag_generator PRIVATE
        protobuf::libprotoc
        protobuf::libprotobuf
        protobuf::libprotobuf-lite
)
target_link_libraries(tag_pruner PRIVATE
        protobuf::libprotoc
        protobuf::libprotobuf
        protobuf::libprotobuf-lite
)

# Generate proto files for Python
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/schema_pb2.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        unprocessed_tile_group.proto
        DEPENDS unprocessed_tile_group.proto
        COMMENT "Generating Python code from unprocessed_tile_group.proto"
)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_schema_pb2.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        processed_tile_group.proto
        DEPENDS processed_tile_group.proto
        COMMENT "Generating Python code from processed_tile_group.proto"
)

add_custom_target(PROTO_PY ALL DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/schema_pb2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_schema_pb2.py
)

# Generate proto files for C++
# First proto file: unprocessed_tile_group
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        unprocessed_tile_group.proto
        DEPENDS unprocessed_tile_group.proto
        COMMENT "Generating C++ code from unprocessed_tile_group.proto"
)

# Second proto file: processed_tile_group
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        processed_tile_group.proto
        DEPENDS processed_tile_group.proto
        COMMENT "Generating C++ code from processed_tile_group.proto"
)

# Create custom targets for both proto files
add_custom_target(PROTO_CPP ALL DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.h
)

# Include Boost directories
include_directories(${Boost_INCLUDE_DIRS})

# Link Boost geometry
target_link_libraries(task_generator PRIVATE Boost::geometry)

# Ensure the executable depends on the generated protobuf files
add_dependencies(task_generator PROTO_CPP PROTO_PY)
add_dependencies(cooccurrence PROTO_CPP PROTO_PY)
add_dependencies(ag_generator PROTO_CPP PROTO_PY)
add_dependencies(tag_pruner PROTO_CPP PROTO_PY)

# Add generated source files to the target
target_sources(task_generator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc
)
target_sources(cooccurrence PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc
)
target_sources(ag_generator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc
)
target_sources(tag_pruner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc
)

# ==============================
# Step 1: Determine System Properties
# ==============================

# Identify the operating system
if(WIN32)
    set(OS_NAME "win")
elseif(APPLE)
    set(OS_NAME "macos")
elseif(UNIX)
    set(OS_NAME "linux")
else()
    set(OS_NAME "UnknownOS")
endif()

# Identify the CPU architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|x64)$")
    set(CPU_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i386|i686|x86)$")
    set(CPU_ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|armv8|arm64)$")
    set(CPU_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7)$")
    set(CPU_ARCH "arm")
else()
    set(CPU_ARCH "UnknownArch")
endif()

# Combine OS and CPU architecture to form a unique build identifier
set(BUILD_IDENTIFIER "${OS_NAME}_${CPU_ARCH}")

# ==============================
# Step 4: Specify Additional Files to Copy
# ==============================

# List of additional files to include with the executable
#set(SELECTED_FILES
#        "${CMAKE_BINARY_DIR}/schema.pb.cc"
#        "${CMAKE_BINARY_DIR}/schema.pb.h"
#        "${CMAKE_BINARY_DIR}/schema_pb2.py"
#        # Add more files here as needed
#)

# ==============================
# Step 5: Create Post-Build Commands
# ==============================

# Set the root directory within the build folder
set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/${BUILD_IDENTIFIER}")
# Ensure the root directory exists and copy the executable
add_custom_command(TARGET task_generator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:task_generator> "${OUT_DIR}"
)

add_custom_command(TARGET cooccurrence POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cooccurrence> "${OUT_DIR}"
)

add_custom_command(TARGET ag_generator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ag_generator> "${OUT_DIR}"
)
add_custom_command(TARGET tag_pruner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tag_pruner> "${OUT_DIR}"
)

#set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/common/")
## Iterate over each selected file and copy it to the root directory
#add_custom_command(TARGET task_generator POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
#)
#foreach(file IN LISTS SELECTED_FILES)
#    add_custom_command(TARGET task_generator POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${file}" "${OUT_DIR}"
#    )
#endforeach()
