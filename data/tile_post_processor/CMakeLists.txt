cmake_minimum_required(VERSION 3.16.3)
project(tile_post_processor)
set(CMAKE_C_COMPILER, "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER, "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS geometry)
find_package(boost_geometry CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Generate proto files for python
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        processed_tile_group.proto
        DEPENDS processed_tile_group.proto
        COMMENT "Generating Python code from processed_tile_group.proto"
)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        unprocessed_tile_group.proto
        DEPENDS unprocessed_tile_group.proto
        COMMENT "Generating Python code from unprocessed_tile_group.proto"
)
add_custom_target(PROTO_PY ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.py ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.py)

# Generate proto files for C++ manually
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        processed_tile_group.proto
        DEPENDS processed_tile_group.proto
        COMMENT "Generating C++ code from processed_tile_group.proto"
)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
        unprocessed_tile_group.proto
        DEPENDS unprocessed_tile_group.proto
        COMMENT "Generating C++ code from unprocessed_tile_group.proto"
)
add_custom_target(PROTO_CPP_PROCESSED ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.h)
add_custom_target(PROTO_CPP_UNPROCESSED ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.h)


add_executable(tile_post_processor main.cpp
        cxxopts.h
        utils.h
        indicators.hpp
)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/proto)
target_link_libraries(tile_post_processor PRIVATE Boost::geometry)
target_link_libraries(tile_post_processor PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

# Add dependencies: ensure Python and C++ proto files are generated before 'tile_post_processor'
add_dependencies(tile_post_processor PROTO_PY PROTO_CPP_PROCESSED PROTO_CPP_UNPROCESSED)
# Add generated C++ protobuf files to the target sources
target_sources(tile_post_processor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/processed_tile_group.pb.h)
target_sources(tile_post_processor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/proto/unprocessed_tile_group.pb.h)

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
# Step 2: Define the Root Directory
# ==============================


# ==============================
# Step 4: Specify Additional Files to Copy
# ==============================

# List of additional files to include with the executable
#set(SELECTED_FILES
#        "${CMAKE_BINARY_DIR}/processed_tile_group.pb.cc"
#        "${CMAKE_BINARY_DIR}/processed_tile_group.pb.h"
#        "${CMAKE_BINARY_DIR}/processed_tile_group_pb2.py"
#        "${CMAKE_BINARY_DIR}/unprocessed_tile_group.pb.cc"
#        "${CMAKE_BINARY_DIR}/unprocessed_tile_group.pb.h"
#        "${CMAKE_BINARY_DIR}/unprocessed_tile_group_pb2.py"
#        # Add more files here as needed
#)

# ==============================
# Step 5: Create Post-Build Commands
# ==============================

# Set the root directory within the build folder
set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/${BUILD_IDENTIFIER}")
# Ensure the root directory exists and copy the executable
add_custom_command(TARGET tile_post_processor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_post_processor> "${OUT_DIR}"
)

#set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/common/")
## Iterate over each selected file and copy it to the root directory
#add_custom_command(TARGET tile_post_processor POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
#)
#foreach(file IN LISTS SELECTED_FILES)
#    add_custom_command(TARGET tile_post_processor POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${file}" "${OUT_DIR}"
#    )
#endforeach()