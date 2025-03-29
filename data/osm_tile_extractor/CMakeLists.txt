cmake_minimum_required(VERSION 3.16.3)
project(osm_tile_extractor)

set(CMAKE_CXX_STANDARD 17)
add_executable(tiling tiling.cpp)
add_executable(util_test testing/util_test.cpp)

find_package(Protobuf CONFIG REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(boost_geometry CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS geometry)

# Generate proto files for python
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/schema_pb2.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --python_out=${CMAKE_CURRENT_BINARY_DIR}
        schema.proto
        DEPENDS schema.proto
        COMMENT "Generating Python code from schema.proto"
)
add_custom_target(PROTO_PY ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/schema_pb2.py)

# Generate proto files for C++ manually
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.h
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
        schema.proto
        DEPENDS schema.proto
        COMMENT "Generating C++ code from schema.proto"
)

add_custom_target(PROTO_CPP ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.h)

# Add include directories for generated files
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})

# Link libraries
target_link_libraries(tiling PRIVATE lz4::lz4)
target_link_libraries(tiling PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)
target_link_libraries(tiling PRIVATE ZLIB::ZLIB)
target_link_libraries(tiling PRIVATE Boost::geometry)
target_link_libraries(util_test PRIVATE Boost::geometry)

# Add dependencies: ensure Python and C++ proto files are generated before 'tiling'
add_dependencies(tiling PROTO_PY PROTO_CPP)

# Add generated C++ protobuf files to the target sources
target_sources(tiling PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/schema.pb.h)

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
set(SELECTED_FILES
        "${CMAKE_BINARY_DIR}/schema.pb.cc"
        "${CMAKE_BINARY_DIR}/schema.pb.h"
        "${CMAKE_BINARY_DIR}/schema_pb2.py"
        # Add more files here as needed
)

# ==============================
# Step 5: Create Post-Build Commands
# ==============================

# Set the root directory within the build folder
set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/${BUILD_IDENTIFIER}")
# Ensure the root directory exists and copy the executable
add_custom_command(TARGET tiling POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tiling> "${OUT_DIR}"
)

set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/common/")
# Iterate over each selected file and copy it to the root directory
add_custom_command(TARGET tiling POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
)
foreach(file IN LISTS SELECTED_FILES)
    add_custom_command(TARGET tiling POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${file}" "${OUT_DIR}"
    )
endforeach()
