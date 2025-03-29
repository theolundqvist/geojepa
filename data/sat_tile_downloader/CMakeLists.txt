# CMakeLists.txt
cmake_minimum_required(VERSION 3.16.3)
project(merge_tiles)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find BS_thread_pool.hpp
find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS "BS_thread_pool.hpp")
if(NOT BSHOSHANY_THREAD_POOL_INCLUDE_DIRS)
    message(FATAL_ERROR "BS_thread_pool.hpp not found.")
endif()

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Find WebP using pkg-config
pkg_check_modules(WEBP REQUIRED libwebp)
if(NOT WEBP_FOUND)
    message(FATAL_ERROR "WebP library not found.")
endif()

# Find libpng using pkg-config
pkg_check_modules(PNG REQUIRED libpng)
if(NOT PNG_FOUND)
    message(FATAL_ERROR "libpng library not found.")
endif()

# Include directories for merge_tiles
include_directories(
        ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS}
        ${WEBP_INCLUDE_DIRS}
        ${PNG_INCLUDE_DIRS}
)

# Link directories for merge_tiles
link_directories(
        ${WEBP_LIBRARY_DIRS}
        ${PNG_LIBRARY_DIRS}
)

# Add executable for merge_tiles
add_executable(merge_tiles main.cpp)
target_link_libraries(merge_tiles PRIVATE ${WEBP_LIBRARIES} ${PNG_LIBRARIES})

# Add executable for split_images
add_executable(split_images split.cpp)
target_include_directories(split_images PRIVATE ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS} ${WEBP_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
target_link_libraries(split_images PRIVATE ${WEBP_LIBRARIES} ${PNG_LIBRARIES})

# Set RPATH for runtime linking
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#if(LINUX)
#    link_directories("${PROJECT_SOURCE_DIR}/platforms/linux")
#elseif(MACOS)
#    link_directories("${PROJECT_SOURCE_DIR}/platforms/macos")
#endif()
#
#if(LINUX)
#    install (FILES "${PROJECT_SOURCE_DIR}/platforms/linux/merge_tiles.so" DESTINATION lib)
#elseif(MACOS)
#    install (FILES "${PROJECT_SOURCE_DIR}/platforms/macos/merge_tiles.dylib" DESTINATION lib)
#endif()

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
add_custom_command(TARGET merge_tiles POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:merge_tiles> "${OUT_DIR}"
)
add_custom_command(TARGET split_images POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:split_images> "${OUT_DIR}"
)

#set(OUT_DIR "${CMAKE_SOURCE_DIR}/builds/common/")
## Iterate over each selected file and copy it to the root directory
#add_custom_command(TARGET merge_tiles POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
#)
#foreach(file IN LISTS SELECTED_FILES)
#    add_custom_command(TARGET merge_tiles POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${file}" "${OUT_DIR}"
#    )
#endforeach()
