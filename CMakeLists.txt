cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
# set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")

# Setting up ccache or sccache
find_program(CCACHE "ccache")
if(CCACHE)
    message("${CCACHE} has been found in the system!")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
else()
    find_program(SCCACHE "sccache")
    if(SCCACHE)
        message("${SCCACHE} has been found in the system!")
        set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE})
    else()
        message("neither ccache or sccache have been found in the system!")
    endif(SCCACHE)
endif(CCACHE)

# Project
set(PROJECT_DESCRIPTION "Robot Vision")
set(ONLINE_REPOSITORY "https://github.com/thetooth/robot-vision")
project(
    "robot-vision"
        DESCRIPTION ${PROJECT_DESCRIPTION}
        HOMEPAGE_URL ${ONLINE_REPOSITORY}
)

set(OpenCV_DIR "/home/user/opencv_dist/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
add_library(opencv INTERFACE IMPORTED)

set(OpenVINO_DIR "/home/user/openvino_dist/runtime/cmake")
find_package(OpenVINO CONFIG REQUIRED)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# Clang compile commands are generated here:
# build/Debug/compile_commands.json
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/int")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/spdlog)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/json)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nats.c)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include
    ${OpenCV_INCLUDE_DIRS}
    openvino::runtime
)

message(STATUS, ${OPENVINO_INCLUDE_DIR})

link_directories(lib)
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    openvino::runtime
    nats_static
    nlohmann_json::nlohmann_json
)

# Copy yolov8n.onnx file to the same folder of the executable file
configure_file(yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)
configure_file(yolov8s.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8s.onnx COPYONLY)
