#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(unified3D LANGUAGES C CXX)

set(CORE_FILES
        core/Device.h
        core/Device.cpp
)

set(PROJECT_FILES
        ${CORE_FILES}
)

add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        metal-cpp
)

#set(KERNEL_FIELS
#        ${CMAKE_CURRENT_SOURCE_DIR}/../shader/builtin/mad_throughput.metal
#        ${CMAKE_CURRENT_SOURCE_DIR}/../shader/builtin/reduce.metal
#)
#
#build_metallib(
#        TARGET metal_kernel_metallib
#        TITLE metal_kernel
#        SOURCES ${KERNEL_FIELS}
#        INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
#        OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
#
#add_dependencies(
#        ${PROJECT_NAME}
#        metal_kernel_metallib
#)
#
#set(MLX_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR}/)
#target_compile_definitions(
#        ${PROJECT_NAME} PRIVATE METAL_PATH="${MLX_METAL_PATH}/metal_kernel.metallib")
#
## python install
#install(
#        FILES ${MLX_METAL_PATH}/metal_kernel.metallib
#        DESTINATION arche_compute
#        COMPONENT metallib
#)