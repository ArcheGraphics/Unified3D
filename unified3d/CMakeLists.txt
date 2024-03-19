#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.27)

project(Unified3D LANGUAGES C CXX)

set(CORE_FILES
        core/Device.h
        core/Device.cpp
        core/Dtype.h
        core/Dtype.cpp
        core/SizeVector.h
        core/SizeVector.cpp
        core/SmallVector.h
        core/SmallVector.cpp
        core/MemoryManager.h
        core/MemoryManager.cpp
        core/ShapeUtil.h
        core/ShapeUtil.cpp
        core/Tensor.h
        core/Tensor.cpp
        core/TensorCheck.h
        core/TensorCheck.cpp
        core/TensorFunction.h
        core/TensorFunction.cpp
        core/TensorKey.h
        core/TensorKey.cpp
        core/TensorList.h
        core/TensorList.cpp
        core/Parallel.h
        core/Parallel.cpp
        core/Indexer.h
        core/Indexer.cpp
        core/AdvancedIndexing.h
        core/AdvancedIndexing.cpp
        # Kernel
        core/kernel/Kernel.h
        core/kernel/Kernel.cpp
        core/kernel/Arange.h
        core/kernel/Arange.cpp
        core/kernel/ArangeCPU.cpp
        core/kernel/BinaryEW.h
        core/kernel/BinaryEW.cpp
        core/kernel/BinaryEWCPU.cpp
        core/kernel/IndexGetSet.h
        core/kernel/IndexGetSet.cpp
        core/kernel/IndexGetSetCPU.cpp
        core/kernel/IndexReduction.h
        core/kernel/IndexReduction.cpp
        core/kernel/IndexReductionCPU.cpp
        core/kernel/NonZero.h
        core/kernel/NonZero.cpp
        core/kernel/NonZeroCPU.cpp
        core/kernel/Reduction.h
        core/kernel/Reduction.cpp
        core/kernel/ReductionCPU.cpp
        core/kernel/UnaryEW.h
        core/kernel/UnaryEW.cpp
        core/kernel/UnaryEWCPU.cpp
        # linalg
        core/linalg/AddMM.h
        core/linalg/AddMM.cpp
        core/linalg/AddMMCPU.cpp
        core/linalg/Det.h
        core/linalg/Det.cpp
        core/linalg/Inverse.h
        core/linalg/Inverse.cpp
        core/linalg/InverseCPU.cpp
        core/linalg/LeastSquares.h
        core/linalg/LeastSquares.cpp
        core/linalg/LeastSquaresCPU.cpp
        core/linalg/LU.h
        core/linalg/LU.cpp
        core/linalg/LUCPU.cpp
        core/linalg/LUImpl.h
        core/linalg/Matmul.h
        core/linalg/Matmul.cpp
        core/linalg/MatmulCPU.cpp
        core/linalg/Solve.h
        core/linalg/Solve.cpp
        core/linalg/SolveCPU.cpp
        core/linalg/SVD.h
        core/linalg/SVD.cpp
        core/linalg/SVDCPU.cpp
        core/linalg/Tri.h
        core/linalg/Tri.cpp
        core/linalg/TriCPU.cpp
        core/linalg/TriImpl.h
)

set(METAL_FILES
        metal/Metal.h
        metal/Metal.cpp
        metal/Device.h
        metal/Device.cpp
        metal/Buffer.h
        metal/Buffer.cpp
)

set(UTILITY_FILES
        utility/Logging.h
        utility/Logging.cpp
        utility/Helper.h
        utility/Helper.cpp
        utility/CPUInfo.h
        utility/CPUInfo.cpp
        utility/Eigen.h
        utility/Eigen.cpp
        utility/FileSystem.h
        utility/FileSystem.cpp
        utility/IJsonConvertible.h
        utility/IJsonConvertible.cpp
        utility/ProgressBar.h
        utility/ProgressBar.cpp
        utility/ProgressReporters.h
        utility/Random.h
        utility/Random.cpp
        utility/Timer.h
        utility/Timer.cpp
)

set(PROJECT_FILES
        ${CORE_FILES}
        ${METAL_FILES}
        ${UTILITY_FILES}
)

add_library(${PROJECT_NAME} STATIC ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/metal-cpp
        /opt/homebrew/opt/lapack/include
        /opt/homebrew/opt/openblas/include
)

target_link_directories(${PROJECT_NAME} PUBLIC
        /opt/homebrew/opt/lapack/lib
        /opt/homebrew/opt/openblas/lib
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        fmt::fmt-header-only
        Eigen3::Eigen
        JsonCpp::JsonCpp
        lapacke
        openblas
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
set(MLX_METAL_PATH ${CMAKE_CURRENT_BINARY_DIR}/)
#target_compile_definitions(
#        ${PROJECT_NAME} PRIVATE METAL_PATH="${MLX_METAL_PATH}/metal_kernel.metallib")
#
## python install
#install(
#        FILES ${MLX_METAL_PATH}/metal_kernel.metallib
#        DESTINATION arche_compute
#        COMPONENT metallib
#)