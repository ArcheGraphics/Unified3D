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
        utility/Download.h
        utility/Download.cpp
        utility/Extract.h
        utility/Extract.cpp
        utility/ExtractZIP.h
        utility/ExtractZIP.cpp
)

set(DATA_FILES
        data/Dataset.h
        data/Dataset.cpp
)

set(TENSOR_FILES
        tensor/io/NumpyIO.h
        tensor/io/NumpyIO.cpp
)

set(GEOMETRY_FILES
        geometry/BoundingVolume.h
        geometry/BoundingVolume.cpp
        geometry/Image.h
        geometry/Image.cpp
        geometry/ImageFactory.cpp
        geometry/RGBDImage.h
        geometry/RGBDImage.cpp
        geometry/RGBDImageFactory.cpp

        geometry/MeshBase.h
        geometry/MeshBase.cpp
        geometry/TetraMesh.h
        geometry/TetraMesh.cpp
        geometry/TetraMeshFactory.cpp
        geometry/TriangleMesh.h
        geometry/TriangleMesh.cpp
        geometry/TriangleMeshFactory.cpp
        geometry/TriangleMeshDeformation.cpp
        geometry/TriangleMeshSimplification.cpp
        geometry/TriangleMeshSubdivide.cpp
        geometry/HalfEdgeTriangleMesh.h
        geometry/HalfEdgeTriangleMesh.cpp

        geometry/PointCloud.h
        geometry/PointCloud.cpp
        geometry/PointCloudCluster.cpp
        geometry/PointCloudFactory.cpp
        geometry/PointCloudPlanarPatchDetection.cpp
        geometry/PointCloudSegmentation.cpp

        geometry/KDTreeFlann.h
        geometry/KDTreeFlann.cpp
        geometry/KDTreeSearchParam.h

        geometry/VoxelGrid.h
        geometry/VoxelGrid.cpp
        geometry/VoxelGridFactory.cpp

        geometry/Geometry.h
        geometry/Geometry2D.h
        geometry/Geometry3D.h
        geometry/Geometry3D.cpp

        geometry/Line3D.h
        geometry/Line3D.cpp
        geometry/LineSet.h
        geometry/LineSet.cpp
        geometry/LineSetFactory.cpp

        geometry/Octree.h
        geometry/Octree.cpp

        geometry/Qhull.h
        geometry/Qhull.cpp

        geometry/IntersectionTest.h
        geometry/IntersectionTest.cpp
        geometry/Keypoint.h
        geometry/ISSKeypoints.cpp
#        geometry/SurfaceReconstructionAlphaShape.cpp
        geometry/SurfaceReconstructionBallPivoting.cpp
#        geometry/SurfaceReconstructionPoisson.cpp
        geometry/EstimateNormals.cpp
)

set(CAMERA_FILES
        camera/PinholeCameraIntrinsic.h
        camera/PinholeCameraIntrinsic.cpp
        camera/PinholeCameraParameters.h
        camera/PinholeCameraParameters.cpp
        camera/PinholeCameraTrajectory.h
        camera/PinholeCameraTrajectory.cpp
)

set(IO_FILES
        io/file_format/FileJPG.cpp
        io/file_format/FileJSON.cpp
        io/file_format/FileLOG.cpp
        io/file_format/FileOFF.cpp
        io/file_format/FilePNG.cpp
        io/file_format/FilePTS.cpp
        io/file_format/FileSTL.cpp
        io/file_format/FileTUM.cpp
        io/file_format/FileXYZ.cpp
        io/file_format/FileXYZN.cpp
        io/file_format/FileXYZRGB.cpp
        io/FileFormatIO.h
        io/FileFormatIO.cpp
        io/IJsonConvertibleIO.h
        io/IJsonConvertibleIO.cpp
        io/ImageIO.h
        io/ImageIO.cpp
        io/LineSetIO.h
        io/LineSetIO.cpp
        io/OctreeIO.h
        io/OctreeIO.cpp
        io/PinholeCameraTrajectoryIO.h
        io/PinholeCameraTrajectoryIO.cpp
        io/PointCloudIO.h
        io/PointCloudIO.cpp
        io/TriangleMeshIO.h
        io/TriangleMeshIO.cpp
        io/VoxelGridIO.h
        io/VoxelGridIO.cpp
)

set(VISUALIZATION_FILES
        visualization/operations.h
        visualization/renderer.h
        visualization/renderer.cpp
        visualization/windows.h
        visualization/windows.cpp
)

set(PROJECT_FILES
        ${CORE_FILES}
        ${METAL_FILES}
        ${UTILITY_FILES}
        ${DATA_FILES}
        ${TENSOR_FILES}
        ${GEOMETRY_FILES}
        ${CAMERA_FILES}
        ${IO_FILES}
        ${VISUALIZATION_FILES}
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
        OpenSSL::SSL OpenSSL::Crypto
        unofficial::minizip::minizip
        Qhull::qhullcpp
        PoissonRecon::PoissonRecon
        OpenVDB::openvdb blosc_static
        ${VTK_LIBRARIES}
        "-framework Metal"
        "-framework MetalKit"
        "-framework AppKit"
        "-framework Foundation"
        "-framework QuartzCore"
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