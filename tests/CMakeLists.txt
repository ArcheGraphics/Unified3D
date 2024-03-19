#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(cpp-tests LANGUAGES C CXX)

set(TEST_FILES
        test_utility/Compare.h
        test_utility/Compare.cpp
        test_utility/Rand.h
        test_utility/Rand.cpp
        test_utility/Raw.h
        test_utility/Raw.cpp
        test_utility/Sort.h
        test_utility/Sort.cpp
)

set(CORE_FILES
        core/CoreTest.h
        core/CoreTest.cpp
        core/Tensor.cpp
)

set(SRC
        ${TEST_FILES}
        ${CORE_FILES}
        Tests.h
        Tests.cpp
        Main.cpp
)

add_executable(${PROJECT_NAME} ${SRC})

enable_testing()

find_package(GTest CONFIG REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Unified3D
        GTest::gtest_main GTest::gmock
)
