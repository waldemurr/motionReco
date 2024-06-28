cmake_minimum_required(VERSION 3.18)
project(MotionReco LANGUAGES CXX VERSION 1.0.1 DESCRIPTION "people motion capture")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ Path Settings ------------------
set(MAIN_PATH "${CMAKE_SOURCE_DIR}") # ------- Makes source dir
list(APPEND CMAKE_PREFIX_PATH "${MAIN_PATH}/libtorch") # torch needs to be added manually

include(GoogleTest)
include(CTest)
include(FetchContent)
find_package( OpenCV REQUIRED ) 
find_package(fmt REQUIRED) 
find_package(Torch REQUIRED)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  if(MSVC)
    add_compile_options(/MD$<$<CONFIG:Debug>:d>)
  endif()   # MSVC
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()     # NOT googletest_POPULATED

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(package_add_test_with_libraries TESTNAME FILES LIBRARIES TEST_WORKING_DIRECTORY)
    add_executable(${TESTNAME} ${FILES})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${LIBRARIES})
    gtest_discover_tests(${TESTNAME}
        WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB VISION_SRC
     "resources/resnet50/*"
     "resources/darknet/*"
     "resources/yolo/*"
     "core/*"
     "*.cpp"
     "detectors/*"
     "utils/*"
     "main.cpp"

)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable( vision ${VISION_SRC} )

target_link_libraries(vision ${Boost_LIBRARIES} ${OpenCV_LIBS} fmt::fmt stdc++fs "${TORCH_LIBRARIES}")
message("LIBS ${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET vision
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:vision>)
endif (MSVC)
