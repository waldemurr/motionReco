cmake_minimum_required(VERSION 3.15)

project(MotionReco VERSION 1.0.1 DESCRIPTION "people motion capture")

include(GoogleTest)
include(CTest)
include(FetchContent)

# change by your contensts
# set(OpenCV_DIR "C:\\Program Files\\opencv\\build")
# set( OpenCV_FOUND 1 )
FIND_PACKAGE( OpenCV REQUIRED )   


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

add_executable( name ${NAME_SRC} ${NAME_HEADERS} )


set( VISION_SRC
    src/main.cpp    
)

# set( NAME_HEADERS
#     include/header.h
# )

add_executable( vision ${NAME_SRC})