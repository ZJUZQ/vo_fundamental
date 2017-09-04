cmake_minimum_required(VERSION 3.0)
project(my_feature)

set (CMAKE_BUILD_TYPE "Release") # Debug, Release, RelWithDebInfo and MinSizeRel
set (CMAKE_CXX_FLAGS "-std=c++11")

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

########################### lib #################################################

# add soruce files
list(APPEND SOURCEFILES
	src/find_features_and_match.cpp
	src/pose_estimation_2d2d.cpp
	src/triangulation.cpp
	src/coordinate_transform.cpp
	src/3d2d_BA_ceres.cpp
	src/3d2d_BA_g2o.cpp
	src/3d3d_SVD.cpp
	src/3d3d_ICP_ceres.cpp
	src/3d3d_ICP_g2o.cpp
	src/direct_sparse_g2o.cpp)

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories ( include
					  "/usr/include/eigen3/" # for eigen
					  ${OpenCV_INCLUDE_DIRS}
					  ${CERES_INCLUDE_DIRS}
					  ${G2O_INCLUDE_DIRS}
					  ${CSPARSE_INCLUDE_DIR})

add_library(zhou SHARED ${SOURCEFILES})
target_link_libraries(zhou ${OpenCV_LIBS} ${CERES_LIBRARIES} ${CSPARSE_LIBRARIES}
					  g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension )


##################################################################################
# test file

add_executable(test_pose_3d3d_ICP_g2o test/test_pose_3d3d_ICP_g2o.cpp)
target_link_libraries(test_pose_3d3d_ICP_g2o zhou)

add_executable(test_useLK test/test_useLK.cpp)
target_link_libraries(test_useLK zhou)

add_executable(test_direct_sparse test/test_direct_sparse.cpp)
target_link_libraries(test_direct_sparse zhou)

add_executable(test_direct_semidense test/test_direct_semidense.cpp)
target_link_libraries(test_direct_semidense zhou)
