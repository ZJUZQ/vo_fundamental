# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vortex/zhou_temp_test/opencv_test/vo_fundamental

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build

# Include any dependencies generated for this target.
include CMakeFiles/test_direct_sparse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_direct_sparse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_direct_sparse.dir/flags.make

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o: CMakeFiles/test_direct_sparse.dir/flags.make
CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o: ../test/test_direct_sparse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o -c /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/test/test_direct_sparse.cpp

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/test/test_direct_sparse.cpp > CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.i

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/test/test_direct_sparse.cpp -o CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.s

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.requires:

.PHONY : CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.requires

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.provides: CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_direct_sparse.dir/build.make CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.provides.build
.PHONY : CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.provides

CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.provides.build: CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o


# Object files for target test_direct_sparse
test_direct_sparse_OBJECTS = \
"CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o"

# External object files for target test_direct_sparse
test_direct_sparse_EXTERNAL_OBJECTS =

../bin/test_direct_sparse: CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o
../bin/test_direct_sparse: CMakeFiles/test_direct_sparse.dir/build.make
../bin/test_direct_sparse: ../lib/libzhou.so
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
../bin/test_direct_sparse: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
../bin/test_direct_sparse: /usr/local/lib/libceres.a
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libgflags.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libtbb.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/test_direct_sparse: /usr/lib/liblapack.so
../bin/test_direct_sparse: /usr/lib/libf77blas.so
../bin/test_direct_sparse: /usr/lib/libatlas.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/librt.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libtbb.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/test_direct_sparse: /usr/lib/liblapack.so
../bin/test_direct_sparse: /usr/lib/libf77blas.so
../bin/test_direct_sparse: /usr/lib/libatlas.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/librt.so
../bin/test_direct_sparse: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/test_direct_sparse: CMakeFiles/test_direct_sparse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test_direct_sparse"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_direct_sparse.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_direct_sparse.dir/build: ../bin/test_direct_sparse

.PHONY : CMakeFiles/test_direct_sparse.dir/build

CMakeFiles/test_direct_sparse.dir/requires: CMakeFiles/test_direct_sparse.dir/test/test_direct_sparse.cpp.o.requires

.PHONY : CMakeFiles/test_direct_sparse.dir/requires

CMakeFiles/test_direct_sparse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_direct_sparse.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_direct_sparse.dir/clean

CMakeFiles/test_direct_sparse.dir/depend:
	cd /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vortex/zhou_temp_test/opencv_test/vo_fundamental /home/vortex/zhou_temp_test/opencv_test/vo_fundamental /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build /home/vortex/zhou_temp_test/opencv_test/vo_fundamental/build/CMakeFiles/test_direct_sparse.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_direct_sparse.dir/depend

