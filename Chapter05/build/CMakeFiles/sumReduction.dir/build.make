# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build

# Include any dependencies generated for this target.
include CMakeFiles/sumReduction.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sumReduction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sumReduction.dir/flags.make

CMakeFiles/sumReduction.dir/sumReduction.cu.o: CMakeFiles/sumReduction.dir/flags.make
CMakeFiles/sumReduction.dir/sumReduction.cu.o: ../sumReduction.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sumReduction.dir/sumReduction.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/sumReduction.cu -o CMakeFiles/sumReduction.dir/sumReduction.cu.o

CMakeFiles/sumReduction.dir/sumReduction.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sumReduction.dir/sumReduction.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sumReduction.dir/sumReduction.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sumReduction.dir/sumReduction.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sumReduction
sumReduction_OBJECTS = \
"CMakeFiles/sumReduction.dir/sumReduction.cu.o"

# External object files for target sumReduction
sumReduction_EXTERNAL_OBJECTS =

CMakeFiles/sumReduction.dir/cmake_device_link.o: CMakeFiles/sumReduction.dir/sumReduction.cu.o
CMakeFiles/sumReduction.dir/cmake_device_link.o: CMakeFiles/sumReduction.dir/build.make
CMakeFiles/sumReduction.dir/cmake_device_link.o: CMakeFiles/sumReduction.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/sumReduction.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sumReduction.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sumReduction.dir/build: CMakeFiles/sumReduction.dir/cmake_device_link.o

.PHONY : CMakeFiles/sumReduction.dir/build

# Object files for target sumReduction
sumReduction_OBJECTS = \
"CMakeFiles/sumReduction.dir/sumReduction.cu.o"

# External object files for target sumReduction
sumReduction_EXTERNAL_OBJECTS =

sumReduction: CMakeFiles/sumReduction.dir/sumReduction.cu.o
sumReduction: CMakeFiles/sumReduction.dir/build.make
sumReduction: CMakeFiles/sumReduction.dir/cmake_device_link.o
sumReduction: CMakeFiles/sumReduction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable sumReduction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sumReduction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sumReduction.dir/build: sumReduction

.PHONY : CMakeFiles/sumReduction.dir/build

CMakeFiles/sumReduction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sumReduction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sumReduction.dir/clean

CMakeFiles/sumReduction.dir/depend:
	cd /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5 /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5 /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build /home/guanrenyang/Programming-Massively-Parallel-Processors/Chapter5/build/CMakeFiles/sumReduction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sumReduction.dir/depend

