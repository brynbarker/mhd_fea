# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.19.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.19.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2

# Utility rule file for release.

# Include the progress variables for this target.
include CMakeFiles/release.dir/progress.make

CMakeFiles/release:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Switching CMAKE_BUILD_TYPE to Release"
	/usr/local/Cellar/cmake/3.19.4/bin/cmake -DCMAKE_BUILD_TYPE=Release /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2
	/usr/local/Cellar/cmake/3.19.4/bin/cmake -E echo "***"
	/usr/local/Cellar/cmake/3.19.4/bin/cmake -E echo "*** Switched to Release mode. Now recompile with:  \$$ make"
	/usr/local/Cellar/cmake/3.19.4/bin/cmake -E echo "***"

release: CMakeFiles/release
release: CMakeFiles/release.dir/build.make

.PHONY : release

# Rule to build all files generated by this target.
CMakeFiles/release.dir/build: release

.PHONY : CMakeFiles/release.dir/build

CMakeFiles/release.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/release.dir/cmake_clean.cmake
.PHONY : CMakeFiles/release.dir/clean

CMakeFiles/release.dir/depend:
	cd /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2 /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2 /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2 /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2 /Users/bryn/BRYN/School/UNC/FiniteElement/project/mhd_fea/code/take2/CMakeFiles/release.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/release.dir/depend

