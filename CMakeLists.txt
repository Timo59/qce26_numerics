cmake_minimum_required(VERSION 3.27)

project(QCE26_NUMERICS
  VERSION     0.1.0
  DESCRIPTION "Numerical experiments conducted in submissions to the IEEE Quantum Week 2026 (QCE26)"
  LANGUAGES   C
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC
)

########################################################################################################################
# NLopt — vendored as git submodule under extern/nlopt
########################################################################################################################
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/nlopt/CMakeLists.txt")
    message(FATAL_ERROR
        "NLopt submodule not initialised.\n"
        "Run: git submodule update --init --recursive")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NLOPT_CXX         OFF CACHE BOOL "" FORCE)
set(NLOPT_PYTHON      OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE      OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB      OFF CACHE BOOL "" FORCE)
set(NLOPT_GUILE       OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG        OFF CACHE BOOL "" FORCE)
set(NLOPT_TESTS       OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/nlopt EXCLUDE_FROM_ALL)

########################################################################################################################
# Orkan (qsim) — vendored as git submodule under extern/orkan
########################################################################################################################
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/orkan/CMakeLists.txt")
    message(FATAL_ERROR
        "Orkan submodule not initialised.\n"
        "Run: git submodule update --init --recursive")
endif()

add_subdirectory(extern/orkan EXCLUDE_FROM_ALL)

########################################################################################################################
# Optimisation wrapper (vendored from Tal)
########################################################################################################################
add_library(opt OBJECT
    src/opt_run.c
    src/opt_nlopt.c
    src/oracle_ctx.c
)
target_include_directories(opt PUBLIC  ${CMAKE_SOURCE_DIR}/include)
target_include_directories(opt PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(opt PUBLIC nlopt)

########################################################################################################################
# Experiment executables
########################################################################################################################
foreach(strategy IN ITEMS adjacency bubbles binsert)
    add_executable(test_${strategy} src/${strategy}.c)
    target_link_libraries(test_${strategy} PRIVATE opt orkan m)

    foreach(ex IN ITEMS 1 2 3)
        add_executable(${strategy}_ex${ex} src/${strategy}_9ex${ex}.c)
        target_link_libraries(${strategy}_ex${ex} PRIVATE opt orkan m)
    endforeach()
endforeach()
