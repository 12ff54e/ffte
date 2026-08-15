cmake_minimum_required(VERSION 3.16)

project(ffte VERSION 0.1.0 LANGUAGES CXX)

option(FFTE_BUILD_TESTS "Build the native correctness tests" ON)

add_library(ffte STATIC src/ffte.cpp)
add_library(ffte::ffte ALIAS ffte)
target_include_directories(
    ffte
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ffte PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(ffte PRIVATE /W4 /permissive-)
else()
    target_compile_options(ffte PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(EMSCRIPTEN)
    add_executable(ffte_wasm src/ffte.cpp)
    target_include_directories(ffte_wasm PRIVATE include)
    target_compile_features(ffte_wasm PRIVATE cxx_std_17)
    set_target_properties(ffte_wasm PROPERTIES OUTPUT_NAME ffte SUFFIX ".js")
    target_link_options(
        ffte_wasm
        PRIVATE
            "-sWASM=1"
            "-sMODULARIZE=1"
            "-sEXPORT_ES6=1"
            "-sEXPORT_NAME=createFFTE"
            "-sALLOW_MEMORY_GROWTH=1"
            "-sENVIRONMENT=web,worker,node"
            "-sFILESYSTEM=0"
            "-sNO_EXIT_RUNTIME=1"
            "-sEXPORTED_FUNCTIONS=['_ffte_r2c_1d_complex_size','_ffte_r2c_2d_complex_size','_ffte_r2c_1d','_ffte_c2r_1d','_ffte_r2c_2d','_ffte_c2r_2d','_malloc','_free']"
            "-sEXPORTED_RUNTIME_METHODS=['HEAPF64']"
    )
endif()

if(FFTE_BUILD_TESTS AND NOT EMSCRIPTEN)
    enable_testing()
    add_executable(ffte_tests tests/test_ffte.cpp)
    target_link_libraries(ffte_tests PRIVATE ffte::ffte)
    target_compile_features(ffte_tests PRIVATE cxx_std_17)
    add_test(NAME ffte_tests COMMAND ffte_tests)
endif()
