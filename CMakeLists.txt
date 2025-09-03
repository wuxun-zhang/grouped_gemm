cmake_minimum_required(VERSION 3.30)
project(megatron_kernels LANGUAGES CXX)

# CMake
cmake_policy(SET CMP0169 OLD)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

set(CMAKE_CXX_COMPILER ${CXX})
set(CMAKE_C_COMPILER ${CC})

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2025.2))
    set(DPCPP_FLAGS " -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier")
else()
    set(DPCPP_FLAGS "-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
endif()

set(CMAKE_CXX_FLAGS " -fsycl -fsycl-targets=spir64_gen -Xs '-device pvc,bmg' -fno-sycl-instrument-device-code ${DPCPP_FLAGS}")

find_package(Python COMPONENTS Development Interpreter Development.SABIModule REQUIRED)

# find torch package
execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)

# find torch package
find_package(Torch REQUIRED)
set(Torch_INCLUDE_DIRS
    "${Torch_DIR}/../../../include"
    "${Torch_DIR}/../../../include/torch/csrc/api/include"
    "${Torch_DIR}/../../../include/torch/csrc"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find DPCPP
execute_process(
    COMMAND which ${CMAKE_CXX_COMPILER}
    OUTPUT_VARIABLE DPCPP_BINARY_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(DPCPP_INCLUDE_DIR "${DPCPP_BINARY_INSTALL_DIR}/../../include")
message(STATUS "DPCPP_INCLUDE_DIR: ${DPCPP_INCLUDE_DIR}")

# for cutlass only
add_compile_definitions(CUTLASS_ENABLE_SYCL)
add_compile_definitions(SYCL_INTEL_TARGET)
add_compile_definitions(NDEBUG)
add_compile_definitions(CUTLASS_ENABLE_HEADERS_ONLY)

include_directories(
    # for cutlass
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/examples/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${Torch_INCLUDE_DIRS}
    ${DPCPP_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

set(SYCL_TARGET_NAME "sycl_targets")
add_subdirectory(csrc/xpu)

# for torch custom registration
set(HOST_TARGET_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch_extensions.cpp
)

set(TORCH_CUSTOM_EXTENSION_NAME "_C")
Python_add_library(${TORCH_CUSTOM_EXTENSION_NAME} MODULE USE_SABI 3.9 WITH_SOABI
    ${HOST_TARGET_FILES}
    $<TARGET_OBJECTS:${SYCL_TARGET_NAME}>
)
set_target_properties(${TORCH_CUSTOM_EXTENSION_NAME} PROPERTIES
    LINKER_LANGUAGE CXX
)
target_compile_definitions(${TORCH_CUSTOM_EXTENSION_NAME} PRIVATE
    "-DTORCH_CUSTOM_EXTENSION_NAME=${TORCH_CUSTOM_EXTENSION_NAME}")
target_link_libraries(${TORCH_CUSTOM_EXTENSION_NAME}
    PRIVATE c10_xpu torch_xpu
)

# install
install(TARGETS ${TORCH_CUSTOM_EXTENSION_NAME}
    LIBRARY DESTINATION grouped_gemm
)
