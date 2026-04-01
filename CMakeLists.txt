cmake_minimum_required(VERSION 3.15)
project(hw_tensor_compilers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# # folder /bin for all executables and files, used by executables:
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============ Dependencies ==================================
add_subdirectory(hw_tensor_compilers/deps/protobuf)
add_subdirectory(hw_tensor_compilers/deps/llvm_mlir)

# ============ Create core lib =================
# create tensor_compiler_lib, and add executable(PROJECT_NAME)
add_subdirectory(hw_tensor_compilers/src) 

# ============ Tests =========================================
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(hw_tensor_compilers/tests)
endif()

# Copy input data/files.onnx:
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data" 
            DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
