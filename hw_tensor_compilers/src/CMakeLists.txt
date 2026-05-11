# ============ Core library =================================

add_subdirectory("graph")
add_subdirectory("passes")
add_subdirectory("graph_operator_kernels")
add_subdirectory("io")

add_library(tensor_compiler_lib STATIC
    ${GRAPH_SOURCES}
    ${PASSES_SOURCES}
    ${KERNELS_SOURCES}
    ${IO_SOURCES}
    
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(tensor_compiler_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}" # currently is /src/
    "${CMAKE_BINARY_DIR}"         # generated protobuf files
)

# ============ External libraries =============================

target_link_libraries(tensor_compiler_lib PUBLIC 
    onnx_proto
    mlir_deps
)
get_target_property(TC_LIBS tensor_compiler_lib INTERFACE_LINK_LIBRARIES)
message(STATUS "Full link libraries for tensor_compiler_lib: ${TC_LIBS}")
message(STATUS "CHECK: LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")

# ============ Main Application ===============================
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC tensor_compiler_lib)
