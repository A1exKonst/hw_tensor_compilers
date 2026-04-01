find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

message(STATUS "LLVM found: ${LLVM_FOUND}")
message(STATUS "MLIR found: ${MLIR_FOUND}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")

add_library(mlir_deps INTERFACE)

target_include_directories(mlir_deps INTERFACE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_definitions(mlir_deps INTERFACE ${LLVM_DEFINITIONS})

target_link_libraries(mlir_deps INTERFACE 
    MLIRIR
    MLIRParser
    MLIRSupport
    LLVMSupport
)

if(WIN32)
    target_link_libraries(mlir_deps INTERFACE ws2_32 ntdll psapi version)
endif()
    
message(STATUS "MLIR Dependencies configured. Include dirs: ${MLIR_INCLUDE_DIRS}")