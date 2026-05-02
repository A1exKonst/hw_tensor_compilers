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

set(llvm_components
    core
    support
    native          # automatic choice of target
    nativecodegen
)

llvm_map_components_to_libnames(llvm_libs ${llvm_components})

add_library(mlir_deps INTERFACE)

target_include_directories(mlir_deps INTERFACE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_definitions(mlir_deps INTERFACE ${LLVM_DEFINITIONS})

target_link_libraries(mlir_deps INTERFACE
    ${llvm_libs}
    MLIRBufferizationTransforms 
    MLIRBufferizationToMemRef
    MLIRIR
    MLIRFuncToLLVM
    MLIRFuncTransforms
    MLIRLinalgToStandard
    MLIRLinalgTransforms
    MLIRTensorToLinalg
    MLIRMemRefToSPIRV
    MLIRReconcileUnrealizedCasts
    #MLIRConvertFuncToLLVM
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRTensorTilingInterfaceImpl
    MLIRBufferizationPipelines
    MLIRExecutionEngine
    #MLIRCRunnerUtils

    # Dialects:
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSPIRVDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRBufferizationDialect
    #MLIRConversionPasses 

    # Core:
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRSupport
    LLVMSupport
)

if(WIN32)
    target_link_libraries(mlir_deps INTERFACE ws2_32 ntdll psapi version)
endif()
    
message(STATUS "MLIR Dependencies configured. Include dirs: ${MLIR_INCLUDE_DIRS}")