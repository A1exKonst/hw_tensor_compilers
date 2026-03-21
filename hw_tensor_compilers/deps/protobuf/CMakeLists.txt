find_package(Protobuf REQUIRED)

set(PROTO_SRC "onnx.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "${PROTO_SRC}")

# add library "onnx_proto"
add_library(onnx_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})

# includes for users
target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

# IDE groups:
source_group("Generated" FILES ${PROTO_SRCS} ${PROTO_HDRS})
