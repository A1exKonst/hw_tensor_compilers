# TODO:
# 0. add command to copy data/

cmake_minimum_required(VERSION 3.15)
project(hw_tensor_compilers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# 1. Поиск Protobuf (должен быть установлен в системе или через vcpkg)
find_package(Protobuf REQUIRED)

# 2. Пути к папкам
set(PROTO_INPUT_DIR "hw_tensor_compilers/deps/protobuf")
set(PROTO_SRC "${PROTO_INPUT_DIR}/onnx.proto")

# 3. Генерация файлов
# Генерирует ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h и onnx.pb.cc
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "${PROTO_SRC}")

set(SOURCES
    "hw_tensor_compilers/include/graph/attributes.hpp"
    "hw_tensor_compilers/src/graph/graph.cpp"
    "hw_tensor_compilers/src/graph/semantics.cpp" 

    "hw_tensor_compilers/src/io/in_graph_onnx.cpp"
    "hw_tensor_compilers/src/io/out_graph_console.cpp"

    "hw_tensor_compilers/src/main.cpp"
)
# "hw_tensor_compilers/src/io/out_graph_graphviz.cpp"

# 4. Создание исполняемого файла
add_executable(${PROJECT_NAME} 
    ${SOURCES}
    ${PROTO_SRCS} 
    ${PROTO_HDRS} 
#    "hw_tensor_compilers/include/graph_2/value.h" 
#    "hw_tensor_compilers/include/graph_2/basic_types.h" 
#    "hw_tensor_compilers/include/graph_2/node.h"
)

# 5. Настройка путей включения (#include)
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/src"
    "${CMAKE_CURRENT_BINARY_DIR}" # Здесь лежат сгенерированные .pb.h
)

# 6. Линковка с библиотеками Protobuf
target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotobuf)

# 7. Специфичные настройки для Visual Studio (группировка файлов)
source_group("Generated" FILES ${PROTO_SRCS} ${PROTO_HDRS})

# add_custom_command(
#    TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#        "${CMAKE_CURRENT_SOURCE_DIR}/data"
#        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
#)
