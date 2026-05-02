# Tensor Compiler

Тензорный компилятор: преобразование высокоуровневых представлений нейронных сетей ONNX в исполняемые файлы MLIR.

## Building

1. **Установите зависимости:**
   В проекте используется [requirements.txt](deps/requirements.txt).

## Technology stack

* C++ 17
* CMake 3.15
* LLVM[MLIR] Dialects : linalg, arith, tensor
* GTest
* Protobuf : onnx.pb.h (onnx.proto)

## Project structure
* src/main.cpp - точка входа.
* tests/test_main.cpp - точка запуска тестов.
* [code_style.txt](code_style.txt) - code style проекта.


Отчет:
0. Граф хранится в памяти в виде flat graph (nodes + values).
1. В построении графа поддерживаются операции: Conv, ReLu, MatMul, Gemm, Add, Mul.
2. Пайплайн обработки файла:
	2.0. input						->	"file.onnx" 
	2.1. onnx_import					-> graph_engine::Graph
	2.1. passes::SemanticsInferer		-> graph_engine::Graph
	2.2. passes::GraphToMLIRConverter	-> mlir::ModuleOp
3. Поддерживается вывод графа в консоль.
4. Есть тесты gtest : ориентированы на корректный вывод graph_engine::Shape и поддержку пайплайна для файлов папки data.


Недостатки:
1. Поддерживается вывод графа в консоль, но graphviz не поддерживается.
2. mlir_converter.h пока что не поддерживает MatMul и Conv




Описание проекта:
Проект принимает файлы формата file.onnx и разворачивает данные в внутренний граф graph_engine::Graph.
Далее осуществляется проход по графу с помощью passes::SemanticsInferer, который проверяет корректность графа и устанавливает типы и размеры тензоров во внутренних операциях графа.
После этого passes::GraphToMLIRConverter осуществляет генерацию mlir объектов и создает mlir::ModuleOp. Операции графа раскладываются в диалект mlir::linalg (при отсутствии соответствующих - в linalg::GenericOp).

