# Tensor Compiler

Тензорный компилятор: преобразование высокоуровневых представлений нейронных сетей ONNX в исполняемые файлы MLIR.

## Building

1. **Установите зависимости:**
   В проекте используется [requirements.txt](hw_tensor_compilers/deps/requirements.txt).

## Technology stack

* C++ 17
* CMake 3.15
* LLVM[MLIR] Dialects : Linalg, Arith, Tensor
* GTest
* Protobuf : onnx.pb.h (onnx.proto)

## Project structure
* src/main.cpp - точка входа.
* tests/test_main.cpp - точка запуска тестов.
* [code_style.md](code_style.md) - code style проекта.

## Functionality

* Проект считывает файлы формата `file.onnx` 
* Разворачивает данные ONNX во внутренний граф `graph_engine::Graph`.
	Граф хранится в памяти в виде flat graph (nodes + values).
* Проверяет корректность графа и определяет типы и размеры тензоров с помощью `passes::SemanticsInfererPass`.
* Генерирует из графа `mlir::ModuleOp` с помощью `passes::MLIRConversionPass`
	Операции графа раскладываются в диалект `mlir::linalg` (при отсутствии соответствующих - в `linalg::GenericOp`).
* Понижает полученное представление в LLVM с помощью `passes::mlir_management`.

## Support and tests
* Поддерживаются операции: Conv, ReLu, MatMul, Gemm, Add, Mul.
* Поддерживается чтение файлов `file.onnx` и вывод графа в консоль
* Тесты gtest: ориентированы на корректный инференс `graph_engine::Shape` и поддержку пайплайна для файлов папки data.
				Понижение в LLVM пока что не поддерживает Conv.

