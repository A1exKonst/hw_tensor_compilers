## hw_tensor_compilers/include/io/

Данная папка представляет собой реализацию `namespace io` - реализации ввода-вывода создаваемых объектов.

* `graph_importer.h`: 
      Контракт на то, как должен быть реализовано чтение графа `class GraphImporter;`
* `graph_exporter.h`: 
      Контракт на то, как должен быть реализован вывод графа `class GraphExporter;`
* `onnx_importer.h`: 
      Реализация контракта чтения графа из файлов формата ".onnx" `class OnnxImporter;`
* `console_graph_exporter.h`: 
      Реализация контракта вывода графа в консоль `class ConsoleGraphImporter;`