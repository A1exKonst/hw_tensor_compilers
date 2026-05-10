## hw_tensor_compilers/include/passes/

Данная папка представляет собой реализацию `namespace passes` - проходов по вычислительному графу нейронной сети.

* `graph_pass.h`: 
      Контракт на то, как должен быть реализован проход по графу `class GraphPass;`
* `passes_pipeline.h`: 
      Создание и управление пайплайном проходов по графу `class PassesPipeline;`
* `pipeline_endpoint.h`:
      Предоставляет механизм выбора точки остановки пайплайна через `enum class PipelineEndpoint`
* `mlir_management/...`: 
      Управление и взаимодействие проекта с библиотекой `mlir`
* `mlir_conversion_pass/...`: 
      Проход по графу. Реализует генерацию модуля `mlir::ModuleOp`, соответствующего данному графу.
* `semantics_inferer_pass/...`: 
      Проход по графу. Реализует вывод типов тензоров графа. 