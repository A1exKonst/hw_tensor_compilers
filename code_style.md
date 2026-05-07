# Code style

## Contributing

* Conventional commits:
* `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `wip`.

## Format
* Padding: 1 tab.
* Brackets: Google style.

## Naming
* Classes: `PascalCase`.
* Variables: `snake_case`.
* Functions: `auto do_snake_case(Args args) -> ReturnType;`.
* Constants: `ALL_CAPS`.

## Project stucture
* Always use namespace for symbol declaration.
* A folder usually corresponds to a namespace or nested namespace

## Avoided patterns
* Do not use `using namespace std;`.
* Do not use macros.

## Function arguments
* `const T&` - argument, which is guaranteed "readonly"
* `T&`       - argument, which will be modified in function body
* `T`        - argument. Caller object is guaranteed "readonly";
                         used to create a sinking copy, as in setter-functions

## Header code example
```cpp
#pragma once

#include <std>
#include <std>

#include "internal_dependency.h"
#include "internal_dependency.h"

#include "extern_dependency.h"
#include "extern_dependency.h"



namespace my_space {
	/**
    * Class description.
    */
    class MyClassName { // PascalCase
    public:
        // 1. Constructor and destructor:
        explicit MyClassName(int value);
        ~MyClassName() = default;

        // 2. public methods (snake_case)
        auto do_something() -> void;

        [[nodiscard]] // attributes are on a row above
        auto get_value() const -> int;

    private:
        // 3. private methods (snake_case):
        auto internal_setup() -> void;

        // 4. Class fields (snake_case with _ at the end)
        int value_;
        std::string name_;
    };

	auto do_action() -> void;
};
```


## Source code example
```cpp

#include <std>
#include <std>

#include "extern_dependency.h"
#include "extern_dependency.h"

#include "internal_dependency.h"
#include "internal_dependency.h"



namespace my_space {

    auto MyClassName::do_something() -> void {
    }

    auto MyClassName::get_value() const -> int{
    }

    auto MyClassName::internal_setup() -> void {
    }

	auto do_action() -> void{
    }
};
```