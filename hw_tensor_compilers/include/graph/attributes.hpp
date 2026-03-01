#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>

namespace graph_engine {
    // int64_t is stated in ONNX standard
    using AttributeValue = std::variant<
        int64_t, 
        float, 
        std::string, 
        std::vector<int64_t>, 
        std::vector<float>
    >;

    using Attributes = std::unordered_map<std::string, AttributeValue>;
};



/*
struct Attributes {
    std::map<std::string, AttributeValue> data;

    
     // @brief ”станавливает или обновл€ет значение атрибута.
    template <typename T>
    void set(const std::string& name, T&& value) {
        // value передаетс€ по универсальной ссылке дл€ эффективности
        data[name] = std::forward<T>(value);
    }

    
    // @brief ѕолучает значение атрибута с проверкой типа.
    template <typename T>
    const T& get(const std::string& name) const {
        auto it = data.find(name);
        if (it == data.end()) {
            throw std::runtime_error("Attribute '" + name + "' not found.");
        }

        try {
            return std::get<T>(it->second);
        }
        catch (const std::bad_variant_access&) {
            throw std::runtime_error("Type mismatch for attribute '" + name + "'.");
        }
    }

    
    // @brief ћетод с возвратом значени€ по умолчанию (удобно дл€ ONNX).
    template <typename T>
    T get_or(const std::string& name, T default_value) const {
        auto it = data.find(name);
        if (it == data.end()) return default_value;

        const T* val = std::get_if<T>(&it->second);
        return val ? *val : default_value;
    }
};
*/