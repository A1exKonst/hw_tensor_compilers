#pragma once
#include <iostream>
#include <unordered_map>
#include <string>

#include "graph/graph.h"

std::ostream& operator<< (std::ostream& out, const graph_engine::AttributeValue& attr);

std::ostream& operator<< (std::ostream& out, const graph_engine::Attributes& attrs);

std::ostream& operator<< (std::ostream& out, graph_engine::DataType dt);

std::ostream& operator<< (std::ostream& out, const graph_engine::Shape& shape);

std::ostream& operator<< (std::ostream& out, graph_engine::OperatorType op);

std::ostream& operator<< (std::ostream& out, const graph_engine::Value& value);

std::ostream& operator<< (std::ostream& out, const graph_engine::Node& node);

std::ostream& operator<< (std::ostream& out, const graph_engine::Graph& graph);
