#pragma once
#include "passes/semantics_inferer_pass/semantics_inferer.h"
#include "passes/semantics_inferer_pass/elementwise_binop_inferer.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class MulInferer : public passes::semantics_inferer::ElementwiseBinOperationInferer {};
}