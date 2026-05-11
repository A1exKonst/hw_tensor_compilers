#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class AddInferer : public passes::semantics_inferer::ElementwiseBinOperationInferer {
	};

}