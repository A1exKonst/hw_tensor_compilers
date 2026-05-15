#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer_kernel.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class AddInfererKernel : public passes::semantics_inferer::ElementwiseBinOperationInfererKernel {
	};

}