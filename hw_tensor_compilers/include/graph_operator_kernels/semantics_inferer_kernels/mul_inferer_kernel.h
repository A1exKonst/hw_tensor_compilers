#pragma once
#include "passes/semantics_inferer_pass/semantics_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer_kernel.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class MulInfererKernel : public passes::semantics_inferer::ElementwiseBinOperationInfererKernel {
	};

}