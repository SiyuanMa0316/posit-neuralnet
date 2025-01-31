#ifndef CUSTOMMODULE_HPP
#define CUSTOMMODULE_HPP

// General headers
#include <torch/torch.h>

struct FloatNetImpl : torch::nn::Module {
	FloatNetImpl() :
		linear1(784, 32),
		linear2(32, 10)
	{ 
		register_module("linear1", linear1);
		register_module("linear2", linear2);
	}

	torch::Tensor forward(torch::Tensor x) {
		// Flatten data
		x = x.view({-1, 784});

		x = linear1(x);
		x = torch::sigmoid(x);

		x = linear2(x);
		return torch::log_softmax(x, /*dim=*/ 1);
	}

	torch::nn::Linear linear1, linear2;
};
TORCH_MODULE(FloatNet);

#endif /* CUSTOMMODULE_HPP */
