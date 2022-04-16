#ifndef TEST_FLOAT_HPP
#define TEST_FLOAT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>

template <class Model, typename DataLoader>
void test_float(	Model& model,
					DataLoader& data_loader,
					size_t dataset_size	){
	
	torch::NoGradGuard no_grad;
	model->eval();

	float test_loss = 0;
	size_t correct = 0;
	size_t correct_top5 = 0;

	for(const auto& batch : data_loader) {
		// Get data and target
		auto data = batch.data;
		auto target = batch.target;
		
		// Convert data and target to float32 and long
		data = data.to(torch::kF32);
		target = target.to(torch::kLong);

		// Forward pass
		auto output = model->forward(data);

		// Calculate loss
		test_loss += torch::nll_loss(	output,
						 				target,
						 				/*weight=*/{},
						 				torch::Reduction::Sum	).template item<float>();
	
		auto pred = output.argmax(1);
		auto top5 = std::get<1>(output.topk(5, 1, true, true)).t();

		correct += pred.eq(target).sum().template item<int64_t>();
    	correct_top5 += top5.eq(target.expand_as(top5)).sum().template item <int64_t>();
	}

	// Get average loss
	test_loss /= dataset_size;

	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: [%5ld/%5ld] %.4f | Top-5: [%5ld/%5ld] %.4f\n",
	  			test_loss,
				correct, dataset_size, static_cast<float>(correct) / dataset_size,
				correct_top5, dataset_size, static_cast<float>(correct_top5) / dataset_size);
}

template <class Model, typename DataLoader>
void test_float_16_GPU(	Model& model,
					DataLoader& data_loader,
					size_t dataset_size	){
	torch::Device device = torch::kCPU;

	if (torch::cuda::is_available()) {
  	std::cout << "CUDA is available! Testing on GPU." << std::endl;
  	device = torch::kCUDA;
	
	torch::NoGradGuard no_grad;
	model->to(device, torch::kF16);
	model->eval();

	float test_loss = 0;
	size_t correct = 0;
	size_t correct_top5 = 0;

	for(const auto& batch : data_loader) {
		// Get data and target
		auto data = batch.data;
		auto target = batch.target;
		
		// Convert data and target to float32 and long
		data = data.to(device, torch::kF16);
		target = target.to(device, torch::kLong);

		// Forward pass
		auto output = model->forward(data);

		// Calculate loss
		test_loss += torch::nll_loss(	output,
						 				target,
						 				/*weight=*/{},
						 				torch::Reduction::Sum	).template item<float>();
	
		auto pred = output.argmax(1);
		auto top5 = std::get<1>(output.topk(5, 1, true, true)).t();

		correct += pred.eq(target).sum().template item<int64_t>();
    	correct_top5 += top5.eq(target.expand_as(top5)).sum().template item <int64_t>();
	}

	// Get average loss
	test_loss /= dataset_size;

	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: [%5ld/%5ld] %.4f | Top-5: [%5ld/%5ld] %.4f\n",
	  			test_loss,
				correct, dataset_size, static_cast<float>(correct) / dataset_size,
				correct_top5, dataset_size, static_cast<float>(correct_top5) / dataset_size);
	
	}
	else{
		std::cout << "CUDA is NOT available!" << std::endl;
	}
}
#endif /* TEST_FLOAT_HPP */
