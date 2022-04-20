// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "LeNet5_float.hpp"
#include "LeNet5_posit.hpp"
#include "test_float.hpp"


// Namespaces
using namespace sw::unum;


// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_10_float.pt"


int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);
	
	// Setup net and log files
    std::cout << "Fashion MNIST Classification" << std::endl;

	// The batch size for testing.
	size_t const kTestBatchSize = 1024;
	
	// Float and Posit networks
	LeNet5_float model_float;

	// Load net parameters from file
	torch::load(model_float, NET_LOAD_FILENAME_FLOAT);


	
	// Load Fashion MNIST testing dataset
	auto test_dataset = torch::data::datasets::MNIST(DATASET_PATH,
							torch::data::datasets::MNIST::Mode::kTest)
							.map(torch::data::transforms::Normalize<>(0.2860, 0.3300))
                    		.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();

	// Create data loader from testing dataset
	auto test_loader = torch::data::make_data_loader(
							std::move(test_dataset),
							torch::data::DataLoaderOptions().batch_size(kTestBatchSize));

	// Test model
	//Posit
	std::cout << std::endl << "Float32" << std::endl;
	test_float(model_float, *test_loader, test_dataset_size);

	std::cout << std::endl << "Float32 GPU" << std::endl;
	test_float32_gpu(model_float, *test_loader, test_dataset_size);

	std::cout << std::endl << "Float16 GPU" << std::endl;
	test_float16_gpu(model_float, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	return 0;
}
