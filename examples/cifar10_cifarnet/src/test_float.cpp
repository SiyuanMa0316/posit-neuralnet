// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "Cifar10Data.hpp"
#include "CifarNet_float.hpp"

// Custom functions
#include "test_float.hpp"

// Namespaces
using namespace sw::unum;


// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_20_float.pt"

// Options
#define LOAD true

int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);
	
	// Setup net and log files
    std::cout << "CIFAR-10 Classification" << std::endl;
    std::cout << "Testing on GPU." << std::endl;
	
	// The batch size for testing.
	size_t const kTestBatchSize = 1024;
	
	// Float network
    CifarNet_float model_float(10);

	// Load net parameters from file
	if(LOAD){
		torch::load(model_float, NET_LOAD_FILENAME_FLOAT);
	}
	
	// Load CIFAR-10 testing dataset
	auto test_dataset = Cifar10Data(DATASET_PATH, false)
			.map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.247, 0.243, 0.261}))
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
	test_float_32_GPU(model_float, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	std::cout << std::endl << "Float16 GPU" << std::endl;
	test_float_16_GPU(model_float, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	std::cout << std::endl << "Float8" << std::endl;
	test_int_8_GPU(model_float, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	return 0;
}
