// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "Cifar100Data.hpp"
#include "CifarNet_float.hpp"

// Custom functions
#include "test_float.hpp"

// Namespaces
using namespace sw::unum;


// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_20_float.pt"


int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);
	
	// Setup net and log files
    std::cout << "CIFAR-100 Classification" << std::endl;
    std::cout << "Testing on CPU." << std::endl;
	
	// The batch size for testing.
	size_t const kTestBatchSize = 1024;
	
	//Posit network
    CifarNet_float model_float(100);

	// Load net parameters from file

	torch::load(model_float, NET_LOAD_FILENAME_FLOAT);

	// Load CIFAR-100 testing dataset
	auto test_dataset = Cifar100Data(DATASET_PATH, false, true)
			.map(torch::data::transforms::Normalize<>({0.5071, 0.4867, 0.4408}, {0.2675, 0.2565, 0.2761}))
			.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();

	// Create data loader from testing dataset
	auto test_loader = torch::data::make_data_loader(
							std::move(test_dataset),
							torch::data::DataLoaderOptions().batch_size(kTestBatchSize));

	// Test model
	//Float16
	std::cout << std::endl << "Posit" << std::endl;
	test_float_16_GPU(model_float, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	return 0;
}
