// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "Cifar10Data.hpp"
#include "CifarNet_float.hpp"
#include "CifarNet_posit.hpp"

// Custom functions
#include "test_posit.hpp"

// Namespaces
using namespace sw::unum;

// Posit configuration
struct Type{
	typedef posit<8, 1> Optimizer;
	typedef posit<8, 1> Forward;
	typedef posit<8, 1> Loss;
	typedef Forward Backward;
	typedef Forward Gradient;
	typedef posit<16, 2> LoadFile;
};

// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_20_float.pt"
#define NET_LOAD_FILENAME_POSIT		"../net/example/model_epoch_0_posit.dat"

// Options
#define LOAD true
#define COPY true

int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);
	
	// Setup net and log files
    std::cout << "CIFAR-10 Classification" << std::endl;
    std::cout << "Testing on CPU." << std::endl;
	std::cout << "Underflow mode: " << UNDERFLOW_MODE << std::endl;
	std::cout << "Quire mode: " << QUIRE_MODE << std::endl;
    std::cout << "PositOptimizer<" << Type::Optimizer::nbits << ", " << Type::Optimizer::es << ">" << std::endl;
    std::cout << "PositForward<" << Type::Forward::nbits << ", " << Type::Forward::es << ">" << std::endl;
    std::cout << "PositLoss<" << Type::Loss::nbits << ", " << Type::Loss::es << ">" << std::endl;
	
	// The batch size for testing.
	size_t const kTestBatchSize = 1024;
	
	// Float and Posit networks
    CifarNet_posit<Type> model_posit(10);
    CifarNet_float model_float(10);

	// Load net parameters from file
	if(LOAD){
		if(COPY)
			torch::load(model_float, NET_LOAD_FILENAME_FLOAT);
		else
			load<Type::LoadFile>(model_posit, NET_LOAD_FILENAME_POSIT);
	}

	// Initialize posit net with the same random parameters as float net
	if(COPY) {
		copy_parameters(model_float->parameters(), model_posit.parameters());
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
	std::cout << std::endl << "Posit" << std::endl;
	test_posit(model_posit, *test_loader, test_dataset_size);

    std::cout << "Finished!\n";

	return 0;
}
