#ifndef TRAIN_POSIT_HPP
#define TRAIN_POSIT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

template <typename T, template<typename> class Model, typename DataLoader, typename Optimizer>
void train_posit(	size_t epoch,
					size_t const num_epochs,
					Model<T>& model,
					DataLoader& data_loader,
					Optimizer& optimizer,
					size_t const kLogInterval,
					size_t const dataset_size	){

	using Target = unsigned char;

	model.train();
	size_t batch_idx = 0;
	size_t total_batch_size = 0;

	for(auto const& batch : data_loader) {
		// Update number of trained samples
		size_t const batch_size = batch.target.size(0);
		total_batch_size += batch_size;

		// Get data and target
		auto data_float = batch.data;
		auto target_float = batch.target;

		// Convert data and target to float32 and uint8
		data_float = data_float.to(torch::kF32);
		target_float = target_float.to(torch::kUInt8);

		// Convert data and target from PyTorch Tensor to StdTensor
		auto data = Tensor_to_StdTensor<float, typename T::Forward>(data_float);
		auto target = Tensor_to_StdTensor<uint8_t, Target>(target_float);
		
#ifndef USING_HL_THREADS
		// Forward pass
		auto output = model.forward(data);
		cross_entropy_loss<typename T::Loss, typename T::Loss, Target> loss(output, target);

		// Backward pass and optimize
		optimizer.zero_grad();
		loss.backward(model);
		optimizer.step();

#else
		// Forward and backward pass
		std::vector<float> losses;
		std::vector<std::vector<StdTensor<typename T::Optimizer>>> gradients;
		forward_backward<cross_entropy_loss<typename T::Loss, typename T::Loss, Target>>(
							model, data, target, losses, gradients);

		// Sum gradients from other threads
		sum_gradients(model.parameters(), gradients);
	
		// Optimizer step
		optimizer.step();
#endif /* USING_HL_THREADS */

		// Print progress
		if (++batch_idx % kLogInterval == 0) {
#ifndef USING_HL_THREADS
			float loss_value = loss.template item<float>();
#else
			float loss_value = std::accumulate(losses.begin(), losses.end(), 0.0) / batch_size;
#endif /* USING_HL_THREADS */

			std::printf("Train Epoch: %.3f/%2ld Data: %5ld/%5ld Loss: %.4f\n",
					epoch-1+static_cast<float>(total_batch_size)/dataset_size,
					num_epochs,
					total_batch_size,
					dataset_size,
					loss_value);
		}
	}
}

#endif /* TRAIN_POSIT_HPP */
