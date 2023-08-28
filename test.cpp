#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA..." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    // 加载整个模型
    std::shared_ptr<torch::jit::script::Module> model;
    try {
        model = torch::jit::load("multi_input_model.pt");
        model->to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // 加载示例输入
    std::vector<torch::Tensor> sample_inputs = torch::load<std::vector<torch::Tensor>>("sample_inputs.pt");

    // 为每个示例输入生成一个随机输入
    std::vector<torch::jit::IValue> random_inputs;
    for (const auto& sample_input : sample_inputs) {
        random_inputs.push_back(torch::randn_like(sample_input).to(device));
    }

    // 使用模型
    torch::Tensor output = model->forward(random_inputs).toTensor();

    std::cout << "Output: " << output << std::endl;

    return 0;
}

