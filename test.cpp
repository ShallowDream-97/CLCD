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

// ... [其他代码保持不变]

// 从文本文件读取Tensor形状
std::vector<std::vector<int64_t>> shapes;
std::ifstream shape_file("sample_input_shapes.txt");
std::string line;
while (std::getline(shape_file, line)) {
    std::vector<int64_t> shape;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoi(item));
    }
    shapes.push_back(shape);
}

// 根据读取的形状生成随机输入
std::vector<torch::jit::IValue> random_inputs;
for (const auto& shape : shapes) {
    random_inputs.push_back(torch::randn(shape).to(device));
}

// 使用模型
torch::Tensor output = model->forward(random_inputs).toTensor();

std::cout << "Output: " << output << std::endl;

return 0;

# ... [其他代码保持不变]

# 生成示例输入并存储
sample_inputs = [torch.randn(1, 10), torch.randn(5, 5), torch.randn(5, 10)]

with open("sample_input_shapes.txt", "w") as file:
    for tensor in sample_inputs:
        shape_str = ",".join(map(str, tensor.shape))
        file.write(shape_str + "\n")
