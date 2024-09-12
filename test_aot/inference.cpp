#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

int main() {
    c10::InferenceMode mode;

    torch::inductor::AOTIModelContainerRunnerCuda runner("/root/reproduce_gnn_results/test_aot/build/model.so");
    
    const int n_nodes = 10;
    const int n_node_features = 12;
    const int n_edges = 20;
    const int n_edge_features = 6;

    std::vector<torch::Tensor> inputs = {
      torch::randn({n_nodes, n_node_features}, at::kCUDA).to(torch::kFloat32),
      torch::randint(0, n_nodes, {2,n_edges}, at::kCUDA).to(torch::kInt64),
      torch::randn({n_edges, n_edge_features}, at::kCUDA).to(torch::kFloat32)
    };

    std::cout << "node_features:\n" << inputs[0] << std::endl;
    std::cout << "edge_index:\n" << inputs[1] << std::endl;
    std::cout << "edge_features:\n" << inputs[2] << std::endl;

    std::vector<torch::Tensor> outputs = runner.run(inputs);
    std::cout << "Result from the first inference:"<< std::endl;
    std::cout << outputs[0] << std::endl;

    return 0;
}
