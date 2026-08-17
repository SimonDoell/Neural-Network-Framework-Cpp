#include <iostream>
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"


float learnFunc(float x) {
    return std::sinf(x * 3.14159265358979f * 2.0f) * 0.5f + 0.5f;
}

void trainBatch(NeuralNetwork& network, uint32_t batch_size) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;

    for (uint32_t i = 0; i < batch_size; ++i) {
        inputs .push_back(Matrix::Vector(1));
        outputs.push_back(Matrix::Vector(1));

        float rand_x = randFloat(0, 1);

        inputs.back()  = (std::vector<float>){rand_x};
        outputs.back() = (std::vector<float>){learnFunc(rand_x)};
    }

    network.train(inputs, outputs);
}

void testAndPrint(NeuralNetwork& network, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        float rand_x = randFloat(0, 1);

        Matrix input = Matrix::Vector(1);
        input = (std::vector<float>){rand_x};

        Matrix res = network.forward(input);

        float expected = learnFunc(rand_x);
        float err = std::abs(expected - res[0]);

        std::cout << "Err: " << err << "\n";
    }
}

int main() {
    NeuralNetwork network(
        Linear<>(1, 32),
        ActivationLayer<ReLU<>>(),
        Linear<>(32, 16),
        ActivationLayer<ReLU<>>(),
        Linear<>(16, 8),
        ActivationLayer<Tanh<>>(),
        Linear<>(8, 1)
    );

    
    return 0;
}














// Matrix mat = Matrix::Matrix(3, 2);
// Matrix col0 = Matrix::Vector(3);
// Matrix col1 = Matrix::Vector(3);
// col0 = {0, 1, 2};
// col1 = {3, 4, 5};
// mat = {
//     col0,
//     col1
// };
// Matrix col = mat.column(1);
// std::cout << col[0] << " " << col[1] << " " << col[2] << "\n"; 