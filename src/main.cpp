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

        inputs.back()  = (std::vector<Matrix::value_type>){rand_x};
        outputs.back() = (std::vector<Matrix::value_type>){learnFunc(rand_x)};
    }

    network.train(inputs, outputs);
}

void testAndPrint(NeuralNetwork& network, uint32_t count) {
    float sum = 0.0f;
    
    for (uint32_t i = 0; i < count; ++i) {
        float rand_x = randFloat(0, 1);

        Matrix input = Matrix::Vector(1);
        input = (std::vector<Matrix::value_type>){rand_x};

        Matrix res = network.forward(input);

        float expected = learnFunc(rand_x);
        float err = std::abs(expected - res[0]);

        sum += err;
        
        std::cout << "Err: " << err << "\n";
    }

    sum /= static_cast<float>(count);

    std::cout << "\nAvg: " << sum << "\n";
}

int main() {
    srand(time(0));
    using LinearType = Linear<XavierInit, Adam<>>;
    
    NeuralNetwork network(
        LinearType(1, 32),
        ActivationLayer<Softsign<>>(),
        LinearType(32, 16),
        ActivationLayer<Softsign<>>(),
        LinearType(16, 8),
        ActivationLayer<Tanh<>>(),
        LinearType(8, 1)
    );

    network.learning_rate = 0.01f;

    for (uint32_t i = 0; i < 100'000; ++i) {
        trainBatch(network, 2);
    }

    for (uint32_t i = 0; i < 20'000; ++i) {
        trainBatch(network, 4);
    }

    for (uint32_t i = 0; i < 2'000; ++i) {
        trainBatch(network, 16);
    }

    for (uint32_t i = 0; i < 1'000; ++i) {
        trainBatch(network, 64);
    }

    testAndPrint(network, 10'000);
    
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