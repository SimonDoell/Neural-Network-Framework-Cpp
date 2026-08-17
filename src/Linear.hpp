#pragma once
#include "Matrix.hpp"
#include "BaseLayer.hpp"
#include "Initilizations.hpp"

float randFloat(float min, float max) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min) + min;
}

template<typename InitFunc = XavierInit>
struct Linear : public Layer {
        using size_type = uint32_t;
    
    public:
        Linear(size_type _in, size_type _out) {
            weights = Matrix::Matrix(_out, _in);
            biases  = Matrix::Vector(_out);

            float range = InitFunc::range(_in, _out);

            weights.forEach([&](float& w){
                w = randFloat(-range, range);
            });
        }

        void forward(Matrix& x) override {
            cached_activations = x;

            // y = W * x + b
            x = matmul(weights, x);

            for (uint32_t c = 0; c < x.Cols(); ++c)
                x.column(c) += biases;
        }

        void backward(Matrix& gradient, float learning_rate) override {
            Matrix prev_grad = matmul(weights.transposed(), gradient);

            Matrix dW = matmul(gradient, cached_activations.transposed());
            Matrix dB = gradient.sumColumns();

            dW /= gradient.Cols();
            dB /= gradient.Cols();

            // SGD
            weights -= dW * learning_rate;
            biases  -= dB * learning_rate;

            gradient = std::move(prev_grad);
        }

        std::unique_ptr<Layer> Clone() const override {
            return std::make_unique<Linear>(*this);
        }
        
    private:
        Matrix weights;
        Matrix biases;
        Matrix cached_activations;
};