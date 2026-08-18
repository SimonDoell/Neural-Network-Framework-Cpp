#pragma once
#include "Matrix.hpp"
#include "BaseLayer.hpp"
#include "Initilizations.hpp"
#include "Adam.hpp"

float randFloat(float min, float max) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min) + min;
}

template<typename Tp>
concept OptimizerType = requires (Tp optimizer, Matrix& w, Matrix& b, const Matrix& dW, const Matrix& dB, float lr) {
    { optimizer.update(w, b, dW, dB, lr) } -> std::same_as<void>;
};

struct SGD {
    public:
        SGD(uint32_t _in, uint32_t _out) {}
        
        void update(Matrix& weights, Matrix& biases, const Matrix& dW, const Matrix& dB, float lr) {
            weights -= dW * lr;
            biases  -= dB * lr;
        }
};

template<typename InitFunc = XavierInit, OptimizerType Optimizer = SGD>
struct Linear : public Layer {
        using size_type  = uint32_t;
        using value_type = Matrix::value_type;
    
    public:
        Linear(size_type _in, size_type _out)
        : optimizer(_in, _out) {
            weights = Matrix::Matrix(_out, _in);
            biases  = Matrix::Vector(_out);

            float range = InitFunc::range(_in, _out);

            weights.forEach([&](value_type& w){
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
            
            optimizer.update(weights, biases, dW, dB, learning_rate);

            gradient = std::move(prev_grad);
        }

        std::unique_ptr<Layer> Clone() const override {
            return std::make_unique<Linear>(*this);
        }
        
    private:
        Optimizer optimizer;
        Matrix weights;
        Matrix biases;
        Matrix cached_activations;
};