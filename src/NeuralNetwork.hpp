#pragma once
#include <memory>
#include <vector>
#include "BaseLayer.hpp"
#include "Linear.hpp"
#include "ActivationLayer.hpp"

struct NeuralNetwork {
    public:
        float learning_rate = 0.01f;

        template<typename... LayerTypes>
        NeuralNetwork(LayerTypes... _layers) {
            (layers.emplace_back(std::make_unique<LayerTypes>(std::move(_layers))), ...);
        }

        Matrix forward(Matrix activation) {
            for (uint32_t i = 0; i < layers.size(); ++i) {
                layers[i]->forward(activation);
            }

            return activation;
        }

        void train(const Matrix& activations, Matrix desired_activations) {
            Matrix result = forward(activations);

            for (uint32_t c = 0; c < result.Cols(); ++c) {
                result.column(c) -= desired_activations.column(c).asVector();
            }

            for (int32_t l = layers.size()-1; l >= 0; --l) {
                layers[l]->backward(result, learning_rate);
            }
        }

        void train(const std::vector<Matrix>& activations, const std::vector<Matrix>& desired_activations) {
            assert(activations.size() == desired_activations.size());
            uint32_t rows = activations[0].Rows();

            for (uint32_t i = 0; i < activations.size(); ++i) {
                assert(activations[i]        .Rows() == rows && activations[i]        .Cols() == 1);
                assert(desired_activations[i].Rows() == rows && desired_activations[i].Cols() == 1);
            }
            
            Matrix activation = Matrix::Matrix(rows, activations.size());
            activation = activations;

            Matrix desired_activation = Matrix::Matrix(rows, activations.size());
            desired_activation = desired_activations;

            train(activation, desired_activation);
        }

    private:
        std::vector<std::unique_ptr<Layer>> layers;
};






// Idea:
// - Two different modes of forward propagation: training and using
//     - Training: for example store the activation in the layer for backprop
//     - Using: using in real world application WITHOUT training -> no need to update the activation
// -> which "mode" is used is determined by the forward function:
//     -> const forward function is "Using"
//     -> non-const function is "Training"
// 
// ToDo:
// - Add Adam optimizer
// - more activation functions