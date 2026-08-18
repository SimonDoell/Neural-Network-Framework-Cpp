#pragma once
#include <cmath>
#include "BaseLayer.hpp"
#include "Activations.hpp"



template<typename Tp>
concept Activation = requires (float x) {
    { Tp::forward   (x) } -> std::same_as<float>;
    { Tp::derivative(x) } -> std::same_as<float>;
};


template<Activation Actv>
struct ActivationLayer : public Layer {
        using value_type = Matrix::value_type;
    
    public:
        void forward(Matrix& x) override {
            cached_activation = x;
            
            x.forEach([&](value_type& activation){
                activation = Actv::forward(activation);
            });
        }

        void backward(Matrix& gradient, float) override {
            gradient.forEach([&](value_type& g, uint32_t r, uint32_t c){
                g *= Actv::derivative(cached_activation[r, c]);
            });
        }

        std::unique_ptr<Layer> Clone() const override {
            return std::make_unique<ActivationLayer<Actv>>(*this);
        }

    private:
        Matrix cached_activation;
};