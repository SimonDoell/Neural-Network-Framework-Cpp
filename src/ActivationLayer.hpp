#pragma once
#include <cmath>
#include "BaseLayer.hpp"

template<float leakage = 0.15f>
struct ReLU {
    static constexpr float forward   (float x) noexcept {return std::max(x, x * leakage);}
    static constexpr float derivative(float x) noexcept {return (x >= 0.0f ? 1.0f : leakage);}
};

template<float relaxation = 1.15f>
struct Tanh {
    static constexpr float forward   (float x) noexcept {return std::tanhf(x) * relaxation;}
    static constexpr float derivative(float x) noexcept {float t = std::tanhf(x); return (1.0f - t * t) * relaxation;}
};



template<typename Tp>
concept Activation = requires (float x) {
    { Tp::forward   (x) } -> std::same_as<float>;
    { Tp::derivative(x) } -> std::same_as<float>;
};


template<Activation Actv>
struct ActivationLayer : public Layer {
    public:
        void forward(Matrix& x) override {
            cached_activation = x;
            
            x.forEach([&](float& activation){
                activation = Actv::forward(activation);
            });
        }

        void backward(Matrix& gradient, float) override {
            gradient.forEach([&](float& g, uint32_t r, uint32_t c){
                g *= Actv::derivative(cached_activation[r, c]);
            });
        }

        std::unique_ptr<Layer> Clone() const override {
            return std::make_unique<ActivationLayer<Actv>>(*this);
        }

    private:
        Matrix cached_activation;
};