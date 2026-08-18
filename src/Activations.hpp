#pragma once
#include <cmath>


struct ReLU {
    static constexpr float forward   (float x) noexcept {return std::max(x, 0.0f);}
    static constexpr float derivative(float x) noexcept {return (x >= 0.0f ? 1.0f : 0.0f);}
};

template<float relaxation = 1.15f>
struct Tanh {
    static constexpr float forward   (float x) noexcept {return std::tanhf(x) * relaxation;}
    static constexpr float derivative(float x) noexcept {float t = std::tanhf(x); return (1.0f - t * t) * relaxation;}
};

struct Sigmoid {
    static constexpr float forward(float x)    {return 1.0f / (1 + std::expf(-x));}
    static constexpr float derivative(float x) {float sig = Sigmoid::forward(x); return sig * (1.0f - sig);}
};

template<float leakage = 0.15f>
struct leakyReLU {
    static constexpr float forward(float x) {return std::max(x, x * leakage);}
    static constexpr float derivative(float x) {return (x >= 0.0f ? 1.0f : leakage);}
};

struct Swish {
    static constexpr float forward(float x)    {return x / (1.0f + std::expf(-x));}
    static constexpr float derivative(float x) {float s = 1.0f / (1.0f + std::expf(-x)); return s + x * s * (1.0f - s);}
};

template<float relaxation = 1.15f>
struct Softplus {
    static constexpr float forward(float x)    {return std::log1pf(std::expf(x)) * relaxation;}
    static constexpr float derivative(float x) {return Sigmoid::forward(x) * relaxation;}
};

template<float relaxation = 1.15f>
struct Softsign {
    static constexpr float forward(float x)    {return (x / (1.0f + std::abs(x))) * relaxation;}
    static constexpr float derivative(float x) {return (1.0f / std::powf(1.0f + std::abs(x), 2)) * relaxation;}
};