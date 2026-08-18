#pragma once
#include <cmath>
#include "Matrix.hpp"

template<float betta_1 = 0.9f, float betta_2 = 0.999f>
struct Adam {
        using value_type = Matrix::value_type;
    
    public:
        Adam(uint32_t _in, uint32_t _out) : in(_in), out(_out) {
            mean_momentum_w     = Matrix::Matrix(_out, _in);
            variance_momentum_w = Matrix::Matrix(_out, _in);

            mean_momentum_b     = Matrix::Vector(_out);
            variance_momentum_b = Matrix::Vector(_out);
        }
        
        void update(Matrix& weights, Matrix& biases, const Matrix& dW, const Matrix& dB, float lr) {
            biases.forEach([&](value_type& b, uint32_t r, uint32_t c){
                value_type g = dB[r];

                // First moment (mean) estimate:
                mean_momentum_b[r] *= betta_1;
                mean_momentum_b[r] += (1.0f - betta_1) * g;

                // Second moment (variance) estimate:
                variance_momentum_b[r] *= betta_2;
                variance_momentum_b[r] += (1.0f - betta_2) * (g * g);

                value_type betta_1_t = 1.0f - std::powf(betta_1, t);
                value_type betta_2_t = 1.0f - std::powf(betta_2, t);

                // Bias correction:
                value_type corrected_mean_momentum_b     = mean_momentum_b[r]     / betta_1_t;
                value_type corrected_variance_momentum_b = variance_momentum_b[r] / betta_2_t;

                // Final weight update:
                b -= corrected_mean_momentum_b / std::sqrtf(corrected_variance_momentum_b + eps) * lr;
            });

            weights.forEach([&](value_type& w, uint32_t r, uint32_t c){
                value_type g = dW[r, c];

                // First moment (mean) estimate:
                mean_momentum_w[r, c] *= betta_1;
                mean_momentum_w[r, c] += (1.0f - betta_1) * g;

                // Second moment (variance) estimate:
                variance_momentum_w[r, c] *= betta_2;
                variance_momentum_w[r, c] += (1.0f - betta_2) * (g * g);

                value_type betta_1_t = 1.0f - std::pow(betta_1, t);
                value_type betta_2_t = 1.0f - std::pow(betta_2, t);

                // Bias correction:
                value_type corrected_mean_momentum_w     = mean_momentum_w[r, c]     / betta_1_t;
                value_type corrected_variance_momentum_w = variance_momentum_w[r, c] / betta_2_t;

                // Final weight update:
                w -= corrected_mean_momentum_w / std::sqrtf(corrected_variance_momentum_w + eps) * lr;
            });

            t++;
        }

    private:
        uint32_t in, out;
        Matrix mean_momentum_w;
        Matrix mean_momentum_b;
        Matrix variance_momentum_w;
        Matrix variance_momentum_b;
        uint32_t t = 1;
        
        static constexpr value_type eps = static_cast<value_type>(1e-6f);
};