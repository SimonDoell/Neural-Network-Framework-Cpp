#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <type_traits>


template<typename Tp>
struct BaseMatrix {
        using value_type = Tp;
        using size_type  = uint32_t;
        using reference  = Tp&;
        using pointer    = Tp*;
        using const_reference  = const Tp&;
        using const_pointer    = const Tp*;

        struct ColumnView {
            public:
                explicit ColumnView(BaseMatrix& _matrix, size_type column_index)
                : matrix(_matrix), base_index(_matrix.getIndex(0, column_index)) {}

                constexpr reference operator[](size_type row) noexcept {
                    assert(row < matrix.rows);
                    return matrix.values[base_index + row];
                }

                constexpr const_reference operator[](size_type row) const noexcept {
                    assert(row < matrix.rows);
                    return matrix.values[base_index + row];
                }

                constexpr ColumnView& operator=(const BaseMatrix& other) {
                    for (uint32_t i = 0; i < matrix.rows; ++i)
                        (*this)[i] = other[i];

                    return *this;
                }

                constexpr ColumnView& operator+=(const BaseMatrix& other) {
                    for (uint32_t i = 0; i < matrix.rows; ++i)
                        (*this)[i] += other[i];

                    return *this;
                }

                constexpr ColumnView& operator-=(const BaseMatrix& other) {
                    for (uint32_t i = 0; i < matrix.rows; ++i)
                        (*this)[i] -= other[i];

                    return *this;
                }

                constexpr BaseMatrix asVector() const {
                    return (BaseMatrix)(*this);
                }
                
                explicit operator BaseMatrix() const {
                    BaseMatrix result = BaseMatrix::Vector(matrix.rows);
                    
                    for (uint32_t i = 0; i < matrix.rows; ++i)
                        result[i] = (*this)[i];
        
                    return result;
                }

                constexpr size_type size() const {
                    return matrix.rows;
                }

            private:
                BaseMatrix& matrix;
                size_type base_index;
        };

    public:
        explicit BaseMatrix(size_type _rows = 1, size_type _cols = 1)
        : rows(_rows), cols(_cols) {resize(_rows, _cols);}

        BaseMatrix(const BaseMatrix&)            = default;
        BaseMatrix(BaseMatrix&&)                 = default;
        BaseMatrix& operator=(const BaseMatrix&) = default;
        BaseMatrix& operator=(BaseMatrix&&)      = default;
        
        constexpr BaseMatrix& operator=(const std::vector<value_type> &_values) noexcept {
          assert(values.size() == _values.size());
          values = _values;
          return *this;
        }

        constexpr BaseMatrix& operator=(const std::vector<BaseMatrix>& matricis) noexcept {
            assert(matricis.size() == cols);
            for (const BaseMatrix& m : matricis)
                assert(m.rows == rows && m.cols == 1);

            for (uint32_t i = 0; i < rows * cols; ++i)
                values[i] = matricis[i / rows][i % rows];
            
            return *this;
        }

        constexpr BaseMatrix operator+(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] + other.values[i];

            return result;
        }

        constexpr BaseMatrix operator-(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] - other.values[i];

            return result;
        }

        // Component-wise multiplication
        constexpr BaseMatrix operator*(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] * other.values[i];

            return result;
        }

        // Component-wise dividing
        constexpr BaseMatrix operator/(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] / other.values[i];

            return result;
        }

        constexpr BaseMatrix operator*(float factor) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] * factor;

            return result;
        }

        constexpr BaseMatrix operator/(float factor) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result.values[i] = this->values[i] / factor;

            return result;
        }

        constexpr BaseMatrix& operator+=(const BaseMatrix& other) noexcept {
            (*this) = (*this) + other;
            return *this;
        }

        constexpr BaseMatrix& operator-=(const BaseMatrix& other) noexcept {
            (*this) = (*this) - other;
            return *this;
        }

        constexpr BaseMatrix& operator*=(const BaseMatrix& other) noexcept {
            (*this) = (*this) * other;
            return *this;
        }

        constexpr BaseMatrix& operator/=(const BaseMatrix& other) noexcept {
            (*this) = (*this) / other;
            return *this;
        }

        constexpr BaseMatrix& operator*=(float factor) noexcept {
            (*this) = (*this) * factor;
            return *this;
        }

        constexpr BaseMatrix& operator/=(float factor) noexcept {
            (*this) = (*this) / factor;
            return *this;
        }

        constexpr bool isSameShape(const BaseMatrix& other) const noexcept {
            return (rows == other.rows) && (cols == other.cols);
        }

        constexpr BaseMatrix transposed() const noexcept {
            BaseMatrix result = BaseMatrix(cols, rows);

            for (uint32_t i = 0; i < rows; ++i)
                for (uint32_t j = 0; j < cols; ++j)
                    result[j, i] = (*this)[i, j];

            return result;
        }

        constexpr BaseMatrix sumColumns() const noexcept {
            BaseMatrix result = BaseMatrix(rows, 1);

            for (uint32_t i = 0; i < rows * cols; ++i)
                result[i % rows] += values[i];
            
            return result;
        }

        constexpr ColumnView column(size_type column_index) noexcept {
            assert(column_index < cols);
            return ColumnView(*this, column_index);
        }

        constexpr reference       operator[](size_type _row)                       noexcept {return values[getIndex(_row,    0)];}
        constexpr reference       operator[](size_type _row, size_type _col)       noexcept {return values[getIndex(_row, _col)];}
        constexpr const_reference operator[](size_type _row)                 const noexcept {return values[getIndex(_row,    0)];}
        constexpr const_reference operator[](size_type _row, size_type _col) const noexcept {return values[getIndex(_row, _col)];}

        void resize(size_type _rows, size_type _cols) {
            rows = _rows;
            cols = _cols;
            values.assign(rows * cols, static_cast<value_type>(0));
        }

        template<typename F>
        void forEach(F&& f) noexcept {
            static_assert((std::is_invocable_r_v<void, F, float&, uint32_t, uint32_t> || std::is_invocable_r_v<void, F, float&>));

            constexpr bool withoutRowCol = std::is_invocable_r_v<void, F, float&>;

            if constexpr (withoutRowCol) {
                for (float& v : values) f(v);
            } else {
                for (uint32_t i = 0; i < rows; ++i)
                    for (uint32_t j = 0; j < cols; ++j)
                        f(values[getIndex(i, j)], i, j);
            }
        }

        constexpr size_type Rows() const noexcept {return rows;}
        constexpr size_type Cols() const noexcept {return cols;}

        static constexpr BaseMatrix Matrix(size_type _rows, size_type _cols) {
            return BaseMatrix(_rows, _cols);
        }

        static constexpr BaseMatrix Vector(size_type _rows) {
            return BaseMatrix(_rows, 1);
        }

        template<typename Typ>
        friend constexpr BaseMatrix<Typ> matmul(const BaseMatrix<Typ>& a, const BaseMatrix<Typ>& b) noexcept;

    private:
        std::vector<value_type> values;
        size_type rows = 1;
        size_type cols = 1;

        constexpr size_type getIndex(size_type _row, size_type _col) const noexcept {
            assert(_row >= 0 && _row < rows);
            assert(_col >= 0 && _col < cols);
            return _row + _col * rows; // column-major
        }
};



template<typename Tp>
constexpr BaseMatrix<Tp> matmul(const BaseMatrix<Tp>& a, const BaseMatrix<Tp>& b) noexcept {
    assert(a.cols == b.rows);

    using Value = BaseMatrix<Tp>::value_type;
    
    BaseMatrix<Tp> result = BaseMatrix<Tp>::Matrix(a.rows, b.cols);

    for (uint32_t i = 0; i < a.rows; ++i) {
        for (uint32_t k = 0; k < b.cols; ++k) {
            Value sum = static_cast<Value>(0);

            for (uint32_t j = 0; j < a.cols; ++j)
                sum += a[i, j] * b[j, k];

            result[i, k] = sum;
        }
    }

    return result;
}


using Matrix  = BaseMatrix<float>;
using dMatrix = BaseMatrix<double>;