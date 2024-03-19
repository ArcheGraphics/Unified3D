//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cmath>

namespace u3d::utility {

/// Small vector class with some basic arithmetic operations that can be used
/// within cuda kernels
template <class T, int N>
struct MiniVec {
    typedef T Scalar_t;

    inline MiniVec() = default;

    template <class... TInit>
    inline explicit MiniVec(TInit... as) : arr{as...} {}

    inline explicit MiniVec(const T* const ptr) {
        for (int i = 0; i < N; ++i) operator[](i) = ptr[i];
    }

    inline T operator[](size_t i) const { return arr[i]; }

    inline T& operator[](size_t i) { return arr[i]; }

    template <class T2>
    inline MiniVec<T2, N> cast() const {
        MiniVec<T2, N> a;
        for (int i = 0; i < N; ++i) a[i] = T2(operator[](i));
        return a;
    }

    inline T dot(const MiniVec<T, N>& a) const {
        T result = 0;
        for (int i = 0; i < N; ++i) result += operator[](i) * a[i];
        return result;
    }

    inline MiniVec<T, N> abs() const {
        MiniVec<T, N> r;
        for (int i = 0; i < N; ++i) r[i] = std::abs(operator[](i));
        return r;
    }

    [[nodiscard]] inline bool all() const {
        bool result = true;
        for (int i = 0; i < N && result; ++i) result = result && operator[](i);
        return result;
    }

    [[nodiscard]] inline bool any() const {
        for (int i = 0; i < N; ++i)
            if (operator[](i)) return true;
        return false;
    }

    T arr[N];
};

template <int N>
inline MiniVec<float, N> floor(const MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = floorf(a[i]);
    return r;
}

template <int N>
inline MiniVec<double, N> floor(const MiniVec<double, N>& a) {
    MiniVec<double, N> r;
    for (int i = 0; i < N; ++i) r[i] = std::floor(a[i]);
    return r;
}

template <int N>
inline MiniVec<float, N> ceil(const MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = ceilf(a[i]);
    return r;
}

template <int N>
inline MiniVec<double, N> ceil(const MiniVec<double, N>& a) {
    MiniVec<double, N> r;
    for (int i = 0; i < N; ++i) r[i] = std::ceil(a[i]);
    return r;
}

template <class T, int N>
inline MiniVec<T, N> operator-(const MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = -a[i];
    return r;
}

template <class T, int N>
inline MiniVec<T, N> operator!(const MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = !a[i];
    return r;
}

#define DEFINE_OPERATOR(op, opas)                                         \
    template <class T, int N>                                             \
    inline MiniVec<T, N> operator op(const MiniVec<T, N>& a,              \
                                     const MiniVec<T, N>& b) {            \
        MiniVec<T, N> c;                                                  \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];                  \
        return c;                                                         \
    }                                                                     \
                                                                          \
    template <class T, int N>                                             \
    inline void operator opas(MiniVec<T, N>& a, const MiniVec<T, N>& b) { \
        for (int i = 0; i < N; ++i) a[i] opas b[i];                       \
    }                                                                     \
                                                                          \
    template <class T, int N>                                             \
    inline MiniVec<T, N> operator op(const MiniVec<T, N>& a, T b) {       \
        MiniVec<T, N> c;                                                  \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b;                     \
        return c;                                                         \
    }                                                                     \
                                                                          \
    template <class T, int N>                                             \
    inline MiniVec<T, N> operator op(T a, const MiniVec<T, N>& b) {       \
        MiniVec<T, N> c;                                                  \
        for (int i = 0; i < N; ++i) c[i] = a op b[i];                     \
        return c;                                                         \
    }                                                                     \
                                                                          \
    template <class T, int N>                                             \
    inline void operator opas(MiniVec<T, N>& a, T b) {                    \
        for (int i = 0; i < N; ++i) a[i] opas b;                          \
    }

DEFINE_OPERATOR(+, +=)
DEFINE_OPERATOR(-, -=)
DEFINE_OPERATOR(*, *=)
DEFINE_OPERATOR(/, /=)
#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                            \
    template <class T, int N>                                          \
    inline MiniVec<bool, N> operator op(const MiniVec<T, N>& a,        \
                                        const MiniVec<T, N>& b) {      \
        MiniVec<bool, N> c;                                            \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];               \
        return c;                                                      \
    }                                                                  \
                                                                       \
    template <class T, int N>                                          \
    inline MiniVec<bool, N> operator op(const MiniVec<T, N>& a, T b) { \
        MiniVec<T, N> c;                                               \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b;                  \
        return c;                                                      \
    }                                                                  \
                                                                       \
    template <class T, int N>                                          \
    inline MiniVec<T, N> operator op(T a, const MiniVec<T, N>& b) {    \
        MiniVec<bool, N> c;                                            \
        for (int i = 0; i < N; ++i) c[i] = a op b[i];                  \
        return c;                                                      \
    }

DEFINE_OPERATOR(<)
DEFINE_OPERATOR(<=)
DEFINE_OPERATOR(>)
DEFINE_OPERATOR(>=)
DEFINE_OPERATOR(==)
DEFINE_OPERATOR(!=)
DEFINE_OPERATOR(&&)
DEFINE_OPERATOR(||)
#undef DEFINE_OPERATOR
#undef FN_SPECIFIERS

}  // namespace u3d::utility
