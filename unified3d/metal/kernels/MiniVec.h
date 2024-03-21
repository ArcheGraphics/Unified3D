//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace u3d {
namespace metal {
/// Small vector class with some basic arithmetic operations that can be used
/// within cuda kernels
template <class T, int N>
struct MiniVec {
    typedef T Scalar_t;
    
    inline MiniVec() {}
    
    template <class... TInit>
    inline explicit MiniVec(TInit... as) : arr{as...} {}
    
    inline explicit MiniVec(const device T* const ptr) {
        for (int i = 0; i < N; ++i) operator[](i) = ptr[i];
    }
    
    inline const T operator[](size_t i) const { return arr[i]; }
    
    inline device T& operator[](size_t i) { return arr[i]; }
    
    template <class T2>
    inline MiniVec<T2, N> cast() const {
        MiniVec<T2, N> a;
        for (int i = 0; i < N; ++i) a[i] = T2(operator[](i));
        return a;
    }
    
    inline T dot(const device MiniVec<T, N>& a) const {
        T result = 0;
        for (int i = 0; i < N; ++i) result += operator[](i) * a[i];
        return result;
    }
    
    inline MiniVec<T, N> abs() const {
        MiniVec<T, N> r;
        for (int i = 0; i < N; ++i) r[i] = ::abs(operator[](i));
        return r;
    }
    
    inline bool all() const {
        bool result = true;
        for (int i = 0; i < N && result; ++i) result = result && operator[](i);
        return result;
    }
    
    inline bool any() const {
        for (int i = 0; i < N; ++i)
            if (operator[](i)) return true;
        return false;
    }
    
    T arr[N];
};

template <int N>
inline MiniVec<float, N> floor(const device MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = floorf(a[i]);
    return r;
}

template <int N>
inline MiniVec<float, N> ceil(const device MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = ceilf(a[i]);
    return r;
}

template <class T, int N>
inline MiniVec<T, N> operator-(const device MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = -a[i];
    return r;
}

template <class T, int N>
inline MiniVec<T, N> operator!(const device MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = !a[i];
    return r;
}

#define DEFINE_OPERATOR(op, opas)                                          \
template <class T, int N>                                              \
inline MiniVec<T, N> operator op(const device MiniVec<T, N>& a,        \
const device MiniVec<T, N>& b) {      \
MiniVec<T, N> c;                                                   \
for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];                   \
return c;                                                          \
}                                                                      \
\
template <class T, int N>                                              \
inline void operator opas(device MiniVec<T, N>& a,                     \
const device MiniVec<T, N>& b) {             \
for (int i = 0; i < N; ++i) a[i] opas b[i];                        \
}                                                                      \
\
template <class T, int N>                                              \
inline MiniVec<T, N> operator op(const device MiniVec<T, N>& a, T b) { \
MiniVec<T, N> c;                                                   \
for (int i = 0; i < N; ++i) c[i] = a[i] op b;                      \
return c;                                                          \
}                                                                      \
\
template <class T, int N>                                              \
inline MiniVec<T, N> operator op(T a, device const MiniVec<T, N>& b) { \
MiniVec<T, N> c;                                                   \
for (int i = 0; i < N; ++i) c[i] = a op b[i];                      \
return c;                                                          \
}                                                                      \
\
template <class T, int N>                                              \
inline void operator opas(device MiniVec<T, N>& a, T b) {              \
for (int i = 0; i < N; ++i) a[i] opas b;                           \
}

DEFINE_OPERATOR(+, +=)
DEFINE_OPERATOR(-, -=)
DEFINE_OPERATOR(*, *=)
DEFINE_OPERATOR(/, /=)
#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                   \
template <class T, int N>                                                 \
inline MiniVec<bool, N> operator op(const device MiniVec<T, N>& a,        \
const device MiniVec<T, N>& b) {      \
MiniVec<bool, N> c;                                                   \
for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];                      \
return c;                                                             \
}                                                                         \
\
template <class T, int N>                                                 \
inline MiniVec<bool, N> operator op(const device MiniVec<T, N>& a, T b) { \
MiniVec<T, N> c;                                                      \
for (int i = 0; i < N; ++i) c[i] = a[i] op b;                         \
return c;                                                             \
}                                                                         \
\
template <class T, int N>                                                 \
inline MiniVec<T, N> operator op(T a, const device MiniVec<T, N>& b) {    \
MiniVec<bool, N> c;                                                   \
for (int i = 0; i < N; ++i) c[i] = a op b[i];                         \
return c;                                                             \
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

}
}
