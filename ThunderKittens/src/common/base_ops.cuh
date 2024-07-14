/**
 * @file
 * @brief Basic operations on generic types.
 */

 #pragma once

 #include <cuda_bf16.h>
 #include <limits>
 #include "base_types.cuh"
 
 namespace kittens {
 
 /**
  * @namespace base_ops
  *
  * @brief A namespace for operations on basic data types.
  */
 namespace base_ops {

 /* ----------  CONST OPS  ---------- */
 
 /**
  * @brief Represents the zero constant operation.
  *
  * This operation returns the zero value of the specified type.
  *
  * @tparam T The data type for which to return the zero value.
  * @return The zero value of type T.
  */
 struct zero {
     template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::zero();      }
 };
 /**
  * @brief Represents the one constant operation.
  *
  * This operation returns the one value of the specified type.
  *
  * @tparam T The data type for which to return the one value.
  * @return The one value of type T.
  */
 struct one {
     template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
 };
 /**
  * @brief Represents the positive infinity constant operation.
  *
  * This operation returns the positive infinity value of the specified type.
  *
  * @tparam T The data type for which to return the positive infinity value.
  * @return The positive infinity value of type T.
  */
 struct pos_infty {
     template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
 };
 /**
  * @brief Represents the negative infinity constant operation.
  *
  * This operation returns the negative infinity value of the specified type.
  *
  * @tparam T The data type for which to return the negative infinity value.
  * @return The negative infinity value of type T.
  */
 struct neg_infty {
     template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
 };
 
 
 /* ----------  UNARY OPS  ---------- */
 
 /**
  * @brief Exponential function operation.
  *
  * This operation calculates the exponential of the input value.
  *
  * @tparam T The data type of the input and output values.
  * @param x[in] The input value.
  * @return The exponential of the input value.
  */
 struct exp {
     template<typename T> static __device__ inline T op(const T &x) { return exp(x); }
 };
 template<> __device__ inline float  exp::op<float> (const float &x ) { return __expf(x);                        }
 template<> __device__ inline float2 exp::op<float2>(const float2 &x) { return float2{__expf(x.x), __expf(x.y)}; }
 template<> __device__ inline bf16   exp::op<bf16>  (const bf16 &x  ) { return hexp(x);                          }
 template<> __device__ inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) { return h2exp(x);                         }
 /**
  * @brief Natural log function operation.
  *
  * This operation calculates the natural logarithm of the input value.
  *
  * @tparam T The data type of the input and output values.
  * @param x[in] The input value.
  * @return The natural logarithm of the input value.
  */
 struct log {
     template<typename T> static __device__ inline T op(const T &x) { return log(x); }
 };
 template<> __device__ inline float  log::op<float> (const float &x ) { return __logf(x);                        }
 template<> __device__ inline float2 log::op<float2>(const float2 &x) { return float2{__logf(x.x), __logf(x.y)}; }
 template<> __device__ inline bf16   log::op<bf16>  (const bf16 &x  ) { return hlog(x);                          }
 template<> __device__ inline bf16_2 log::op<bf16_2>(const bf16_2 &x) { return h2log(x);                         }
 /**
  * @brief Absolute value operation.
  *
  * This operation calculates the absolute value of the input.
  *
  * @tparam T The data type of the input and output values.
  * @param x[in] The input value.
  * @return The absolute value of the input.
  */
 struct abs {
     template<typename T> static __device__ inline T op(const T &x) { return abs(x); }
 };
 template<> __device__ inline float  abs::op<float> (const float &x ) { return fabsf(x);                       }
 template<> __device__ inline float2 abs::op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }
 template<> __device__ inline bf16   abs::op<bf16>  (const bf16 &x  ) { return __habs(x);                      }
 template<> __device__ inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) { return __habs2(x);                     }
 /**
  * @brief Rectified Linear Unit (ReLU) operation.
  *
  * This operation applies the ReLU function to the input, which is the
  * maximum of zero and the input value.
  *
  * @tparam T The data type of the input and output values.
  * @param x[in] The input value.
  * @return The result of ReLU function applied to the input.
  */
 struct relu {
     template<typename T> static __device__ inline T op(const T &x) { return max(x, base_types::constants<T>::zero()); }
 };
 template<> __device__ inline float  relu::op<float> (const float &x ) { return max(x, 0.f);                                  }
 template<> __device__ inline float2 relu::op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)};         }
 template<> __device__ inline bf16   relu::op<bf16>  (const bf16 &x  ) { return __hmax(x, base_types::constants<bf16>::zero());    }
 template<> __device__ inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) { return __hmax2(x, base_types::constants<bf16_2>::zero()); }

 // @Genghan: Add sqrt
 struct sqrt {
    template<typename T> static __device__ inline T op(const T &x) { return sqrt(x); }
};
template<> __device__ inline float  sqrt::op<float> (const float &x ) { return __fsqrt_rn(x);                                  }
template<> __device__ inline float2 sqrt::op<float2>(const float2 &x) { return float2{__fsqrt_rn(x.x), __fsqrt_rn(x.y)};         }
template<> __device__ inline bf16   sqrt::op<bf16>  (const bf16 &x  ) { return hsqrt(x);    }
template<> __device__ inline bf16_2 sqrt::op<bf16_2>(const bf16_2 &x) { return h2sqrt(x); }
// @Xinhao: add half and half_2
template<> __device__ inline half   sqrt::op<half>  (const half &x  ) { return hsqrt(x);    }
template<> __device__ inline half_2 sqrt::op<half_2>(const half_2 &x) { return h2sqrt(x); }


 /**
  * @brief Copy operation.
  *
  * This operation returns the input value unchanged.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The input value.
  * @return The same value as the input.
  */
 struct copy { // for non-compile-time setters.
     template<typename T> static __device__ inline T op(const T &a) { return a; }
 };
 
 
 /* ----------  BINARY OPS  ---------- */
 
 /**
  * @brief Copy2 operation.
  *
  * This operation returns the second input value unchanged.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value (ignored).
  * @param b[in] The second input value.
  * @return The same value as the second input.
  */
 struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return b; }
 };
 /**
  * @brief Sum operation.
  *
  * This operation calculates the sum of two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The sum of the input values.
  */
 struct sum {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return a+b; }
 };
 template<> __device__ inline float2 sum::op<float2>(const float2 &a, const float2 &b) { return float2{a.x+b.x, a.y+b.y}; }
 template<> __device__ inline bf16   sum::op<bf16>  (const bf16   &a, const bf16   &b) { return __hadd(a, b);             }
 template<> __device__ inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hadd2(a, b);            }
 template<> __device__ inline half sum::op<half>(const half &a, const half &b) { return __hadd(a, b);            } // @Genghan
 template<> __device__ inline half_2 sum::op<half_2>(const half_2 &a, const half_2 &b) { return __hadd2(a, b);            }
 /**
  * @brief Subtraction operation.
  *
  * This operation calculates the difference between two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The difference between the input values.
  */
 struct sub {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return a-b; }
 };
 template<> __device__ inline float2 sub::op<float2>(const float2 &a, const float2 &b) { return float2{a.x-b.x, a.y-b.y}; }
 template<> __device__ inline bf16   sub::op<bf16>  (const bf16   &a, const bf16   &b) { return __hsub(a, b);             }
 template<> __device__ inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hsub2(a, b);            }
 template<> __device__ inline half sub::op<half>(const half &a, const half &b) { return __hsub(a, b);            } // @Genghan
 template<> __device__ inline half_2 sub::op<half_2>(const half_2 &a, const half_2 &b) { return __hsub2(a, b);            }
 /**
  * @brief Multiplication operation.
  *
  * This operation calculates the product of two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The product of the input values.
  */
 struct mul {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return a*b; }
 };
 template<> __device__ inline float2 mul::op<float2>(const float2 &a, const float2 &b) { return float2{a.x*b.x, a.y*b.y}; }
 template<> __device__ inline bf16   mul::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmul(a, b);             }
 template<> __device__ inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmul2(a, b);            }
 // @Xinhao add half and half_2
 template<> __device__ inline half mul::op<half>    (const half   &a, const half   &b) { return __hmul(a, b);             }
 template<> __device__ inline half_2 mul::op<half_2>(const half_2 &a, const half_2 &b) { return __hmul2(a, b);            }
 /**
  * @brief Division operation.
  *
  * This operation calculates the quotient of two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The quotient of the input values.
  */
 struct div {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return a/b; }
 };
 template<> __device__ inline float2 div::op<float2>(const float2 &a, const float2 &b) { return float2{a.x/b.x, a.y/b.y}; }
 template<> __device__ inline bf16   div::op<bf16>  (const bf16   &a, const bf16   &b) { return __hdiv(a, b);             }
 template<> __device__ inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __h2div(a, b);            } // this op is a special snowflake
 // @Xinhao add half and half_2
 template<> __device__ inline half div::op<half>    (const half   &a, const half   &b) { return __hdiv(a, b);             }
 template<> __device__ inline half_2 div::op<half_2>(const half_2 &a, const half_2 &b) { return __h2div(a, b);            }
 /**
  * @brief Maximum operation.
  *
  * This operation calculates the maximum of two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The maximum of the input values.
  */
  struct max {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::max(a, b); }
 };
 template<>  __device__ inline float2 max::op<float2>(const float2 &a, const float2 &b) { return float2{::max(a.x, b.x), ::max(a.y, b.y)}; }
 template<>  __device__ inline bf16   max::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmax(a, b);                             }
 template<>  __device__ inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmax2(a, b);                            }
 /**
  * @brief Minimum operation.
  *
  * This operation calculates the minimum of two input values.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @return The minimum of the input values.
  */
 struct min {
     template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::min(a, b); }
 };
 template<>  __device__ inline float2 min::op<float2>(const float2 &a, const float2 &b) { return float2{::min(a.x, b.x), ::min(a.y, b.y)}; }
 template<>  __device__ inline bf16   min::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmin(a, b);                         }
 template<>  __device__ inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmin2(a, b);                        }
 
 
 /* ----------  TERNARY OPS  ---------- */
 
 /**
  * @brief Fused multiply-add operation A * B + C.
  *
  * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The second input value.
  * @param c[in] The third input value to be added.
  * @return The result of the fused multiply-add operation.
  */
 struct fma_AxBtC {
     template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
         return sum::op<T>(mul::op<T>(a, b), c);
     }
 };
 /**
  * @brief Fused multiply-add operation A * C + B.
  *
  * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
  * This is particularly useful for attention mechanisms in neural networks.
  *
  * @tparam T The data type of the input and output values.
  * @param a[in] The first input value.
  * @param b[in] The third input value to be added.
  * @param c[in] The second input value.
  * @return The result of the fused multiply-add operation.
  */
 struct fma_AxCtB { // this is the one needed for attention
     template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
         return sum::op<T>(mul::op<T>(a, c), b);
     }
 };

// @Genghan: Add gelu
struct tanh {
    template<typename T> static __device__ inline T op(const T &x) { return sub(1, div(2, sum(1,exp(mul(2.0,x))))); }
 };
 template<> __device__ inline float tanh::op<float> (const float &x) {
     return 1 - 2 / (2.0 * x + 1);
 }
 template<> __device__ inline float2 tanh::op<float2>(const float2 &x) {
     return float2{tanh::op<float>(x.x), tanh::op<float>(x.y)};
 }
 template<> __device__ inline half tanh::op<half> (const half &x) {
//  2 * F.sigmoid(2 * x) - 1
    half sigmoid_2x = hrcp(
            __hadd(
                __float2half(1.0),
                hexp(__hmul(__float2half(-2.0), x))
             )
     );
     return __hsub(
                __hmul(__float2half(2.0), sigmoid_2x),
                __float2half(1.0)
             );
 }
 template<> __device__ inline half_2 tanh::op<half_2> (const half_2 &x) {
     return half_2{tanh::op<half>(x.x), tanh::op<half>(x.y)};
 }

 struct cubed {
    template<typename T> static __device__ inline T op(const T &x) { return mul(mul(x,x),x); }
 };
 template<> __device__ inline float cubed::op<float> (const float &x) { return x * x * x; }
 template<> __device__ inline float2 cubed::op<float2>(const float2 &x) { return float2{x.x * x.x * x.x, x.y * x.y* x.y}; }
 template<> __device__ inline half cubed::op<half> (const half &x) { return __hmul(x, __hmul(x, x)); }
 template<> __device__ inline half_2 cubed::op<half_2> (const half_2 &x) {
     return half_2{cubed::op<half>(x.x), cubed::op<half>(x.y)};
 }


 struct gelu {
    template<typename T> static __device__ inline T op(const T &x) { return x; }
 };
 template<> __device__ inline float gelu::op<float> (const float &x) {
     return 0.5f * x * (1 + tanh::op<float>(base_types::constants<float>::s2pi() * (x + 0.044715f * cubed::op<float>(x))));
 }
 template<> __device__ inline float2 gelu::op<float2>(const float2 &x) {
     return float2{
         0.5f * x.x * (1 + tanh::op<float>(base_types::constants<float>::s2pi() * (x.x + 0.044715f * cubed::op<float>(x.x)))),
         0.5f * x.y * (1 + tanh::op<float>(base_types::constants<float>::s2pi() * (x.y + 0.044715f * cubed::op<float>(x.y))))
     };
 }

template<> __device__ inline half gelu::op<half> (const half &x) {
//  f = 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    return __hmul(
            __hmul(__float2half(0.5), x),
            __hadd(__float2half(1.0),
                   tanh::op<half>(__hmul(__float2half(0.79788456f),
                                         __hadd(x, __hmul(__float2half(0.044715f),
                                                          cubed::op<half>(x)))
                                  )
                   )
            )
    );

 }

 template<> __device__ inline half_2 gelu::op<half_2> (const half_2 &x) {
     return half_2{gelu::op<half>(x.x), gelu::op<half>(x.y)};
 }


 // @Xinhao: add diff_gelu
 struct diff_gelu {
     template<typename T> static __device__ inline T op(const T &x){
         T tanh_out = tanh::op<T>(0.79788456 * x * (1 + 0.044715 * x * x));
         T ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out);
         return ff;
     }
 };
 template<> __device__ inline float diff_gelu::op<float> (const float &x) {
     float tanh_out = tanh::op<float>(0.79788456f * x * (1.0f + 0.044715f * x * x));
     float ff = 0.5 * x * ((1.0 - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * x * x)) + 0.5 * (1.0 + tanh_out);
     return ff;
 }
 // @Xinhao
 template<> __device__ inline half diff_gelu::op<half> (const half &x) {
     half tanh_input = __hmul(__float2half(0.79788456f), __hmul(x, __hadd(__float2half(1.0f), __hmul(__float2half(0.044715f), __hmul(x, x)))));
     half tanh_out = tanh::op<half>(tanh_input);
     half ff_1 = __hmul(__float2half(0.5f),
                        __hmul(x,
                               __hmul(__hsub(__float2half(1.0f), __hmul(tanh_out, tanh_out)), __hadd(__float2half(0.79788456f),
                                                                                                     __hmul(__float2half(0.1070322243f), __hmul(x, x)))
                                      )
                               )
                        );
     half ff_2 = __hmul(__float2half(0.5f), __hadd(__float2half(1.0f), tanh_out));
     half ff = __hadd(ff_1, ff_2);
     return ff;
 }
 template<> __device__ inline half_2 diff_gelu::op<half_2> (const half_2 &x) {
     return half_2{diff_gelu::op<half>(x.x), diff_gelu::op<half>(x.y)};
 }

 // @Genghan
struct rsqrt {
    template<typename T> static __device__ inline T op(const T &x) { return div(base_types::constants<T>::one(), sqrt(x)); }
};
template<> __device__ inline float  rsqrt::op<float> (const float &x ) { return __frsqrt_rn(x);                                  }
template<> __device__ inline float2 rsqrt::op<float2>(const float2 &x) { return float2{__frsqrt_rn(x.x), __frsqrt_rn(x.y)};         }
template<> __device__ inline bf16   rsqrt::op<bf16>  (const bf16 &x  ) { return hrsqrt(x);    }
template<> __device__ inline bf16_2 rsqrt::op<bf16_2>(const bf16_2 &x) { return h2rsqrt(x); }
// @Xinhao: add half and half_2
template<> __device__ inline half   rsqrt::op<half>  (const half &x  ) { return hrsqrt(x);    }
template<> __device__ inline half_2 rsqrt::op<half_2>(const half_2 &x) { return h2rsqrt(x); }

struct no_op {
    template<typename T> static __device__ inline T op(const T &x) { return x; }
};
 
 } // namespace base_ops
 
 } // namespace kittens
 