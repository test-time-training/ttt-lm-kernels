/**
 * @file
 * @brief Map operations: between tiles, and those which apply vectors to tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
template<typename op, ducks::rt::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}
/**
 * @brief Applies a binary operation to each element of a tile with an unpacked scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Unpacked scalar parameter for the binary operation.
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // The optimizing compiler should eliminate this pack in the 32-bit case but not in the 16-bit case
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}
/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

/* ----------  Row tile maps  ----------*/

/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::col_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::col_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}


// Three-operand row map. Mostly useful for FMA instructions.

/**
 * @brief Applies an operation across the rows of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::col_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_top_row);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of two tiles in a column-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::col_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], row_values[i][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}

/* ----------  Col major tile maps  ----------*/

/**
 * @brief Applies an operation across the columns of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::row_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::row_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x); //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}

// Three-operand col map
/**
 * @brief Applies an operation across the columns of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::row_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], col_values[j][0]);
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of two tiles in a column-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::inner_dim == rt_base<typename T::dtype, typename T::layout>::row_vec_pack); // compatible layout
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x); //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_left_col);
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<ducks::rt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the natural logarithm function on.
 */
template<ducks::rt::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the absolute value function on.
 */
template<ducks::rt::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the ReLU function on.
 */
template<ducks::rt::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to copy from.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}

// @Genghan: Add sqrt
template<ducks::rt::all T>
__device__ static inline void sqrt(T &dst, const T &src) {
    unary_map<base_ops::sqrt, T>(dst, src);
}

// @Genghan: Add gelu
template<ducks::rt::float_like T>
__device__ static inline void gelu(T &dst, const T &src) {
    unary_map<base_ops::gelu, T>(dst, src);
}

// @Xinhao: add diff_gelu
template<ducks::rt::float_like T>
__device__ static inline void diff_gelu(T &dst, const T &src) {
    unary_map<base_ops::diff_gelu, T>(dst, src);
}

template<ducks::rt::all T>
__device__ static inline void rsqrt(T &dst, const T &src) {
    unary_map<base_ops::rsqrt, T>(dst, src);
}

template<ducks::rt::all T>
__device__ static inline void no_op(T &dst, const T &src) {
    unary_map<base_ops::no_op, T>(dst, src);
}

/**
 * @brief Applies the max operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Applies the min operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile or scalar for the addition.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile or scalar for the subtraction.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template<ducks::rt::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

}