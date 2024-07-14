#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace vec {
namespace global_to_register {

struct vec_load_store {
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_loadstore";
    template<int S, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        kittens::col_vec<kittens::rt_bf<S, S, L>> reg_vec;
        kittens::load(reg_vec, input);
        kittens::store(output, reg_vec);
    }
};

void tests(test_data &results);

}
}
}
}

#endif