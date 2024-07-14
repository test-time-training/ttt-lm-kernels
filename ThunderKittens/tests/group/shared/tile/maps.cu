#include "maps.cuh"

#ifdef TEST_GROUP_SHARED_TILE_MAPS

struct test_exp {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<H%NW==0 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_exp";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = __bfloat162float(__float2bfloat16(::expf(i_ref[i]))); // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();
        G::load(shared_tile, input, W*16);
        __syncthreads();
        G::exp(shared_tile, shared_tile);
        __syncthreads();
        G::store(output, shared_tile, W*16);
    }
};

void group::shared::tile::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d<test_exp, SIZE, SIZE, 2>::run(results);

    if constexpr (TEST_INTENSITY > 1) {

        sweep_st_layout_size_2d<test_exp, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_st_layout_size_2d<test_exp, 12, 5, 12>::run(results);

        }

    }
}

#endif