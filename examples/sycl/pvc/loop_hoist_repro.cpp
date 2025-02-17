// This reproducer demonstrates an issue with decomposed 2D block load operations.
// The code calls two 1x16 block load operfations within a dynamic-sized loop.
// The input data are such that each thread *should* receive two values equal to
// the sg-local thread ID, and the same + 16
//
// If  IGC_allowDecompose2DBlockFuncs=0, this passes, but if
// IGC_allowDecompose2DBlockFuncs=1, it fails:
//
// clang++ -fsycl -fsycl-targets=intel_gpu_pvc loop_hoist_repro.cpp && ./a.out

#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

using coord_t = vector_t<int, 2>;

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x) inline x { return 0; }
#endif

SYCL_DEVICE_BUILTIN(ushort __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, coord_t coord));

int main(int argc, const char** argv)
{
  sycl::queue q{};

  using T = short;
  constexpr int height = 32;
  constexpr int logical_width = 32;
  constexpr int width = logical_width * sizeof(T);
  constexpr int pitch = width; 
  constexpr int alloc_size = height * logical_width;
  constexpr int thread_count = alloc_size / 2;

  // Allocate & initialize data
  T* dev_T = sycl::malloc_device<T>(alloc_size, q);
  std::vector<T> host_data(alloc_size);
  std::iota(host_data.begin(), host_data.end(), static_cast<T>(0));
  q.memcpy(dev_T, host_data.data(), alloc_size * sizeof(T)).wait();

  int* dev_success = sycl::malloc_device<int>(thread_count, q);

  q.parallel_for(sycl::nd_range<1>{thread_count, 64}, [=](sycl::nd_item<1> item)[[sycl::reqd_sub_group_size(16)]]{

    int dynamic_loop_range = dev_T[256]; // = 256
    std::array<coord_t, 2> coord_arr = {coord_t{0, 0}, coord_t{16, 0}};
    std::array<T, 2> copy_result;

    for (int j = 0; j < dynamic_loop_range; ++j){
      for (int i = 0; i < 2; ++i){
        copy_result[i] = sycl::bit_cast<T>(
            __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
                (long)(dev_T), width - 1, height - 1, pitch - 1, coord_arr[i]));
      }
    }

    int thread_id = item.get_global_linear_id();
    int sg_local_id = thread_id % 16;
    dev_success[thread_id] = copy_result[0] == sg_local_id && copy_result[1] == sg_local_id + 16;
  }).wait();

  std::vector<int> host_success(thread_count);
  q.memcpy(host_success.data(), dev_success, thread_count*sizeof(int)).wait(); 
  bool passed = std::all_of(host_success.begin(), host_success.end(), [](int a){return a;});
  std::cout << "Passed: " << passed << std::endl;
}
