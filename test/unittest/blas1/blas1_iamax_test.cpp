#include "blas_test.hpp"
#include "unittest/blas1/blas1_iaminmax_common.hpp"
#include <limits>

template <typename scalar_t>
void run_test(const combination_t combi) {
  using tuple_t = IndexValueTuple<int, scalar_t>;

  int size;
  int incX;
  generation_mode_t mode;
  std::tie(size, incX, mode) = combi;

  // Input vector
  std::vector<scalar_t> x_v(size * incX);
  populate_data<scalar_t>(mode, 0.0, x_v);

  // Output scalar
  std::vector<tuple_t> out_s(1, tuple_t(0, 0.0));

  // Reference implementation
  int out_cpu_s = reference_blas::iamax(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incX));
  ex.get_policy_handler().copy_to_device(x_v.data(), gpu_x_v, size * incX);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<tuple_t>(int(1));
  ex.get_policy_handler().copy_to_device(out_s.data(), gpu_out_s, 1);

  _iamax(ex, size, gpu_x_v, incX, gpu_out_s);
  auto event = ex.get_policy_handler().copy_to_host(gpu_out_s, out_s.data(), 1);
  ex.get_policy_handler().wait();

  // Validate the result
  ASSERT_EQ(out_cpu_s, out_s[0].ind);
  ASSERT_EQ(x_v[out_s[0].ind * incX], out_s[0].val);
  ASSERT_EQ(x_v[out_cpu_s * incX], out_s[0].val);
}

class IamaxFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(IamaxFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(iamax, IamaxFloat, combi);

#if DOUBLE_SUPPORT
class IamaxDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(IamaxDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(iamax, IamaxDouble, combi);
#endif
