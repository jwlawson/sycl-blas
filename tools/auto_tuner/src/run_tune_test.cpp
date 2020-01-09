/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename run_tune_test.cpp
 *
 **************************************************************************/

#include "get_time_for_config.hpp"
#include "tune.hpp"
#include "utils.hpp"

#include <limits>


template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config, typename DataType>
TestResultEntry run_tune_for_params(int m, int k, int n, int batch) {
  constexpr int seed = 42;
  std::mt19937 rnd(seed);
  auto host_a = get_random_vector<DataType>(k * m * batch, -1, 1, rnd);
  auto host_b = get_random_vector<DataType>(n * k * batch, -1, 1, rnd);
  auto host_c = get_random_vector<DataType>(m * n * batch, -1, 1, rnd);
  auto expected_c = host_c;
  auto result_c = host_c;

  const auto device_a = blas::make_sycl_iterator_buffer(host_a, host_a.size());
  const auto device_b = blas::make_sycl_iterator_buffer(host_b, host_b.size());
  auto device_c = blas::make_sycl_iterator_buffer(host_c, host_c.size());

  const int lda = Config::TransA ? k : m;
  const int ldb = Config::TransB ? n : k;
  const int ldc = m;

  constexpr DataType alpha = 1;
  constexpr DataType beta = 1;
  constexpr int n_reps = 32;

  GemmArgs<DataType> args{m,        n,        k,   alpha, device_a,
                          lda,      device_b, ldb, beta,  host_c,
                          device_c, result_c, ldc, batch, expected_c};
  return tune<Cls, Tile, DoubleBuffer, Nbca, Nbcb, Config, DataType>(n_reps,
                                                                     args);
}


#define RETURN_IF_MATCH(TRA, TRB, MEM, ALGO, ...)                       \
  template TestResultEntry                                              \
  run_tune_for_params<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO>, float>( \
      int m, int k, int n, int batch);

#define BENCH_PARAMS(...)                    \
  RETURN_IF_MATCH(true, true, __VA_ARGS__);  \
  RETURN_IF_MATCH(true, false, __VA_ARGS__); \
  RETURN_IF_MATCH(false, true, __VA_ARGS__); \
  RETURN_IF_MATCH(false, false, __VA_ARGS__);

#include "generated_combinations.def"

#undef BENCH_PARAMS
