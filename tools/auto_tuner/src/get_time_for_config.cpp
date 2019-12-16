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
 *  @filename get_time_for_config.cpp
 *
 **************************************************************************/


#include "get_time_for_config.hpp"
#include "tune.hpp"
#include "utils.hpp"

#include <limits>

namespace {

template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config>
bool matches_params(int cache_size, int item_rows, int item_cols, int wg_rows,
                    int wg_cols, int tile_rows, int tile_cols,
                    bool bank_conflict_a, bool bank_conflict_b, int mem_type,
                    int algorithm, bool transpose_a, bool transpose_b) {
  bool matches = Cls == cache_size;
  matches = matches && Tile::item_rows == item_rows;
  matches = matches && Tile::item_cols == item_cols;
  matches = matches && Tile::wg_rows == wg_rows;
  matches = matches && Tile::wg_cols == wg_cols;
  matches = matches && Tile::tl_rows == tile_rows;
  matches = matches && Tile::tl_cols == tile_cols;
  matches = matches && Config::TransA == transpose_a;
  matches = matches && Config::TransB == transpose_b;
  matches = matches && Nbca == bank_conflict_a;
  matches = matches && Nbcb == bank_conflict_b;
  matches = matches && static_cast<int>(Config::MemoryMode) == mem_type;
  matches = matches && static_cast<int>(Config::ShapeMode) == algorithm;

  return matches;
}

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
  constexpr int n_reps = 128;

  GemmArgs<DataType> args{m,        n,        k,   alpha, device_a,
                          lda,      device_b, ldb, beta,  host_c,
                          device_c, result_c, ldc, batch, expected_c};
  return tune<Cls, Tile, DoubleBuffer, Nbca, Nbcb, Config, DataType>(n_reps,
                                                                     args);
}

}  // namespace

template <typename T>
double get_time_for_config(int cache_size, int item_rows, int item_cols,
                           int wg_rows, int wg_cols, int tile_rows,
                           int tile_cols, bool bank_conflict_a,
                           bool bank_conflict_b, int mem_type, int algorithm,
                           bool transpose_a, bool transpose_b, int m, int k,
                           int n, int batch) {
#define RETURN_IF_MATCH(TRA, TRB, MEM, ALGO, ...)                             \
  do {                                                                        \
    if (matches_params<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO>>(         \
            cache_size, item_rows, item_cols, wg_rows, wg_cols, tile_rows,    \
            tile_cols, bank_conflict_a, bank_conflict_b, mem_type, algorithm, \
            transpose_a, transpose_b)) {                                      \
      auto result =                                                           \
          run_tune_for_params<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO>,   \
                              T>(m, k, n, batch);                             \
      return result.sec;                                                      \
    }                                                                         \
  } while (0)

#define BENCH_PARAMS(...)                    \
  RETURN_IF_MATCH(true, true, __VA_ARGS__);  \
  RETURN_IF_MATCH(true, false, __VA_ARGS__); \
  RETURN_IF_MATCH(false, true, __VA_ARGS__); \
  RETURN_IF_MATCH(false, false, __VA_ARGS__);

#include "generated_combinations.def"

  return std::numeric_limits<double>::max();

#undef BENCH_PARAMS
}

#define INSTANTIATE_FOR_TYPE(DTYPE)                                           \
  template double get_time_for_config<DTYPE>(                                 \
      int cache_size, int item_rows, int item_cols, int wg_rows, int wg_cols, \
      int tile_rows, int tile_cols, bool bank_conflict_a,                     \
      bool bank_conflict_b, int mem_type, int algorithm, bool transpose_a,    \
      bool transpose_b, int m, int k, int n, int batch)

INSTANTIATE_FOR_TYPE(float);

#undef INSTANTIATE_FOR_TYPE
