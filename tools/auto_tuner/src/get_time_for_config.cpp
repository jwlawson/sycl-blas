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
                    bool double_buffer, bool bank_conflict_a,
                    bool bank_conflict_b, int mem_type, int algorithm,
                    bool transpose_a, bool transpose_b) {
  bool matches = Cls == cache_size;
  matches = matches && Tile::item_rows == item_rows;
  matches = matches && Tile::item_cols == item_cols;
  matches = matches && Tile::wg_rows == wg_rows;
  matches = matches && Tile::wg_cols == wg_cols;
  matches = matches && Tile::tl_rows == tile_rows;
  matches = matches && Tile::tl_cols == tile_cols;
  matches = matches && Config::TransA == transpose_a;
  matches = matches && Config::TransB == transpose_b;
  matches = matches && DoubleBuffer == double_buffer;
  matches = matches && Nbca == bank_conflict_a;
  matches = matches && Nbcb == bank_conflict_b;
  matches = matches && static_cast<int>(Config::MemoryMode) == mem_type;
  matches = matches && static_cast<int>(Config::ShapeMode) == algorithm;

  return matches;
}

}  // namespace

template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config, typename DataType>
TestResultEntry run_tune_for_params(int m, int k, int n, int batch);


template <typename T>
double get_time_for_config(int cache_size, int item_rows, int item_cols,
                           int wg_rows, int wg_cols, int tile_rows,
                           int tile_cols, bool double_buffer,
                           bool bank_conflict_a, bool bank_conflict_b,
                           int mem_type, int algorithm, bool transpose_a,
                           bool transpose_b, int m, int k, int n, int batch) {
  constexpr auto error_val = 100000.0;
#define RETURN_IF_MATCH(TRA, TRB, MEM, ALGO, ...)                           \
  do {                                                                      \
    if (matches_params<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO>>(       \
            cache_size, item_rows, item_cols, wg_rows, wg_cols, tile_rows,  \
            tile_cols, double_buffer, bank_conflict_a, bank_conflict_b,     \
            mem_type, algorithm, transpose_a, transpose_b)) {               \
      auto result =                                                         \
          run_tune_for_params<__VA_ARGS__, GemmConfig<TRA, TRB, MEM, ALGO>, \
                              T>(m, k, n, batch);                           \
      if (result.error < 1.0) {                                             \
        return error_val;                                                   \
      } else {                                                              \
        return result.sec;                                                  \
      }                                                                     \
    }                                                                       \
  } while (0)

#define BENCH_PARAMS(...)                    \
  RETURN_IF_MATCH(true, true, __VA_ARGS__);  \
  RETURN_IF_MATCH(true, false, __VA_ARGS__); \
  RETURN_IF_MATCH(false, true, __VA_ARGS__); \
  RETURN_IF_MATCH(false, false, __VA_ARGS__);

#include "generated_combinations.def"

  return error_val;

#undef BENCH_PARAMS
}

#define INSTANTIATE_FOR_TYPE(DTYPE)                                           \
  template double get_time_for_config<DTYPE>(                                 \
      int cache_size, int item_rows, int item_cols, int wg_rows, int wg_cols, \
      int tile_rows, int tile_cols, bool double_buffer, bool bank_conflict_a, \
      bool bank_conflict_b, int mem_type, int algorithm, bool transpose_a,    \
      bool transpose_b, int m, int k, int n, int batch)

INSTANTIATE_FOR_TYPE(float);

#undef INSTANTIATE_FOR_TYPE
