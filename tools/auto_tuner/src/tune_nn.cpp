/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename tune_nn.cpp
 *
 **************************************************************************/

#include "gemm_tuner.hpp"
#include <CL/sycl.hpp>
#include <cstdlib>

#include "csv.h"

int sleep_for_millis = 0;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " csv_filename [num_reps] [sleep]"
              << std::endl;
    return -1;
  }

  char const* const csv_file = argv[1];

  constexpr int seed = 42;
  int num_reps = 16;
  if (argc > 2) {
    num_reps = std::atoi(argv[2]);
  }
  if (argc > 3) {
    sleep_for_millis = std::atoi(argv[3]);
  }

  io::CSVReader<6> reader(csv_file);
  reader.read_header(io::ignore_extra_column, "TransposeLHS", "TransposeRHS",
                     "M", "N", "K", "batch");
  char* trans_lhs_str;
  char* trans_rhs_str;
  int m;
  int n;
  int k;
  int batch;
  while (reader.read_row(trans_lhs_str, trans_rhs_str, m, n, k, batch)) {
    bool transpose_lhs = trans_lhs_str[0] == 't';
    bool transpose_rhs = trans_rhs_str[0] == 't';
    if (!transpose_lhs and !transpose_rhs) {
      run_tune_gemm<false, false, float>(seed, m, k, n, batch, num_reps);
    }
  }
  return 0;
}
