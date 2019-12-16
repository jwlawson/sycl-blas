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
 *  @filename tune.hpp
 *
 **************************************************************************/

#ifndef SYCLBLAS_TOOLS_AUTO_TUNER_GET_TIME_FOR_CONFIG_HPP_
#define SYCLBLAS_TOOLS_AUTO_TUNER_GET_TIME_FOR_CONFIG_HPP_

template<typename T>
double get_time_for_config(int cache_size, int item_rows, int item_cols,
                           int wg_rows, int wg_cols, int tile_rows,
                           int tile_cols, bool bank_conflict_a,
                           bool bank_conflict_b, int mem_type,
                           int algorithm, bool transpose_a,
                           bool transpose_b, int m, int k, int n, int batch);

#endif  // SYCLBLAS_TOOLS_AUTO_TUNER_GET_TIME_FOR_CONFIG_HPP_
