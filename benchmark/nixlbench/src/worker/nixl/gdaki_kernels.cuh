/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GDAKI_KERNELS_CUH
#define GDAKI_KERNELS_CUH

#include <iostream>
#include <string_view>
#include "utils/utils.h"
#include <nixl_types.h>

#define MAX_THREADS 1024

nixl_status_t
checkDeviceKernelParams(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        int threads_per_block,
                        int blocks_per_grid);

// Common parameters for device kernel launches
struct deviceKernelParams {
    int num_iterations;
    std::string_view level;
    size_t count;
    const size_t *lens;
    void *const *local_addrs;
    const uint64_t *remote_addrs;
    int threads_per_block = 256;
    int blocks_per_grid = 1;
    cudaStream_t stream = 0;
    uint64_t signal_inc = 0;
    uint64_t remote_addr = 0;
};

// Launch Device kernel for device-side transfer execution (full transfers)
nixl_status_t
launchDeviceKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params);

// Launch Device kernel for partial transfers (supports thread/warp/block coordination)
nixl_status_t
launchDevicePartialKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params);

uint64_t
readNixlGpuSignal(const void *signal_addr, std::string_view gpulevel);

#endif // GDAKI_KERNELS_CUH
