#include <metal_stdlib>
using namespace metal;

kernel void explode(const device uint8_t *in [[ buffer(0) ]],
                    device uint *width_height [[ buffer(1) ]],
                    device uint8_t *out [[ buffer(2) ]],
                    uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.y + id.x;
      uint8_t count = in[offset];
      count = count % 4;
      if (id.x > 0) {
          count += in[offset - 1] / 4;
      }
      if (id.y > 0) {
          count += in[offset - width] / 4;
      }
      if (id.x < width - 1) {
          count += in[offset + 1] / 4;
      }
      if (id.y < height - 1) {
          count += in[offset + width] / 4;
      }
      out[offset] = count;
  }
}

kernel void explode_and_count(const device uint8_t *in [[ buffer(0) ]],
                              device uint *width_height [[ buffer(1) ]],
                              device uint8_t *out [[ buffer(2) ]],
                              device uint32_t *counts [[ buffer(3) ]],
                              uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.y + id.x;
      uint8_t count = in[offset];
      counts[offset] += count / 4;
      count = count % 4;
      if (id.x > 0) {
          count += in[offset - 1] / 4;
      }
      if (id.y > 0) {
          count += in[offset - width] / 4;
      }
      if (id.x < width - 1) {
          count += in[offset + 1] / 4;
      }
      if (id.y < height - 1) {
          count += in[offset + width] / 4;
      }
      out[offset] = count;
  }
}

kernel void symmetric_explode(const device uint8_t *in [[ buffer(0) ]],
                              device uint *width_height [[ buffer(1) ]],
                              device uint8_t *out [[ buffer(2) ]],
                              uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.y + id.x;
      uint8_t count = in[offset];
      count = count % 4;
      if (id.x > 0) {
          count += in[offset - 1] / 4;
      } else {
          count += in[offset + 1] / 4;
      }
      if (id.y > 0) {
          count += in[offset - width] / 4;
      } else {
          count += in[offset + width] / 4;
      }
      if (id.x < width - 1) {
          count += in[offset + 1] / 4;
      }
      if (id.y < height - 1) {
          count += in[offset + width] / 4;
      }
      out[offset] = count;
  }
}

kernel void make_rgb(const device uint8_t *in [[ buffer(0) ]],
                     device uint *width_height [[ buffer(1) ]],
                     device uint8_t *out [[ buffer(2) ]],
                     uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.x + id.y;
      out[3 * offset] = in[offset] * 107;
      out[3 * offset + 1] = in[offset] * 187;
      out[3 * offset + 2] = in[offset] * 87;
  }
}

kernel void make_count_rgb(const device uint32_t *in [[ buffer(0) ]],
                           device uint *width_height [[ buffer(1) ]],
                           device uint8_t *out [[ buffer(2) ]],
                           uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.x + id.y;
      out[3 * offset] = in[offset] >> 9;
      out[3 * offset + 1] = in[offset] >> 9;
      out[3 * offset + 2] = in[offset] >> 9;
  }
}

kernel void two_times(device uint8_t *inout [[ buffer(0) ]],
                     device uint *width_height [[ buffer(1) ]],
                     uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.x + id.y;
      inout[offset] <<= 1;
  }
}

kernel void two_times_count(device uint32_t *inout [[ buffer(0) ]],
                            device uint *width_height [[ buffer(1) ]],
                            uint2 id [[ thread_position_in_grid ]]) {
  uint height = width_height[0];
  uint width = width_height[1];

  if (id.y < height && id.x < width) {
      uint offset = width * id.x + id.y;
      inout[offset] <<= 1;
  }
}

kernel void max_grains(const device uint8_t *array [[ buffer(0) ]],
                       volatile device atomic_int *result [[ buffer(1) ]],
                       threadgroup int *shared_sums [[ threadgroup(0) ]],
                       uint tid [[ thread_index_in_threadgroup ]],
                       uint id [[ thread_position_in_grid ]],
                       uint blockDim [[ threads_per_threadgroup ]]) {
    
  // Really need to ensure id is in range
  shared_sums[tid] = array[id];
  
  threadgroup_barrier(mem_flags::mem_none);
  
  for (uint s = blockDim >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sums[tid] = max(shared_sums[tid], shared_sums[tid + s]);
    }
    threadgroup_barrier(mem_flags::mem_none);
  }
  
  if (tid == 0) {
    atomic_fetch_max_explicit(result, shared_sums[0], memory_order_relaxed);
  }
}

