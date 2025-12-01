/**
 * High-Performance N-Body Gravity Simulation - Full Version v3
 *
 * GPU0: Pb1 (devices mass=0, compute min_dist) - cooperative kernel
 * GPU1: Pb2 + Pb3 (devices with time-varying mass) - per-step kernel launch
 *
 * Architecture:
 * - Pb1: Single cooperative kernel launch for all 200k steps (occupies GPU0)
 * - Pb2: Per-step kernel launches, checkpoint every 5000 steps
 * - Pb3: Per-step kernel launches in background streams (can run parallel with Pb2)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <climits>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>

namespace cg = cooperative_groups;

// ============================================================================
// Constants
// ============================================================================
namespace param
{
    constexpr int n_steps = 200000;
    constexpr double dt = 60.0;
    constexpr double eps = 1e-3;
    constexpr double eps_sq = eps * eps;
    constexpr double G = 6.674e-11;
    constexpr double planet_radius = 1e7;
    constexpr double missile_speed = 1e6;
    constexpr int checkpoint_interval = 5000;

    __host__ __device__ double gravity_device_mass(double m0, double t)
    {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
    }

    __host__ __device__ double get_missile_cost(double t)
    {
        return 1e5 + 1e3 * t;
    }
}

// ============================================================================
// HIP Error Checking
// ============================================================================
#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = call;                                              \
        if (err != hipSuccess)                                              \
        {                                                                   \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================================
// Warp-level reduction (AMD wavefront = 64)
// ============================================================================
__device__ __forceinline__ double warp_reduce_sum(double val)
{
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        val += __shfl_down(val, offset, 64);
    }
    return val;
}

// ============================================================================
// Pb1 Kernel - devices mass = 0, compute min_dist only
// Runs on GPU0, cooperative kernel for full simulation
// ============================================================================
__global__ void __launch_bounds__(256) pb1_kernel(
    double *__restrict__ qx, double *__restrict__ qy, double *__restrict__ qz,
    double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
    const double *__restrict__ mass,
    double *__restrict__ ax_global, double *__restrict__ ay_global, double *__restrict__ az_global,
    int n, int planet_id, int asteroid_id,
    double *__restrict__ min_dist_out,
    int n_steps)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;

    __shared__ double s_qx[256], s_qy[256], s_qz[256], s_m[256];
    __shared__ double s_reduce[4];

    if (bid == 0 && tid == 0)
    {
        *min_dist_out = 1e100;
    }
    grid.sync();

    // Initial distance (step 0)
    if (bid == 0 && tid == 0)
    {
        double dx = qx[planet_id] - qx[asteroid_id];
        double dy = qy[planet_id] - qy[asteroid_id];
        double dz = qz[planet_id] - qz[asteroid_id];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < *min_dist_out)
            *min_dist_out = dist;
    }
    grid.sync();

    for (int step = 1; step <= n_steps; step++)
    {
        // Phase 1: Compute accelerations
        for (int body_i = bid; body_i < n; body_i += grid_size)
        {
            double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
            double qi_x = qx[body_i], qi_y = qy[body_i], qi_z = qz[body_i];

            int num_tiles = (n + block_size - 1) / block_size;
            for (int tile = 0; tile < num_tiles; tile++)
            {
                int j = tile * block_size + tid;
                if (j < n)
                {
                    s_qx[tid] = qx[j];
                    s_qy[tid] = qy[j];
                    s_qz[tid] = qz[j];
                    s_m[tid] = mass[j];
                }
                else
                {
                    s_qx[tid] = s_qy[tid] = s_qz[tid] = s_m[tid] = 0.0;
                }
                __syncthreads();

                int global_j = tile * block_size + tid;
                if (global_j < n && global_j != body_i)
                {
                    double mj = s_m[tid];
                    double dx = s_qx[tid] - qi_x;
                    double dy = s_qy[tid] - qi_y;
                    double dz = s_qz[tid] - qi_z;

                    double dist_sq = dx * dx + dy * dy + dz * dz + param::eps_sq;
                    double dist3 = pow(dist_sq, 1.5);
                    double force_factor = param::G * mj / dist3;

                    acc_x += force_factor * dx;
                    acc_y += force_factor * dy;
                    acc_z += force_factor * dz;
                }
                __syncthreads();
            }

            // Block reduction
            int warp_id = tid / 64, lane_id = tid % 64;
            acc_x = warp_reduce_sum(acc_x);
            acc_y = warp_reduce_sum(acc_y);
            acc_z = warp_reduce_sum(acc_z);

            if (lane_id == 0)
                s_reduce[warp_id] = acc_x;
            __syncthreads();
            if (tid == 0)
                ax_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_y;
            __syncthreads();
            if (tid == 0)
                ay_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_z;
            __syncthreads();
            if (tid == 0)
                az_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
            __syncthreads();
        }
        grid.sync();

        // Phase 2: Update velocities and positions
        for (int i = bid * block_size + tid; i < n; i += grid_size * block_size)
        {
            double ax = ax_global[i], ay = ay_global[i], az = az_global[i];
            vx[i] = __fma_rn(ax, param::dt, vx[i]);
            vy[i] = __fma_rn(ay, param::dt, vy[i]);
            vz[i] = __fma_rn(az, param::dt, vz[i]);
            qx[i] = __fma_rn(vx[i], param::dt, qx[i]);
            qy[i] = __fma_rn(vy[i], param::dt, qy[i]);
            qz[i] = __fma_rn(vz[i], param::dt, qz[i]);
        }
        grid.sync();

        // Phase 3: Check distance
        if (bid == 0 && tid == 0)
        {
            double dx = qx[planet_id] - qx[asteroid_id];
            double dy = qy[planet_id] - qy[asteroid_id];
            double dz = qz[planet_id] - qz[asteroid_id];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < *min_dist_out)
                *min_dist_out = dist;
        }
        grid.sync();
    }
}

// ============================================================================
// Pb2 Cooperative Kernel - devices with time-varying mass, detect collision
// Also records intercept steps for each device
// ============================================================================
__global__ void __launch_bounds__(256) pb2_kernel(
    double *__restrict__ qx, double *__restrict__ qy, double *__restrict__ qz,
    double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
    const double *__restrict__ mass0,
    const int *__restrict__ body_type,
    double *__restrict__ ax_global, double *__restrict__ ay_global, double *__restrict__ az_global,
    int n, int planet_id, int asteroid_id,
    const int *__restrict__ device_ids, int num_devices,
    int *__restrict__ collision_step_out,
    int *__restrict__ intercept_steps, // One per device
    int n_steps)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;

    __shared__ double s_qx[256], s_qy[256], s_qz[256], s_m[256];
    __shared__ double s_reduce[4];

    // Initialize outputs
    if (bid == 0 && tid == 0)
    {
        *collision_step_out = INT_MAX;
    }
    // Initialize intercept steps
    for (int d = bid * block_size + tid; d < num_devices; d += grid_size * block_size)
    {
        intercept_steps[d] = INT_MAX;
    }
    grid.sync();

    for (int step = 1; step <= n_steps; step++)
    {
        double t = step * param::dt;
        double missile_traveled = t * param::missile_speed;

        // Phase 1: Compute accelerations with time-varying mass for devices
        for (int body_i = bid; body_i < n; body_i += grid_size)
        {
            double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
            double qi_x = qx[body_i], qi_y = qy[body_i], qi_z = qz[body_i];

            int num_tiles = (n + block_size - 1) / block_size;
            for (int tile = 0; tile < num_tiles; tile++)
            {
                int j = tile * block_size + tid;
                if (j < n)
                {
                    s_qx[tid] = qx[j];
                    s_qy[tid] = qy[j];
                    s_qz[tid] = qz[j];
                    double m0 = mass0[j];
                    s_m[tid] = (body_type[j] == 1) ? param::gravity_device_mass(m0, t) : m0;
                }
                else
                {
                    s_qx[tid] = s_qy[tid] = s_qz[tid] = s_m[tid] = 0.0;
                }
                __syncthreads();

                int global_j = tile * block_size + tid;
                if (global_j < n && global_j != body_i)
                {
                    double mj = s_m[tid];
                    double dx = s_qx[tid] - qi_x;
                    double dy = s_qy[tid] - qi_y;
                    double dz = s_qz[tid] - qi_z;

                    double dist_sq = dx * dx + dy * dy + dz * dz + param::eps_sq;
                    double dist3 = pow(dist_sq, 1.5);
                    double force_factor = param::G * mj / dist3;

                    acc_x += force_factor * dx;
                    acc_y += force_factor * dy;
                    acc_z += force_factor * dz;
                }
                __syncthreads();
            }

            // Block reduction
            int warp_id = tid / 64, lane_id = tid % 64;
            acc_x = warp_reduce_sum(acc_x);
            acc_y = warp_reduce_sum(acc_y);
            acc_z = warp_reduce_sum(acc_z);

            if (lane_id == 0)
                s_reduce[warp_id] = acc_x;
            __syncthreads();
            if (tid == 0)
                ax_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_y;
            __syncthreads();
            if (tid == 0)
                ay_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_z;
            __syncthreads();
            if (tid == 0)
                az_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
            __syncthreads();
        }
        grid.sync();

        // Phase 2: Update velocities and positions
        for (int i = bid * block_size + tid; i < n; i += grid_size * block_size)
        {
            double ax = ax_global[i], ay = ay_global[i], az = az_global[i];
            vx[i] = __fma_rn(ax, param::dt, vx[i]);
            vy[i] = __fma_rn(ay, param::dt, vy[i]);
            vz[i] = __fma_rn(az, param::dt, vz[i]);
            qx[i] = __fma_rn(vx[i], param::dt, qx[i]);
            qy[i] = __fma_rn(vy[i], param::dt, qy[i]);
            qz[i] = __fma_rn(vz[i], param::dt, qz[i]);
        }
        grid.sync();

        // Phase 3: Check collision and intercept
        if (bid == 0 && tid == 0)
        {
            // Collision check
            double dx = qx[planet_id] - qx[asteroid_id];
            double dy = qy[planet_id] - qy[asteroid_id];
            double dz = qz[planet_id] - qz[asteroid_id];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < param::planet_radius * param::planet_radius)
            {
                atomicMin(collision_step_out, step);
            }
        }

        // Intercept check (parallel over devices)
        for (int d = bid * block_size + tid; d < num_devices; d += grid_size * block_size)
        {
            if (intercept_steps[d] == INT_MAX)
            { // Not yet intercepted
                int dev_id = device_ids[d];
                double dx = qx[dev_id] - qx[planet_id];
                double dy = qy[dev_id] - qy[planet_id];
                double dz = qz[dev_id] - qz[planet_id];
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                if (missile_traveled >= dist)
                {
                    atomicMin(&intercept_steps[d], step);
                }
            }
        }
        grid.sync();
    }
}

// ============================================================================
// Pb3 Cooperative Kernel - simulates with ONE specific device destroyed after intercept
// Self-contained: calculates intercept step internally
// ============================================================================
__global__ void __launch_bounds__(256) pb3_kernel(
    double *__restrict__ qx, double *__restrict__ qy, double *__restrict__ qz,
    double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
    const double *__restrict__ mass0,
    const int *__restrict__ body_type,
    double *__restrict__ ax_global, double *__restrict__ ay_global, double *__restrict__ az_global,
    int n, int planet_id, int asteroid_id,
    int target_device_id, // The device to destroy
    int *__restrict__ collision_step_out,
    int *__restrict__ intercept_step_out,
    int n_steps)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;

    __shared__ double s_qx[256], s_qy[256], s_qz[256], s_m[256];
    __shared__ double s_reduce[4];
    __shared__ int s_intercept_step;
    __shared__ bool s_device_destroyed;

    // Initialize outputs
    if (bid == 0 && tid == 0)
    {
        *collision_step_out = INT_MAX;
        *intercept_step_out = INT_MAX;
        s_intercept_step = INT_MAX;
        s_device_destroyed = false;
    }
    grid.sync();

    for (int step = 1; step <= n_steps; step++)
    {
        double t = step * param::dt;
        double missile_traveled = t * param::missile_speed;

        // Check if target device is intercepted this step (only if not already destroyed)
        if (bid == 0 && tid == 0 && !s_device_destroyed)
        {
            double dx = qx[target_device_id] - qx[planet_id];
            double dy = qy[target_device_id] - qy[planet_id];
            double dz = qz[target_device_id] - qz[planet_id];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (missile_traveled >= dist)
            {
                s_intercept_step = step;
                s_device_destroyed = true;
                *intercept_step_out = step;
            }
        }
        grid.sync();

        // Phase 1: Compute accelerations
        for (int body_i = bid; body_i < n; body_i += grid_size)
        {
            double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
            double qi_x = qx[body_i], qi_y = qy[body_i], qi_z = qz[body_i];

            int num_tiles = (n + block_size - 1) / block_size;
            for (int tile = 0; tile < num_tiles; tile++)
            {
                int j = tile * block_size + tid;
                if (j < n)
                {
                    s_qx[tid] = qx[j];
                    s_qy[tid] = qy[j];
                    s_qz[tid] = qz[j];
                    double m0 = mass0[j];
                    if (body_type[j] == 1)
                    {
                        // Device: check if it's the destroyed one
                        if (j == target_device_id && s_device_destroyed)
                        {
                            s_m[tid] = 0.0; // Destroyed
                        }
                        else
                        {
                            s_m[tid] = param::gravity_device_mass(m0, t);
                        }
                    }
                    else
                    {
                        s_m[tid] = m0;
                    }
                }
                else
                {
                    s_qx[tid] = s_qy[tid] = s_qz[tid] = s_m[tid] = 0.0;
                }
                __syncthreads();

                int global_j = tile * block_size + tid;
                if (global_j < n && global_j != body_i)
                {
                    double mj = s_m[tid];
                    double dx = s_qx[tid] - qi_x;
                    double dy = s_qy[tid] - qi_y;
                    double dz = s_qz[tid] - qi_z;

                    double dist_sq = dx * dx + dy * dy + dz * dz + param::eps_sq;
                    double dist3 = pow(dist_sq, 1.5);
                    double force_factor = param::G * mj / dist3;

                    acc_x += force_factor * dx;
                    acc_y += force_factor * dy;
                    acc_z += force_factor * dz;
                }
                __syncthreads();
            }

            // Block reduction
            int warp_id = tid / 64, lane_id = tid % 64;
            acc_x = warp_reduce_sum(acc_x);
            acc_y = warp_reduce_sum(acc_y);
            acc_z = warp_reduce_sum(acc_z);

            if (lane_id == 0)
                s_reduce[warp_id] = acc_x;
            __syncthreads();
            if (tid == 0)
                ax_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_y;
            __syncthreads();
            if (tid == 0)
                ay_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_z;
            __syncthreads();
            if (tid == 0)
                az_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
            __syncthreads();
        }
        grid.sync();

        // Phase 2: Update velocities and positions
        for (int i = bid * block_size + tid; i < n; i += grid_size * block_size)
        {
            double ax = ax_global[i], ay = ay_global[i], az = az_global[i];
            vx[i] = __fma_rn(ax, param::dt, vx[i]);
            vy[i] = __fma_rn(ay, param::dt, vy[i]);
            vz[i] = __fma_rn(az, param::dt, vz[i]);
            qx[i] = __fma_rn(vx[i], param::dt, qx[i]);
            qy[i] = __fma_rn(vy[i], param::dt, qy[i]);
            qz[i] = __fma_rn(vz[i], param::dt, qz[i]);
        }
        grid.sync();

        // Phase 3: Check collision
        if (bid == 0 && tid == 0)
        {
            double dx = qx[planet_id] - qx[asteroid_id];
            double dy = qy[planet_id] - qy[asteroid_id];
            double dz = qz[planet_id] - qz[asteroid_id];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < param::planet_radius * param::planet_radius)
            {
                atomicMin(collision_step_out, step);
            }
        }
        grid.sync();
    }
}

// ============================================================================
// Pb3 Multi-Device Cooperative Kernel
// Processes multiple devices in ONE kernel launch
// Grid is divided into num_devices "rows", each row handles one device scenario
// Uses global memory for device_destroyed flags so all blocks of same device share state
// ============================================================================
__global__ void __launch_bounds__(256) pb3_multi_kernel(
    // Per-device state arrays (num_devices * n elements each, or num_devices elements for flags)
    double *__restrict__ all_qx, double *__restrict__ all_qy, double *__restrict__ all_qz,
    double *__restrict__ all_vx, double *__restrict__ all_vy, double *__restrict__ all_vz,
    double *__restrict__ all_ax, double *__restrict__ all_ay, double *__restrict__ all_az,
    // Shared data (n elements each)
    const double *__restrict__ mass0,
    const int *__restrict__ body_type,
    // Problem parameters
    int n, int planet_id, int asteroid_id,
    // Device info
    const int *__restrict__ target_device_ids, // Which device to destroy for each row
    int num_devices,                           // Number of devices (rows)
    int blocks_per_device,                     // How many blocks per device simulation
    // Outputs (num_devices elements each)
    int *__restrict__ collision_steps_out,
    int *__restrict__ intercept_steps_out,
    // Per-device destroyed flags in GLOBAL memory (num_devices elements)
    int *__restrict__ device_destroyed_flags,
    int n_steps)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;

    // Determine which device (row) this block belongs to
    const int device_idx = bid / blocks_per_device;
    const int local_bid = bid % blocks_per_device; // Block index within this device's simulation

    if (device_idx >= num_devices)
        return; // Extra blocks, do nothing

    // Get pointers to this device's data
    const size_t offset = device_idx * n;
    double *qx = all_qx + offset;
    double *qy = all_qy + offset;
    double *qz = all_qz + offset;
    double *vx = all_vx + offset;
    double *vy = all_vy + offset;
    double *vz = all_vz + offset;
    double *ax_global = all_ax + offset;
    double *ay_global = all_ay + offset;
    double *az_global = all_az + offset;

    const int target_device_id = target_device_ids[device_idx];

    __shared__ double s_qx[256], s_qy[256], s_qz[256], s_m[256];
    __shared__ double s_reduce[4];

    // Initialize per-device state (only first block of each device)
    if (local_bid == 0 && tid == 0)
    {
        collision_steps_out[device_idx] = INT_MAX;
        intercept_steps_out[device_idx] = INT_MAX;
        device_destroyed_flags[device_idx] = 0;
    }
    grid.sync();

    for (int step = 1; step <= n_steps; step++)
    {
        double t = step * param::dt;
        double missile_traveled = t * param::missile_speed;

        // Check if target device is intercepted this step (only first block of each device)
        if (local_bid == 0 && tid == 0 && device_destroyed_flags[device_idx] == 0)
        {
            double dx = qx[target_device_id] - qx[planet_id];
            double dy = qy[target_device_id] - qy[planet_id];
            double dz = qz[target_device_id] - qz[planet_id];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (missile_traveled >= dist)
            {
                device_destroyed_flags[device_idx] = 1;
                intercept_steps_out[device_idx] = step;
            }
        }
        grid.sync();

        // Read destroyed flag from global memory
        int is_destroyed = device_destroyed_flags[device_idx];

        // Phase 1: Compute accelerations (each block handles some bodies)
        for (int body_i = local_bid; body_i < n; body_i += blocks_per_device)
        {
            double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
            double qi_x = qx[body_i], qi_y = qy[body_i], qi_z = qz[body_i];

            int num_tiles = (n + block_size - 1) / block_size;
            for (int tile = 0; tile < num_tiles; tile++)
            {
                int j = tile * block_size + tid;
                if (j < n)
                {
                    s_qx[tid] = qx[j];
                    s_qy[tid] = qy[j];
                    s_qz[tid] = qz[j];
                    double m0 = mass0[j];
                    if (body_type[j] == 1)
                    {
                        if (j == target_device_id && is_destroyed)
                        {
                            s_m[tid] = 0.0;
                        }
                        else
                        {
                            s_m[tid] = param::gravity_device_mass(m0, t);
                        }
                    }
                    else
                    {
                        s_m[tid] = m0;
                    }
                }
                else
                {
                    s_qx[tid] = s_qy[tid] = s_qz[tid] = s_m[tid] = 0.0;
                }
                __syncthreads();

                int global_j = tile * block_size + tid;
                if (global_j < n && global_j != body_i)
                {
                    double mj = s_m[tid];
                    double dx = s_qx[tid] - qi_x;
                    double dy = s_qy[tid] - qi_y;
                    double dz = s_qz[tid] - qi_z;

                    double dist_sq = dx * dx + dy * dy + dz * dz + param::eps_sq;
                    double dist3 = pow(dist_sq, 1.5);
                    double force_factor = param::G * mj / dist3;

                    acc_x += force_factor * dx;
                    acc_y += force_factor * dy;
                    acc_z += force_factor * dz;
                }
                __syncthreads();
            }

            // Block reduction
            int warp_id = tid / 64, lane_id = tid % 64;
            acc_x = warp_reduce_sum(acc_x);
            acc_y = warp_reduce_sum(acc_y);
            acc_z = warp_reduce_sum(acc_z);

            if (lane_id == 0)
                s_reduce[warp_id] = acc_x;
            __syncthreads();
            if (tid == 0)
                ax_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_y;
            __syncthreads();
            if (tid == 0)
                ay_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

            if (lane_id == 0)
                s_reduce[warp_id] = acc_z;
            __syncthreads();
            if (tid == 0)
                az_global[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
            __syncthreads();
        }
        grid.sync();

        // Phase 2: Update velocities and positions
        for (int i = local_bid * block_size + tid; i < n; i += blocks_per_device * block_size)
        {
            double ax_i = ax_global[i], ay_i = ay_global[i], az_i = az_global[i];
            vx[i] = __fma_rn(ax_i, param::dt, vx[i]);
            vy[i] = __fma_rn(ay_i, param::dt, vy[i]);
            vz[i] = __fma_rn(az_i, param::dt, vz[i]);
            qx[i] = __fma_rn(vx[i], param::dt, qx[i]);
            qy[i] = __fma_rn(vy[i], param::dt, qy[i]);
            qz[i] = __fma_rn(vz[i], param::dt, qz[i]);
        }
        grid.sync();

        // Phase 3: Check collision (only first block of each device)
        if (local_bid == 0 && tid == 0)
        {
            double dx = qx[planet_id] - qx[asteroid_id];
            double dy = qy[planet_id] - qy[asteroid_id];
            double dz = qz[planet_id] - qz[asteroid_id];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < param::planet_radius * param::planet_radius)
            {
                atomicMin(&collision_steps_out[device_idx], step);
            }
        }
        grid.sync();
    }
}

// ============================================================================
// Pb2/Pb3 Force Kernel - compute accelerations with time-varying device mass
// ============================================================================
__global__ void __launch_bounds__(256) force_kernel(
    const double *__restrict__ qx, const double *__restrict__ qy, const double *__restrict__ qz,
    const double *__restrict__ mass0,
    const int *__restrict__ body_type,
    double *__restrict__ ax, double *__restrict__ ay, double *__restrict__ az,
    int n, double t, int destroyed_device_id) // destroyed_device_id = -1 for Pb2
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = gridDim.x;

    __shared__ double s_qx[256], s_qy[256], s_qz[256], s_m[256];
    __shared__ double s_reduce[4];

    for (int body_i = bid; body_i < n; body_i += grid_size)
    {
        double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
        double qi_x = qx[body_i], qi_y = qy[body_i], qi_z = qz[body_i];

        int num_tiles = (n + block_size - 1) / block_size;
        for (int tile = 0; tile < num_tiles; tile++)
        {
            int j = tile * block_size + tid;
            if (j < n)
            {
                s_qx[tid] = qx[j];
                s_qy[tid] = qy[j];
                s_qz[tid] = qz[j];
                double mj = mass0[j];
                if (body_type[j] == 1)
                {
                    if (j == destroyed_device_id)
                    {
                        mj = 0.0; // Destroyed device
                    }
                    else
                    {
                        mj = param::gravity_device_mass(mj, t);
                    }
                }
                s_m[tid] = mj;
            }
            else
            {
                s_qx[tid] = s_qy[tid] = s_qz[tid] = s_m[tid] = 0.0;
            }
            __syncthreads();

            int global_j = tile * block_size + tid;
            if (global_j < n && global_j != body_i)
            {
                double mj = s_m[tid];
                double dx = s_qx[tid] - qi_x;
                double dy = s_qy[tid] - qi_y;
                double dz = s_qz[tid] - qi_z;

                double dist_sq = dx * dx + dy * dy + dz * dz + param::eps_sq;
                double dist3 = pow(dist_sq, 1.5);
                double force_factor = param::G * mj / dist3;

                acc_x += force_factor * dx;
                acc_y += force_factor * dy;
                acc_z += force_factor * dz;
            }
            __syncthreads();
        }

        // Block reduction
        int warp_id = tid / 64, lane_id = tid % 64;
        acc_x = warp_reduce_sum(acc_x);
        acc_y = warp_reduce_sum(acc_y);
        acc_z = warp_reduce_sum(acc_z);

        if (lane_id == 0)
            s_reduce[warp_id] = acc_x;
        __syncthreads();
        if (tid == 0)
            ax[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

        if (lane_id == 0)
            s_reduce[warp_id] = acc_y;
        __syncthreads();
        if (tid == 0)
            ay[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];

        if (lane_id == 0)
            s_reduce[warp_id] = acc_z;
        __syncthreads();
        if (tid == 0)
            az[body_i] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
        __syncthreads();
    }
}

// ============================================================================
// Update Kernel - update velocities and positions
// ============================================================================
__global__ void update_kernel(
    double *__restrict__ qx, double *__restrict__ qy, double *__restrict__ qz,
    double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
    const double *__restrict__ ax, const double *__restrict__ ay, const double *__restrict__ az,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        vx[i] = __fma_rn(ax[i], param::dt, vx[i]);
        vy[i] = __fma_rn(ay[i], param::dt, vy[i]);
        vz[i] = __fma_rn(az[i], param::dt, vz[i]);
        qx[i] = __fma_rn(vx[i], param::dt, qx[i]);
        qy[i] = __fma_rn(vy[i], param::dt, qy[i]);
        qz[i] = __fma_rn(vz[i], param::dt, qz[i]);
    }
}

// ============================================================================
// Collision Check Kernel
// ============================================================================
// Collision Check Kernel - records first collision step using atomicMin
// ============================================================================
__global__ void collision_kernel(
    const double *__restrict__ qx, const double *__restrict__ qy, const double *__restrict__ qz,
    int planet_id, int asteroid_id,
    int step,
    int *__restrict__ first_collision_step) // Initialize to INT_MAX, use atomicMin
{
    double dx = qx[planet_id] - qx[asteroid_id];
    double dy = qy[planet_id] - qy[asteroid_id];
    double dz = qz[planet_id] - qz[asteroid_id];
    double dist_sq = dx * dx + dy * dy + dz * dz;

    if (dist_sq < param::planet_radius * param::planet_radius)
    {
        atomicMin(first_collision_step, step);
    }
}

// ============================================================================
// Intercept Check Kernel - checks all devices for interception, records first intercept step
// Each thread handles one device
// ============================================================================
__global__ void intercept_kernel(
    const double *__restrict__ qx, const double *__restrict__ qy, const double *__restrict__ qz,
    const int *__restrict__ device_ids, int num_devices,
    int planet_id, int step, double missile_speed,
    int *__restrict__ intercept_steps) // One per device, initialized to INT_MAX
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= num_devices)
        return;

    // Only check if not yet intercepted
    if (intercept_steps[d] < INT_MAX)
        return;

    int dev_id = device_ids[d];
    double dx = qx[dev_id] - qx[planet_id];
    double dy = qy[dev_id] - qy[planet_id];
    double dz = qz[dev_id] - qz[planet_id];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    double t = step * param::dt;
    double missile_traveled = t * missile_speed;

    if (missile_traveled >= dist)
    {
        atomicMin(&intercept_steps[d], step);
    }
}

// ============================================================================
// I/O Functions
// ============================================================================
void read_input(const char *filename, int &n, int &planet, int &asteroid,
                std::vector<double> &qx, std::vector<double> &qy, std::vector<double> &qz,
                std::vector<double> &vx, std::vector<double> &vy, std::vector<double> &vz,
                std::vector<double> &mass, std::vector<int> &type,
                std::vector<int> &device_ids)
{
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;

    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    mass.resize(n);
    type.resize(n);

    for (int i = 0; i < n; i++)
    {
        std::string type_str;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> mass[i] >> type_str;
        if (type_str == "device")
        {
            type[i] = 1;
            device_ids.push_back(i);
        }
        else
        {
            type[i] = 0;
        }
    }
}

void write_output(const char *filename, double min_dist, int hit_time_step,
                  int gravity_device_id, double missile_cost)
{
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1)
         << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// ============================================================================
// Get max grid size for cooperative launch
// ============================================================================
int get_max_grid_size(int device, void *kernel, int block_size)
{
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    int max_blocks_per_sm;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));

    return max_blocks_per_sm * prop.multiProcessorCount;
}

// ============================================================================
// Pb3 simulation state
// ============================================================================
struct Pb3State
{
    int device_id;       // Which device is destroyed
    int gpu_id;          // Which GPU this Pb3 runs on (0 or 1)
    int intercept_step;  // When device was intercepted
    int current_step;    // Current simulation step
    bool collision;      // Has collision occurred?
    bool completed;      // Simulation finished?
    double missile_cost; // Result

    // Device pointers (allocated on the assigned GPU)
    double *qx, *qy, *qz;
    double *vx, *vy, *vz;
    double *ax, *ay, *az;
    int *collision_flag;
    int *intercept_flag; // For pb3_kernel output

    // Shared data pointers (mass0, body_type) - need per-GPU copy
    double *mass0;
    int *body_type;

    hipStream_t stream;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // Read input
    int n, planet_id, asteroid_id;
    std::vector<double> qx, qy, qz, vx, vy, vz, mass;
    std::vector<int> type, device_ids;
    read_input(argv[1], n, planet_id, asteroid_id, qx, qy, qz, vx, vy, vz, mass, type, device_ids);

    int num_devices_body = device_ids.size();
    fprintf(stderr, "N=%d, planet=%d, asteroid=%d, num_devices=%d\n",
            n, planet_id, asteroid_id, num_devices_body);

    // Check GPU count
    int gpu_count;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    fprintf(stderr, "Available GPUs: %d\n", gpu_count);

    const int GPU0 = 0;                        // Pb1
    const int GPU1 = (gpu_count >= 2) ? 1 : 0; // Pb2 + Pb3
    const int block_size = 256;

    // ========================================================================
    // GPU0: Pb1 Setup
    // ========================================================================
    HIP_CHECK(hipSetDevice(GPU0));

    int max_grid_pb1 = get_max_grid_size(GPU0, (void *)pb1_kernel, block_size);
    int grid_size_pb1 = std::min(n, max_grid_pb1);
    fprintf(stderr, "GPU0 (Pb1): grid_size=%d\n", grid_size_pb1);

    // Allocate GPU0 memory
    double *d0_qx, *d0_qy, *d0_qz, *d0_vx, *d0_vy, *d0_vz;
    double *d0_ax, *d0_ay, *d0_az, *d0_mass, *d0_min_dist;

    HIP_CHECK(hipMalloc(&d0_qx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_qy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_qz, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_vx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_vy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_vz, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_ax, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_ay, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_az, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_mass, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d0_min_dist, sizeof(double)));

    // Prepare Pb1 data (devices mass = 0)
    std::vector<double> mass_pb1 = mass;
    for (int i = 0; i < n; i++)
    {
        if (type[i] == 1)
            mass_pb1[i] = 0.0;
    }

    HIP_CHECK(hipMemcpy(d0_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d0_mass, mass_pb1.data(), n * sizeof(double), hipMemcpyHostToDevice));

    hipStream_t stream_pb1;
    HIP_CHECK(hipStreamCreate(&stream_pb1));

    // ========================================================================
    // GPU1: Pb2 + Pb3 Setup
    // ========================================================================
    HIP_CHECK(hipSetDevice(GPU1));

    int grid_size_force = std::min(n, 480);
    fprintf(stderr, "GPU1 (Pb2/Pb3): grid_size=%d\n", grid_size_force);

    // Allocate GPU1 memory for Pb2
    double *d1_qx, *d1_qy, *d1_qz, *d1_vx, *d1_vy, *d1_vz;
    double *d1_ax, *d1_ay, *d1_az, *d1_mass0;
    int *d1_body_type, *d1_collision_flag;

    HIP_CHECK(hipMalloc(&d1_qx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_qy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_qz, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_vx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_vy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_vz, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_ax, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_ay, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_az, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_mass0, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d1_body_type, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d1_collision_flag, sizeof(int)));

    // Initialize collision flag to INT_MAX (no collision yet)
    int init_collision = INT_MAX;
    HIP_CHECK(hipMemcpy(d1_collision_flag, &init_collision, sizeof(int), hipMemcpyHostToDevice));

    // Checkpoint storage
    double *d1_ckpt_qx, *d1_ckpt_qy, *d1_ckpt_qz;
    double *d1_ckpt_vx, *d1_ckpt_vy, *d1_ckpt_vz;
    size_t ckpt_size = num_devices_body * n * sizeof(double);
    HIP_CHECK(hipMalloc(&d1_ckpt_qx, ckpt_size));
    HIP_CHECK(hipMalloc(&d1_ckpt_qy, ckpt_size));
    HIP_CHECK(hipMalloc(&d1_ckpt_qz, ckpt_size));
    HIP_CHECK(hipMalloc(&d1_ckpt_vx, ckpt_size));
    HIP_CHECK(hipMalloc(&d1_ckpt_vy, ckpt_size));
    HIP_CHECK(hipMalloc(&d1_ckpt_vz, ckpt_size));

    HIP_CHECK(hipMemcpy(d1_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_mass0, mass.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d1_body_type, type.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Device IDs and intercept steps on GPU for per-step interception check
    int *d1_device_ids, *d1_intercept_steps;
    HIP_CHECK(hipMalloc(&d1_device_ids, num_devices_body * sizeof(int)));
    HIP_CHECK(hipMalloc(&d1_intercept_steps, num_devices_body * sizeof(int)));
    HIP_CHECK(hipMemcpy(d1_device_ids, device_ids.data(), num_devices_body * sizeof(int), hipMemcpyHostToDevice));
    // Initialize all intercept steps to INT_MAX
    std::vector<int> init_intercept_steps(num_devices_body, INT_MAX);
    HIP_CHECK(hipMemcpy(d1_intercept_steps, init_intercept_steps.data(), num_devices_body * sizeof(int), hipMemcpyHostToDevice));
    std::vector<int> h_intercept_steps(num_devices_body, INT_MAX);

    hipStream_t stream_pb2;
    HIP_CHECK(hipStreamCreate(&stream_pb2));

    // Pb3 states (one per device) - distribute across GPUs
    // GPU0 gets first half, GPU1 gets second half
    std::vector<Pb3State> pb3_states(num_devices_body);
    int pb3_split = (num_devices_body + 1) / 2; // First pb3_split go to GPU0

    for (int d = 0; d < num_devices_body; d++)
    {
        int assigned_gpu = (d < pb3_split) ? GPU0 : GPU1;
        pb3_states[d].device_id = device_ids[d];
        pb3_states[d].gpu_id = assigned_gpu;
        pb3_states[d].intercept_step = 0;
        pb3_states[d].current_step = 0;
        pb3_states[d].collision = false;
        pb3_states[d].completed = false;
        pb3_states[d].missile_cost = -1.0;

        // Allocate on the assigned GPU
        HIP_CHECK(hipSetDevice(assigned_gpu));
        HIP_CHECK(hipStreamCreate(&pb3_states[d].stream));
        HIP_CHECK(hipMalloc(&pb3_states[d].qx, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].qy, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].qz, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].vx, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].vy, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].vz, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].ax, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].ay, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].az, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].collision_flag, sizeof(int)));
        HIP_CHECK(hipMalloc(&pb3_states[d].intercept_flag, sizeof(int)));
        HIP_CHECK(hipMalloc(&pb3_states[d].mass0, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&pb3_states[d].body_type, n * sizeof(int)));

        // Initialize flags
        HIP_CHECK(hipMemcpy(pb3_states[d].collision_flag, &init_collision, sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(pb3_states[d].intercept_flag, &init_collision, sizeof(int), hipMemcpyHostToDevice));
        // Copy mass and type data
        HIP_CHECK(hipMemcpy(pb3_states[d].mass0, mass.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(pb3_states[d].body_type, type.data(), n * sizeof(int), hipMemcpyHostToDevice));

        fprintf(stderr, "  Pb3[%d] for device %d assigned to GPU%d\n", d, device_ids[d], assigned_gpu);
    }

    // Host tracking
    // (device_intercepted not needed anymore since Pb2 runs all at once)

    // ========================================================================
    // Launch Pb1 on GPU0 (full 200k steps, async)
    // ========================================================================
    fprintf(stderr, "Launching Pb1 on GPU%d...\n", GPU0);
    HIP_CHECK(hipSetDevice(GPU0));

    int n_steps_pb1 = param::n_steps;
    void *args_pb1[] = {
        &d0_qx, &d0_qy, &d0_qz, &d0_vx, &d0_vy, &d0_vz, &d0_mass,
        &d0_ax, &d0_ay, &d0_az,
        &n, &planet_id, &asteroid_id, &d0_min_dist,
        &n_steps_pb1};

    HIP_CHECK(hipLaunchCooperativeKernel(
        (void *)pb1_kernel,
        dim3(grid_size_pb1), dim3(block_size),
        args_pb1, 0, stream_pb1));

    // ========================================================================
    // Run Pb2 on GPU1 using cooperative kernel (全部 200k 步在 kernel 內完成)
    // ========================================================================
    fprintf(stderr, "Running Pb2 on GPU%d (cooperative kernel)...\n", GPU1);
    HIP_CHECK(hipSetDevice(GPU1));

    int hit_time_step = -2;

    dim3 grid_force(grid_size_force);
    dim3 grid_update((n + block_size - 1) / block_size);
    dim3 block(block_size);

    // Get max grid size for Pb2 cooperative kernel
    int grid_size_pb2 = get_max_grid_size(GPU1, (void *)pb2_kernel, block_size);
    grid_size_pb2 = std::min(grid_size_pb2, n); // Don't need more blocks than bodies
    fprintf(stderr, "GPU%d (Pb2): grid_size=%d\n", GPU1, grid_size_pb2);

    int n_steps_pb2 = param::n_steps;
    void *args_pb2[] = {
        &d1_qx, &d1_qy, &d1_qz, &d1_vx, &d1_vy, &d1_vz,
        &d1_mass0, &d1_body_type, &d1_ax, &d1_ay, &d1_az,
        &n, &planet_id, &asteroid_id,
        &d1_device_ids, &num_devices_body,
        &d1_collision_flag, &d1_intercept_steps,
        &n_steps_pb2};

    HIP_CHECK(hipLaunchCooperativeKernel(
        (void *)pb2_kernel,
        dim3(grid_size_pb2), dim3(block_size),
        args_pb2, 0, stream_pb2));

    // Wait for Pb2 to complete
    HIP_CHECK(hipStreamSynchronize(stream_pb2));

    // Get Pb2 results
    int h_collision_flag;
    HIP_CHECK(hipMemcpy(&h_collision_flag, d1_collision_flag, sizeof(int), hipMemcpyDeviceToHost));
    if (h_collision_flag < INT_MAX)
    {
        hit_time_step = h_collision_flag;
        fprintf(stderr, "  Pb2: Collision at step %d\n", hit_time_step);
    }
    else
    {
        fprintf(stderr, "  Pb2: No collision\n");
    }

    // Get intercept steps from Pb2 (informational only)
    HIP_CHECK(hipMemcpy(h_intercept_steps.data(), d1_intercept_steps,
                        num_devices_body * sizeof(int), hipMemcpyDeviceToHost));

    // ========================================================================
    // Run Pb3 on BOTH GPUs using pb3_multi_kernel
    // Each GPU runs ONE kernel that handles multiple devices simultaneously
    // GPU0: devices [0, pb3_split)
    // GPU1: devices [pb3_split, num_devices_body)
    // ========================================================================
    fprintf(stderr, "Running Pb3 simulations on both GPUs (multi-device kernels)...\n");
    fprintf(stderr, "  GPU0 handles %d devices, GPU1 handles %d devices\n",
            pb3_split, num_devices_body - pb3_split);

    int num_devices_gpu0 = pb3_split;
    int num_devices_gpu1 = num_devices_body - pb3_split;

    // Get max grid size for pb3_multi_kernel on each GPU
    int max_grid_gpu0 = get_max_grid_size(GPU0, (void *)pb3_multi_kernel, block_size);
    int max_grid_gpu1 = get_max_grid_size(GPU1, (void *)pb3_multi_kernel, block_size);

    // Calculate blocks_per_device (at least 1, divide grid among devices)
    int blocks_per_device_gpu0 = (num_devices_gpu0 > 0) ? std::max(1, std::min(n, max_grid_gpu0 / num_devices_gpu0)) : 0;
    int blocks_per_device_gpu1 = (num_devices_gpu1 > 0) ? std::max(1, std::min(n, max_grid_gpu1 / num_devices_gpu1)) : 0;

    int total_blocks_gpu0 = blocks_per_device_gpu0 * num_devices_gpu0;
    int total_blocks_gpu1 = blocks_per_device_gpu1 * num_devices_gpu1;

    fprintf(stderr, "  GPU0: %d devices × %d blocks/device = %d total blocks (max=%d)\n",
            num_devices_gpu0, blocks_per_device_gpu0, total_blocks_gpu0, max_grid_gpu0);
    fprintf(stderr, "  GPU1: %d devices × %d blocks/device = %d total blocks (max=%d)\n",
            num_devices_gpu1, blocks_per_device_gpu1, total_blocks_gpu1, max_grid_gpu1);

    // Allocate combined arrays for pb3_multi_kernel
    // GPU0's Pb3 data
    double *d0_pb3_qx, *d0_pb3_qy, *d0_pb3_qz;
    double *d0_pb3_vx, *d0_pb3_vy, *d0_pb3_vz;
    double *d0_pb3_ax, *d0_pb3_ay, *d0_pb3_az;
    int *d0_pb3_collision, *d0_pb3_intercept, *d0_pb3_targets;
    double *d0_pb3_mass0;
    int *d0_pb3_body_type;

    // GPU1's Pb3 data
    double *d1_pb3_qx, *d1_pb3_qy, *d1_pb3_qz;
    double *d1_pb3_vx, *d1_pb3_vy, *d1_pb3_vz;
    double *d1_pb3_ax, *d1_pb3_ay, *d1_pb3_az;
    int *d1_pb3_collision, *d1_pb3_intercept, *d1_pb3_targets;
    double *d1_pb3_mass0;
    int *d1_pb3_body_type;

    hipStream_t stream_pb3_gpu0, stream_pb3_gpu1;

    // Wait for Pb1 on GPU0 first
    fprintf(stderr, "Waiting for Pb1 on GPU0...\n");
    HIP_CHECK(hipSetDevice(GPU0));
    HIP_CHECK(hipStreamSynchronize(stream_pb1));
    double min_dist;
    HIP_CHECK(hipMemcpy(&min_dist, d0_min_dist, sizeof(double), hipMemcpyDeviceToHost));

    // Allocate and initialize GPU0's Pb3 data
    if (num_devices_gpu0 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU0));
        HIP_CHECK(hipStreamCreate(&stream_pb3_gpu0));

        size_t pb3_size = num_devices_gpu0 * n * sizeof(double);
        HIP_CHECK(hipMalloc(&d0_pb3_qx, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_qy, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_qz, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_vx, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_vy, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_vz, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_ax, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_ay, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_az, pb3_size));
        HIP_CHECK(hipMalloc(&d0_pb3_collision, num_devices_gpu0 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d0_pb3_intercept, num_devices_gpu0 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d0_pb3_targets, num_devices_gpu0 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d0_pb3_mass0, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d0_pb3_body_type, n * sizeof(int)));

        // Copy shared data
        HIP_CHECK(hipMemcpy(d0_pb3_mass0, mass.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d0_pb3_body_type, type.data(), n * sizeof(int), hipMemcpyHostToDevice));

        // Allocate destroyed flags array
        int *d0_pb3_destroyed;
        HIP_CHECK(hipMalloc(&d0_pb3_destroyed, num_devices_gpu0 * sizeof(int)));
        std::vector<int> init_destroyed(num_devices_gpu0, 0);
        HIP_CHECK(hipMemcpy(d0_pb3_destroyed, init_destroyed.data(), num_devices_gpu0 * sizeof(int), hipMemcpyHostToDevice));

        // Copy initial state for each device and set targets
        std::vector<int> targets_gpu0(num_devices_gpu0);
        for (int d = 0; d < num_devices_gpu0; d++)
        {
            size_t offset = d * n;
            HIP_CHECK(hipMemcpy(d0_pb3_qx + offset, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d0_pb3_qy + offset, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d0_pb3_qz + offset, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d0_pb3_vx + offset, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d0_pb3_vy + offset, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d0_pb3_vz + offset, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            targets_gpu0[d] = device_ids[d];
        }
        HIP_CHECK(hipMemcpy(d0_pb3_targets, targets_gpu0.data(), num_devices_gpu0 * sizeof(int), hipMemcpyHostToDevice));

        // Initialize collision/intercept outputs
        std::vector<int> init_vals(num_devices_gpu0, INT_MAX);
        HIP_CHECK(hipMemcpy(d0_pb3_collision, init_vals.data(), num_devices_gpu0 * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d0_pb3_intercept, init_vals.data(), num_devices_gpu0 * sizeof(int), hipMemcpyHostToDevice));

        // Launch pb3_multi_kernel on GPU0
        fprintf(stderr, "Launching Pb3 multi-kernel on GPU0...\n");
        int n_steps_pb3 = param::n_steps;
        void *args_pb3_gpu0[] = {
            &d0_pb3_qx, &d0_pb3_qy, &d0_pb3_qz,
            &d0_pb3_vx, &d0_pb3_vy, &d0_pb3_vz,
            &d0_pb3_ax, &d0_pb3_ay, &d0_pb3_az,
            &d0_pb3_mass0, &d0_pb3_body_type,
            &n, &planet_id, &asteroid_id,
            &d0_pb3_targets, &num_devices_gpu0, &blocks_per_device_gpu0,
            &d0_pb3_collision, &d0_pb3_intercept,
            &d0_pb3_destroyed,
            &n_steps_pb3};

        HIP_CHECK(hipLaunchCooperativeKernel(
            (void *)pb3_multi_kernel,
            dim3(total_blocks_gpu0), dim3(block_size),
            args_pb3_gpu0, 0, stream_pb3_gpu0));
    }

    // Allocate and initialize GPU1's Pb3 data
    if (num_devices_gpu1 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU1));
        HIP_CHECK(hipStreamCreate(&stream_pb3_gpu1));

        size_t pb3_size = num_devices_gpu1 * n * sizeof(double);
        HIP_CHECK(hipMalloc(&d1_pb3_qx, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_qy, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_qz, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_vx, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_vy, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_vz, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_ax, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_ay, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_az, pb3_size));
        HIP_CHECK(hipMalloc(&d1_pb3_collision, num_devices_gpu1 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d1_pb3_intercept, num_devices_gpu1 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d1_pb3_targets, num_devices_gpu1 * sizeof(int)));
        HIP_CHECK(hipMalloc(&d1_pb3_mass0, n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d1_pb3_body_type, n * sizeof(int)));

        // Copy shared data
        HIP_CHECK(hipMemcpy(d1_pb3_mass0, mass.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d1_pb3_body_type, type.data(), n * sizeof(int), hipMemcpyHostToDevice));

        // Allocate destroyed flags array
        int *d1_pb3_destroyed;
        HIP_CHECK(hipMalloc(&d1_pb3_destroyed, num_devices_gpu1 * sizeof(int)));
        std::vector<int> init_destroyed(num_devices_gpu1, 0);
        HIP_CHECK(hipMemcpy(d1_pb3_destroyed, init_destroyed.data(), num_devices_gpu1 * sizeof(int), hipMemcpyHostToDevice));

        // Copy initial state for each device and set targets
        std::vector<int> targets_gpu1(num_devices_gpu1);
        for (int d = 0; d < num_devices_gpu1; d++)
        {
            size_t offset = d * n;
            HIP_CHECK(hipMemcpy(d1_pb3_qx + offset, qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d1_pb3_qy + offset, qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d1_pb3_qz + offset, qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d1_pb3_vx + offset, vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d1_pb3_vy + offset, vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d1_pb3_vz + offset, vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
            targets_gpu1[d] = device_ids[pb3_split + d]; // Offset by pb3_split
        }
        HIP_CHECK(hipMemcpy(d1_pb3_targets, targets_gpu1.data(), num_devices_gpu1 * sizeof(int), hipMemcpyHostToDevice));

        // Initialize collision/intercept outputs
        std::vector<int> init_vals(num_devices_gpu1, INT_MAX);
        HIP_CHECK(hipMemcpy(d1_pb3_collision, init_vals.data(), num_devices_gpu1 * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d1_pb3_intercept, init_vals.data(), num_devices_gpu1 * sizeof(int), hipMemcpyHostToDevice));

        // Launch pb3_multi_kernel on GPU1
        fprintf(stderr, "Launching Pb3 multi-kernel on GPU1...\n");
        int n_steps_pb3 = param::n_steps;
        void *args_pb3_gpu1[] = {
            &d1_pb3_qx, &d1_pb3_qy, &d1_pb3_qz,
            &d1_pb3_vx, &d1_pb3_vy, &d1_pb3_vz,
            &d1_pb3_ax, &d1_pb3_ay, &d1_pb3_az,
            &d1_pb3_mass0, &d1_pb3_body_type,
            &n, &planet_id, &asteroid_id,
            &d1_pb3_targets, &num_devices_gpu1, &blocks_per_device_gpu1,
            &d1_pb3_collision, &d1_pb3_intercept,
            &d1_pb3_destroyed,
            &n_steps_pb3};

        HIP_CHECK(hipLaunchCooperativeKernel(
            (void *)pb3_multi_kernel,
            dim3(total_blocks_gpu1), dim3(block_size),
            args_pb3_gpu1, 0, stream_pb3_gpu1));
    }

    // Wait for both Pb3 kernels to complete
    if (num_devices_gpu0 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU0));
        HIP_CHECK(hipStreamSynchronize(stream_pb3_gpu0));
    }
    if (num_devices_gpu1 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU1));
        HIP_CHECK(hipStreamSynchronize(stream_pb3_gpu1));
    }

    // Collect results from GPU0
    std::vector<int> h_collision_gpu0(num_devices_gpu0), h_intercept_gpu0(num_devices_gpu0);
    std::vector<int> h_collision_gpu1(num_devices_gpu1), h_intercept_gpu1(num_devices_gpu1);

    if (num_devices_gpu0 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU0));
        HIP_CHECK(hipMemcpy(h_collision_gpu0.data(), d0_pb3_collision, num_devices_gpu0 * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_intercept_gpu0.data(), d0_pb3_intercept, num_devices_gpu0 * sizeof(int), hipMemcpyDeviceToHost));
    }
    if (num_devices_gpu1 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU1));
        HIP_CHECK(hipMemcpy(h_collision_gpu1.data(), d1_pb3_collision, num_devices_gpu1 * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_intercept_gpu1.data(), d1_pb3_intercept, num_devices_gpu1 * sizeof(int), hipMemcpyDeviceToHost));
    }

    // Process results
    fprintf(stderr, "Collecting Pb3 results...\n");
    for (int d = 0; d < num_devices_body; d++)
    {
        int collision_step, intercept_step;
        if (d < pb3_split)
        {
            collision_step = h_collision_gpu0[d];
            intercept_step = h_intercept_gpu0[d];
        }
        else
        {
            collision_step = h_collision_gpu1[d - pb3_split];
            intercept_step = h_intercept_gpu1[d - pb3_split];
        }

        pb3_states[d].intercept_step = intercept_step;

        if (intercept_step < INT_MAX)
        {
            if (collision_step < INT_MAX)
            {
                pb3_states[d].collision = true;
                pb3_states[d].missile_cost = -1.0;
            }
            else
            {
                double launch_time = (intercept_step + 1) * param::dt;
                pb3_states[d].missile_cost = param::get_missile_cost(launch_time);
                pb3_states[d].collision = false;
            }
        }
        else
        {
            pb3_states[d].intercept_step = 0;
            pb3_states[d].missile_cost = -1.0;
        }
        pb3_states[d].completed = true;
        fprintf(stderr, "  Device %d (GPU%d): intercept=%d, collision=%s, cost=%.2e\n",
                device_ids[d], pb3_states[d].gpu_id, intercept_step,
                pb3_states[d].collision ? "yes" : "no",
                pb3_states[d].missile_cost);
    }

    // Free Pb3 multi-kernel memory
    if (num_devices_gpu0 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU0));
        HIP_CHECK(hipFree(d0_pb3_qx));
        HIP_CHECK(hipFree(d0_pb3_qy));
        HIP_CHECK(hipFree(d0_pb3_qz));
        HIP_CHECK(hipFree(d0_pb3_vx));
        HIP_CHECK(hipFree(d0_pb3_vy));
        HIP_CHECK(hipFree(d0_pb3_vz));
        HIP_CHECK(hipFree(d0_pb3_ax));
        HIP_CHECK(hipFree(d0_pb3_ay));
        HIP_CHECK(hipFree(d0_pb3_az));
        HIP_CHECK(hipFree(d0_pb3_collision));
        HIP_CHECK(hipFree(d0_pb3_intercept));
        HIP_CHECK(hipFree(d0_pb3_targets));
        HIP_CHECK(hipFree(d0_pb3_mass0));
        HIP_CHECK(hipFree(d0_pb3_body_type));
        HIP_CHECK(hipStreamDestroy(stream_pb3_gpu0));
    }
    if (num_devices_gpu1 > 0)
    {
        HIP_CHECK(hipSetDevice(GPU1));
        HIP_CHECK(hipFree(d1_pb3_qx));
        HIP_CHECK(hipFree(d1_pb3_qy));
        HIP_CHECK(hipFree(d1_pb3_qz));
        HIP_CHECK(hipFree(d1_pb3_vx));
        HIP_CHECK(hipFree(d1_pb3_vy));
        HIP_CHECK(hipFree(d1_pb3_vz));
        HIP_CHECK(hipFree(d1_pb3_ax));
        HIP_CHECK(hipFree(d1_pb3_ay));
        HIP_CHECK(hipFree(d1_pb3_az));
        HIP_CHECK(hipFree(d1_pb3_collision));
        HIP_CHECK(hipFree(d1_pb3_intercept));
        HIP_CHECK(hipFree(d1_pb3_targets));
        HIP_CHECK(hipFree(d1_pb3_mass0));
        HIP_CHECK(hipFree(d1_pb3_body_type));
        HIP_CHECK(hipStreamDestroy(stream_pb3_gpu1));
    }

    // ========================================================================
    // Find best Pb3 result
    // ========================================================================
    int best_device_id = -1;
    double best_missile_cost = 1e30; // Initialize to large value for finding minimum

    for (int d = 0; d < num_devices_body; d++)
    {
        if (pb3_states[d].intercept_step > 0)
        {
            double cost = pb3_states[d].missile_cost;
            fprintf(stderr, "  Device %d: missile_cost = %.6e\n", device_ids[d], cost);

            // Select device with LOWEST positive cost (saves the most)
            if (cost > 0 && cost < best_missile_cost)
            {
                best_missile_cost = cost;
                best_device_id = device_ids[d];
            }
        }
    }

    // If no device found, reset missile_cost to 0
    if (best_device_id == -1)
    {
        best_missile_cost = 0.0;
    }

    // ========================================================================
    // Output results
    // ========================================================================
    write_output(argv[2], min_dist, hit_time_step, best_device_id, best_missile_cost);

    fprintf(stderr, "\nResults:\n");
    fprintf(stderr, "  min_dist = %.15e\n", min_dist);
    fprintf(stderr, "  hit_time_step = %d\n", hit_time_step);
    fprintf(stderr, "  gravity_device_id = %d\n", best_device_id);
    fprintf(stderr, "  missile_cost = %.6e\n", best_missile_cost);

    // ========================================================================
    // Cleanup
    // ========================================================================
    HIP_CHECK(hipSetDevice(GPU0));
    HIP_CHECK(hipFree(d0_qx));
    HIP_CHECK(hipFree(d0_qy));
    HIP_CHECK(hipFree(d0_qz));
    HIP_CHECK(hipFree(d0_vx));
    HIP_CHECK(hipFree(d0_vy));
    HIP_CHECK(hipFree(d0_vz));
    HIP_CHECK(hipFree(d0_ax));
    HIP_CHECK(hipFree(d0_ay));
    HIP_CHECK(hipFree(d0_az));
    HIP_CHECK(hipFree(d0_mass));
    HIP_CHECK(hipFree(d0_min_dist));
    HIP_CHECK(hipStreamDestroy(stream_pb1));

    HIP_CHECK(hipSetDevice(GPU1));
    HIP_CHECK(hipFree(d1_qx));
    HIP_CHECK(hipFree(d1_qy));
    HIP_CHECK(hipFree(d1_qz));
    HIP_CHECK(hipFree(d1_vx));
    HIP_CHECK(hipFree(d1_vy));
    HIP_CHECK(hipFree(d1_vz));
    HIP_CHECK(hipFree(d1_ax));
    HIP_CHECK(hipFree(d1_ay));
    HIP_CHECK(hipFree(d1_az));
    HIP_CHECK(hipFree(d1_mass0));
    HIP_CHECK(hipFree(d1_body_type));
    HIP_CHECK(hipFree(d1_collision_flag));
    HIP_CHECK(hipFree(d1_ckpt_qx));
    HIP_CHECK(hipFree(d1_ckpt_qy));
    HIP_CHECK(hipFree(d1_ckpt_qz));
    HIP_CHECK(hipFree(d1_ckpt_vx));
    HIP_CHECK(hipFree(d1_ckpt_vy));
    HIP_CHECK(hipFree(d1_ckpt_vz));
    HIP_CHECK(hipStreamDestroy(stream_pb2));

    for (int d = 0; d < num_devices_body; d++)
    {
        HIP_CHECK(hipFree(pb3_states[d].qx));
        HIP_CHECK(hipFree(pb3_states[d].qy));
        HIP_CHECK(hipFree(pb3_states[d].qz));
        HIP_CHECK(hipFree(pb3_states[d].vx));
        HIP_CHECK(hipFree(pb3_states[d].vy));
        HIP_CHECK(hipFree(pb3_states[d].vz));
        HIP_CHECK(hipFree(pb3_states[d].ax));
        HIP_CHECK(hipFree(pb3_states[d].ay));
        HIP_CHECK(hipFree(pb3_states[d].az));
        HIP_CHECK(hipFree(pb3_states[d].collision_flag));
        HIP_CHECK(hipStreamDestroy(pb3_states[d].stream));
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "\nTotal runtime: %.3f s\n", std::chrono::duration<double>(total_end - total_start).count());

    return 0;
}
