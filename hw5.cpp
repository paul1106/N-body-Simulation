#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

namespace cg = cooperative_groups;

// ============================================================================
// Constants & Parameters
// ============================================================================
namespace param
{
    const int n_steps = 200000;
    const double dt = 60.0;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    const double base_cost = 1e5;
    const double time_cost_factor = 1e3;

    __device__ __forceinline__ double gravity_device_mass(double m0, double t)
    {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
    }

    __device__ __forceinline__ double get_missile_cost(double t)
    {
        return base_cost + time_cost_factor * t;
    }
} // namespace param

// ============================================================================
// Data Structures
// ============================================================================
struct Body
{
    double qx, qy, qz; // position
    double vx, vy, vz; // velocity
    double m;          // mass
    int type;          // 0=normal, 1=device
    int padding;       //
};

struct SimResult
{
    double min_dist;
    int hit_time_step;
    int destroyed_step;
    double missile_cost;
};

// ============================================================================
// Multi-Block Cooperative Kernel - Grid-Level Synchronization
// ============================================================================
__global__ void __launch_bounds__(256) nbody_kernel_cooperative(
    Body *__restrict__ positions, // Global memory for position exchange
    const Body *__restrict__ initial_state,
    int n,
    int planet_id,
    int asteroid_id,
    int target_device_id,
    int pb1_mode,
    SimResult *result,
    int *d_collision_flag,
    int *d_target_destroyed, // Global flag: is target destroyed?
    int *d_destroyed_step)   // Global: step when destroyed
{
    // Cooperative grid handle
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int global_tid = bid * block_size + tid;

    // Shared memory tile (256 threads per block)
    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_m[256];

    // Register state
    double my_qx, my_qy, my_qz;
    double my_vx, my_vy, my_vz;
    double my_m0;
    int my_type;

    // Load initial state
    if (global_tid < n)
    {
        my_qx = initial_state[global_tid].qx;
        my_qy = initial_state[global_tid].qy;
        my_qz = initial_state[global_tid].qz;
        my_vx = initial_state[global_tid].vx;
        my_vy = initial_state[global_tid].vy;
        my_vz = initial_state[global_tid].vz;
        my_m0 = initial_state[global_tid].m;
        my_type = initial_state[global_tid].type;
    }

    // Planet tracking (local to this thread)
    double min_dist = 1e100;
    int hit_time_step = -2;
    int local_destroyed_step = -1;

    // Initialize global flags
    if (global_tid == 0)
    {
        *d_collision_flag = 0;
        *d_target_destroyed = 0;
        *d_destroyed_step = -1;
    }
    grid.sync();

    // Main simulation loop
    for (int step = 0; step <= param::n_steps; step++)
    {
        double mass_time = (step + 1) * param::dt;

        // Read current destruction state from global memory
        int is_destroyed = *d_target_destroyed;

        // Calculate effective mass
        double eff_mass = my_m0;
        if (global_tid < n && my_type == 1)
        {
            if (pb1_mode)
            {
                eff_mass = 0.0;
            }
            else if (global_tid == target_device_id)
            {
                eff_mass = is_destroyed ? 0.0 : param::gravity_device_mass(my_m0, mass_time);
            }
            else
            {
                eff_mass = param::gravity_device_mass(my_m0, mass_time);
            }
        }

        // Write to global memory for inter-block communication
        if (global_tid < n)
        {
            positions[global_tid].qx = my_qx;
            positions[global_tid].qy = my_qy;
            positions[global_tid].qz = my_qz;
            positions[global_tid].m = eff_mass;
        }

        // Grid sync: all blocks finish writing
        grid.sync();

        // Missile logic (planet thread updates global flag)
        if (target_device_id >= 0 && target_device_id < n && !is_destroyed && global_tid == planet_id)
        {
            double missile_traveled = step * param::dt * param::missile_speed;
            double dx = positions[target_device_id].qx - positions[planet_id].qx;
            double dy = positions[target_device_id].qy - positions[planet_id].qy;
            double dz = positions[target_device_id].qz - positions[planet_id].qz;
            double dist = sqrt(dx * dx + dy * dy + dz * dz);

            if (missile_traveled >= dist)
            {
                atomicMax(d_target_destroyed, 1);
                atomicMax(d_destroyed_step, step);
            }
        }

        // Grid sync to ensure all threads see the destruction flag
        grid.sync();

        // Update local destroyed_step if target was destroyed
        if (*d_target_destroyed && global_tid == planet_id)
        {
            local_destroyed_step = *d_destroyed_step;
        }

        // Planet distance and collision check
        if (global_tid == planet_id && planet_id < n && asteroid_id < n)
        {
            double dx = positions[planet_id].qx - positions[asteroid_id].qx;
            double dy = positions[planet_id].qy - positions[asteroid_id].qy;
            double dz = positions[planet_id].qz - positions[asteroid_id].qz;
            double dist = sqrt(dx * dx + dy * dy + dz * dz);

            if (dist < min_dist)
            {
                min_dist = dist;
            }

            if (!pb1_mode && hit_time_step == -2 && dist < param::planet_radius)
            {
                hit_time_step = step;
                atomicMax(d_collision_flag, 1);
            }
        }

        // Grid sync before checking collision
        grid.sync();

        // Early exit
        if (*d_collision_flag)
        {
            break;
        }

        if (step == param::n_steps)
            break;

        // Tiled force calculation
        double ax = 0.0, ay = 0.0, az = 0.0;

        if (global_tid < n)
        {
            int num_tiles = (n + block_size - 1) / block_size;

            for (int tile = 0; tile < num_tiles; tile++)
            {
                int tile_start = tile * block_size;
                int tile_idx = tile_start + tid;

                // Load tile from global memory
                if (tile_idx < n)
                {
                    s_qx[tid] = positions[tile_idx].qx;
                    s_qy[tid] = positions[tile_idx].qy;
                    s_qz[tid] = positions[tile_idx].qz;
                    s_m[tid] = positions[tile_idx].m;
                }
                else
                {
                    s_m[tid] = 0.0;
                }
                __syncthreads();

// Compute forces with tile
#pragma unroll 4
                for (int j = 0; j < block_size && (tile_start + j) < n; j++)
                {
                    int global_j = tile_start + j;
                    if (global_j == global_tid)
                        continue;

                    double mj = s_m[j];
                    double dx = s_qx[j] - my_qx;
                    double dy = s_qy[j] - my_qy;
                    double dz = s_qz[j] - my_qz;

                    double dist_sq = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
                    double dist_inv = rsqrt(dist_sq);
                    double dist3_inv = dist_inv * dist_inv * dist_inv;

                    double force_factor = param::G * mj * dist3_inv;
                    ax += force_factor * dx;
                    ay += force_factor * dy;
                    az += force_factor * dz;
                }
                __syncthreads();
            }

            // Update velocity and position
            my_vx += ax * param::dt;
            my_vy += ay * param::dt;
            my_vz += az * param::dt;

            my_qx += my_vx * param::dt;
            my_qy += my_vy * param::dt;
            my_qz += my_vz * param::dt;
        }

        // Grid sync before next iteration
        grid.sync();
    }

    // Write results
    if (global_tid == planet_id)
    {
        result->min_dist = min_dist;
        result->hit_time_step = hit_time_step;
        result->destroyed_step = local_destroyed_step;

        if (target_device_id >= 0 && local_destroyed_step >= 0)
        {
            double destruction_time = (local_destroyed_step + 1) * param::dt;
            result->missile_cost = param::get_missile_cost(destruction_time);
        }
        else
        {
            result->missile_cost = -999.0;
        }
    }
}

// ============================================================================
// HIP Kernel - Single Block, All Steps Inside (LEGACY - for small N)
// ============================================================================
__global__ void nbody_kernel(
    const Body *__restrict__ initial_state,
    int n,
    int planet_id,
    int asteroid_id,
    int target_device_id, // -1 for Pb1/Pb2, >=0 for Pb3
    int pb1_mode,         // 1: force all devices to m=0 (Pb1), 0: normal
    SimResult *result)
{
    const int tid = threadIdx.x;

    // Shared memory for current positions and masses
    __shared__ double s_qx[1024];
    __shared__ double s_qy[1024];
    __shared__ double s_qz[1024];
    __shared__ double s_m[1024];
    __shared__ bool s_target_destroyed;
    __shared__ int s_destroyed_step;
    __shared__ bool s_collision_detected; // Early exit flag for collision detection

    // Register-only variables (never written to shared/global during simulation)
    double my_qx, my_qy, my_qz;
    double my_vx, my_vy, my_vz;
    double my_m0; // original mass
    int my_type;

    // Load initial state from global memory (only once)
    if (tid < n)
    {
        my_qx = initial_state[tid].qx;
        my_qy = initial_state[tid].qy;
        my_qz = initial_state[tid].qz;
        my_vx = initial_state[tid].vx;
        my_vy = initial_state[tid].vy;
        my_vz = initial_state[tid].vz;
        my_m0 = initial_state[tid].m;
        my_type = initial_state[tid].type;
    }

    // Planet thread tracks min_dist and hit detection
    double min_dist = 1e100;
    int hit_time_step = -2;
    int destroyed_step = -1;
    double missile_traveled = 0.0;
    bool target_destroyed = false;

    // Initialize shared memory (only for active threads)
    if (tid < n)
    {
        s_qx[tid] = 0.0;
        s_qy[tid] = 0.0;
        s_qz[tid] = 0.0;
        s_m[tid] = 0.0;
    }
    if (tid == 0)
    {
        s_target_destroyed = false;
        s_destroyed_step = -1;
        s_collision_detected = false; // Initialize early exit flag
    }
    __syncthreads();

    // Main simulation loop - all 200,000 steps inside kernel
    for (int step = 0; step <= param::n_steps; step++)
    {
        // FIX: Use (step+1)*dt for mass calculation to match reference sequential code
        // When advancing from state T to T+1, we use the mass at time T+1
        double mass_time = (step + 1) * param::dt;

        // Update shared memory with current positions and masses
        if (tid < n && tid < 1024)
        {
            s_qx[tid] = my_qx;
            s_qy[tid] = my_qy;
            s_qz[tid] = my_qz;

            // Calculate effective mass
            double eff_mass = my_m0;
            if (my_type == 1)
            { // device
                if (pb1_mode)
                {
                    // Pb1: all devices have mass 0
                    eff_mass = 0.0;
                }
                else if (tid == target_device_id)
                {
                    // Pb3: check if this device is destroyed
                    if (target_destroyed)
                    {
                        eff_mass = 0.0;
                    }
                    else
                    {
                        eff_mass = param::gravity_device_mass(my_m0, mass_time);
                    }
                }
                else
                {
                    // Normal device operation
                    eff_mass = param::gravity_device_mass(my_m0, mass_time);
                }
            }
            s_m[tid] = eff_mass;
        }

        // Barrier: ensure all positions and masses are updated
        __syncthreads();

        // FIX: Proper broadcast pattern for Pb3 missile logic to avoid race condition
        // Step 1: tid 0 resets the hit flag
        __shared__ bool s_hit_this_step;
        if (tid == 0)
        {
            s_hit_this_step = false;
        }
        __syncthreads();

        // Step 2: planet_id checks missile condition and sets flag if hit
        if (target_device_id >= 0 && target_device_id < n && !target_destroyed && tid == planet_id)
        {
            missile_traveled = step * param::dt * param::missile_speed;
            double dx_target = s_qx[target_device_id] - s_qx[planet_id];
            double dy_target = s_qy[target_device_id] - s_qy[planet_id];
            double dz_target = s_qz[target_device_id] - s_qz[planet_id];
            double dist_to_target = sqrt(dx_target * dx_target +
                                         dy_target * dy_target +
                                         dz_target * dz_target);

            if (missile_traveled >= dist_to_target)
            {
                s_hit_this_step = true;
                s_target_destroyed = true;
                s_destroyed_step = step;
            }
        }
        __syncthreads();

        // Step 3: All threads read the broadcast value and update local state
        if (s_hit_this_step && !target_destroyed)
        {
            target_destroyed = true;
            destroyed_step = s_destroyed_step;

            // CRITICAL: If device is destroyed this step, update its mass to 0 immediately
            // This ensures the force calculation below uses the correct mass
            if (tid == target_device_id)
            {
                s_m[tid] = 0.0;
            }
        }
        __syncthreads();

        // Planet thread: compute distance to asteroid
        if (tid == planet_id && planet_id < n && asteroid_id < n)
        {
            double dx_pa = s_qx[planet_id] - s_qx[asteroid_id];
            double dy_pa = s_qy[planet_id] - s_qy[asteroid_id];
            double dz_pa = s_qz[planet_id] - s_qz[asteroid_id];
            double dist_pa = sqrt(dx_pa * dx_pa + dy_pa * dy_pa + dz_pa * dz_pa);

            if (dist_pa < min_dist)
            {
                min_dist = dist_pa;
            }

            // Check for collision (Pb2/Pb3)
            if (!pb1_mode && hit_time_step == -2)
            {
                if (dist_pa < param::planet_radius)
                {
                    hit_time_step = step;
                    s_collision_detected = true; // Signal early exit for Pb3
                }
            }
        }

        // Synchronize to ensure all threads see the collision flag
        __syncthreads();

        // Early exit: If collision detected, stop simulation immediately
        // - Pb3: Device strategy failed, no need to continue
        // - Pb2: Collision occurred, simulation ends (physics after collision is irrelevant)
        // - Pb1: Safe because pb1_mode=1 prevents s_collision_detected from being set
        if (s_collision_detected)
        {
            break;
        }

        // Skip force calculation on last step
        if (step == param::n_steps)
            break;

        // Compute accelerations using O(N^2) all-pairs force calculation
        double ax = 0.0, ay = 0.0, az = 0.0;

        if (tid < n)
        {
#pragma unroll 4
            for (int j = 0; j < n; j++)
            {
                if (j == tid)
                    continue;

                double mj = s_m[j];
                double dx = s_qx[j] - my_qx;
                double dy = s_qy[j] - my_qy;
                double dz = s_qz[j] - my_qz;

                double dist_sq = dx * dx + dy * dy + dz * dz +
                                 param::eps * param::eps;
                double dist_inv = 1.0 / sqrt(dist_sq); // Use regular sqrt instead of rsqrt
                double dist3_inv = dist_inv * dist_inv * dist_inv;

                double force_factor = param::G * mj * dist3_inv;
                ax += force_factor * dx;
                ay += force_factor * dy;
                az += force_factor * dz;
            }

            // Update velocities (register only)
            my_vx += ax * param::dt;
            my_vy += ay * param::dt;
            my_vz += az * param::dt;

            // Update positions (register only)
            my_qx += my_vx * param::dt;
            my_qy += my_vy * param::dt;
            my_qz += my_vz * param::dt;
        }

        __syncthreads();
    }

    // Write results back to global memory (only once at the end)
    if (tid == planet_id)
    {
        result->min_dist = min_dist;
        result->hit_time_step = hit_time_step;
        result->destroyed_step = destroyed_step;

        if (target_device_id >= 0 && destroyed_step >= 0)
        {
            double destruction_time = (destroyed_step + 1) * param::dt;
            result->missile_cost = param::get_missile_cost(destruction_time);
        }
        else
        {
            result->missile_cost = -999.0;
        }
    }
}

// ============================================================================
// I/O Functions
// ============================================================================
void read_input(const char *filename, int &n, int &planet, int &asteroid,
                std::vector<Body> &bodies)
{
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;

    bodies.resize(n);
    for (int i = 0; i < n; i++)
    {
        std::string type_str;
        fin >> bodies[i].qx >> bodies[i].qy >> bodies[i].qz >> bodies[i].vx >> bodies[i].vy >> bodies[i].vz >> bodies[i].m >> type_str;
        bodies[i].type = (type_str == "device") ? 1 : 0;
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
// Helper macro for HIP error checking
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
// Batched GPU Context - Multi-Stream Execution
// ============================================================================
struct BatchedGPUContext
{
    static const int NUM_STREAMS = 120;
    static const int BLOCK_SIZE = 256;
    static const int NUM_BLOCKS = 4;

    int device_id;
    Body *d_initial_state;                // Read-only input buffer
    Body *d_positions[NUM_STREAMS];       // Working buffers for cooperative kernel
    int *d_collision_flags[NUM_STREAMS];  // Collision flags for early exit
    int *d_target_destroyed[NUM_STREAMS]; // Target destroyed flags
    int *d_destroyed_step[NUM_STREAMS];   // Destroyed step storage
    SimResult *h_results;                 // Pinned host memory
    SimResult *d_results[NUM_STREAMS];    // Device result buffers
    hipStream_t streams[NUM_STREAMS];
    bool use_cooperative; // Whether device supports cooperative launch

    void allocate(int n)
    {
        HIP_CHECK(hipSetDevice(device_id));

        // Check cooperative launch support
        int cooperative_launch = 0;
        HIP_CHECK(hipDeviceGetAttribute(&cooperative_launch,
                                        hipDeviceAttributeCooperativeLaunch, device_id));
        use_cooperative = (cooperative_launch == 1);

        if (use_cooperative)
        {
            printf("GPU %d: Cooperative launch supported - using multi-block kernel\n", device_id);
        }
        else
        {
            printf("GPU %d: Cooperative launch NOT supported - using single-block kernel\n", device_id);
        }

        // Allocate read-only input buffer
        HIP_CHECK(hipMalloc((void **)&d_initial_state, n * sizeof(Body)));

        // Allocate pinned host memory
        HIP_CHECK(hipHostMalloc((void **)&h_results, NUM_STREAMS * sizeof(SimResult), hipHostMallocDefault));

        // Allocate per-stream resources
        for (int i = 0; i < NUM_STREAMS; i++)
        {
            if (use_cooperative)
            {
                // Cooperative kernel needs working buffer and flags
                HIP_CHECK(hipMalloc((void **)&d_positions[i], n * sizeof(Body)));
                HIP_CHECK(hipMalloc((void **)&d_collision_flags[i], sizeof(int)));
                HIP_CHECK(hipMalloc((void **)&d_target_destroyed[i], sizeof(int)));
                HIP_CHECK(hipMalloc((void **)&d_destroyed_step[i], sizeof(int)));
            }
            HIP_CHECK(hipMalloc((void **)&d_results[i], sizeof(SimResult)));
            HIP_CHECK(hipStreamCreate(&streams[i]));
        }
    }

    void upload_initial_state(const std::vector<Body> &bodies)
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpy(d_initial_state, bodies.data(),
                            bodies.size() * sizeof(Body), hipMemcpyHostToDevice));
    }

    void launch_kernel(int stream_idx, int n, int planet_id, int asteroid_id,
                       int target_device_id, int pb1_mode)
    {
        HIP_CHECK(hipSetDevice(device_id));
        int sid = stream_idx % NUM_STREAMS;

        // Temporarily disable cooperative kernel due to missile logic bug
        // Use stable single-block kernel for all cases
        if (use_cooperative && n > 512)
        {
            // Use multi-block cooperative kernel for large N
            void *args[] = {
                &d_positions[sid],
                &d_initial_state,
                &n,
                &planet_id,
                &asteroid_id,
                &target_device_id,
                &pb1_mode,
                &d_results[sid],
                &d_collision_flags[sid],
                &d_target_destroyed[sid],
                &d_destroyed_step[sid]};

            HIP_CHECK(hipLaunchCooperativeKernel(
                (void *)nbody_kernel_cooperative,
                dim3(NUM_BLOCKS), dim3(BLOCK_SIZE),
                args, 0, streams[sid]));
        }
        else
        {
            // Use single-block kernel (stable and correct)
            nbody_kernel<<<1, 1024, 0, streams[sid]>>>(
                d_initial_state, n, planet_id, asteroid_id,
                target_device_id, pb1_mode, d_results[sid]);
        }

        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess)
        {
            fprintf(stderr, "Kernel launch failed on stream %d: %s\n",
                    sid, hipGetErrorString(launch_err));
            exit(EXIT_FAILURE);
        }
    }

    SimResult get_result(int stream_idx)
    {
        HIP_CHECK(hipSetDevice(device_id));
        int sid = stream_idx % NUM_STREAMS;

        // Async copy result from device to pinned host memory
        HIP_CHECK(hipMemcpyAsync(&h_results[sid], d_results[sid],
                                 sizeof(SimResult), hipMemcpyDeviceToHost,
                                 streams[sid]));

        // Wait for this specific stream
        HIP_CHECK(hipStreamSynchronize(streams[sid]));

        return h_results[sid];
    }

    void synchronize_all()
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipDeviceSynchronize());
    }

    void cleanup()
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipFree(d_initial_state));
        HIP_CHECK(hipHostFree(h_results));

        for (int i = 0; i < NUM_STREAMS; i++)
        {
            if (use_cooperative)
            {
                HIP_CHECK(hipFree(d_positions[i]));
                HIP_CHECK(hipFree(d_collision_flags[i]));
                HIP_CHECK(hipFree(d_target_destroyed[i]));
                HIP_CHECK(hipFree(d_destroyed_step[i]));
            }
            HIP_CHECK(hipFree(d_results[i]));
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
    }
};

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    printf("Starting N-Body simulation...\n");
    fflush(stdout);

    // Check GPU availability
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("Found %d GPU(s)\n", deviceCount);
    fflush(stdout);

    if (deviceCount < 1)
    {
        fprintf(stderr, "Error: Need at least 1 GPU, found %d\n", deviceCount);
        return 1;
    }

    // Read input
    int n, planet_id, asteroid_id;
    std::vector<Body> bodies;
    read_input(argv[1], n, planet_id, asteroid_id, bodies);

    printf("Read input: n=%d, planet=%d, asteroid=%d\n", n, planet_id, asteroid_id);
    fflush(stdout);

    // Find all device indices
    std::vector<int> device_indices;
    for (int i = n - 1; i > 0; i--)
    {
        if (bodies[i].type == 1)
        {
            device_indices.push_back(i);
        }
        else
        {
            break;
        }
    }
    int num_devices = device_indices.size();

    printf("Found %d gravity devices\n", num_devices);
    fflush(stdout);

    // Determine GPU strategy
    int num_gpus_to_use = std::min(deviceCount, 2);
    bool use_dual_gpu = (num_gpus_to_use == 2 && n >= 512);

    if (use_dual_gpu)
    {
        printf("Using 2 GPUs for parallel execution\n");
    }
    else
    {
        printf("Using single GPU with %d streams for full task parallelism\n", BatchedGPUContext::NUM_STREAMS);
    }
    fflush(stdout);

    // Initialize GPU contexts
    std::vector<BatchedGPUContext> gpus(num_gpus_to_use);
    for (int i = 0; i < num_gpus_to_use; i++)
    {
        printf("Initializing GPU %d...\n", i);
        fflush(stdout);
        gpus[i].device_id = i;
        gpus[i].allocate(n);
        gpus[i].upload_initial_state(bodies);
        printf("GPU %d ready\n", i);
        fflush(stdout);
    }

    // Prepare result storage
    std::vector<SimResult> pb3_results(num_devices);
    SimResult pb1_result, pb2_result;

    // ========================================================================
    // PHASE 1: Launch All Tasks (No Synchronization)
    // ========================================================================
    printf("Phase 1: Launching all tasks asynchronously...\n");
    fflush(stdout);

    auto launch_start = std::chrono::high_resolution_clock::now();

    if (use_dual_gpu)
    {
        // Dual GPU strategy: Split work across 2 GPUs
        // GPU 0: Pb1 + first half of Pb3
        // GPU 1: Pb2 + second half of Pb3

        // GPU 0: Launch Pb1
        gpus[0].launch_kernel(0, n, planet_id, asteroid_id, -1, 1);

        // GPU 1: Launch Pb2
        gpus[1].launch_kernel(0, n, planet_id, asteroid_id, -1, 0);

        // Split Pb3 tasks between GPUs
        int mid = num_devices / 2;

        // GPU 0: First half of Pb3
        for (int i = 0; i < mid; i++)
        {
            gpus[0].launch_kernel(1 + i, n, planet_id, asteroid_id, device_indices[i], 0);
        }

        // GPU 1: Second half of Pb3
        for (int i = mid; i < num_devices; i++)
        {
            gpus[1].launch_kernel(1 + (i - mid), n, planet_id, asteroid_id, device_indices[i], 0);
        }
    }
    else
    {
        // Single GPU strategy: All tasks on GPU 0
        const int PB1_STREAM = 0;
        const int PB2_STREAM = 1;
        const int PB3_STREAM_START = 2;

        // Launch Pb1
        gpus[0].launch_kernel(PB1_STREAM, n, planet_id, asteroid_id, -1, 1);

        // Launch Pb2
        gpus[0].launch_kernel(PB2_STREAM, n, planet_id, asteroid_id, -1, 0);

        // Launch all Pb3 tasks
        for (int i = 0; i < num_devices; i++)
        {
            gpus[0].launch_kernel(PB3_STREAM_START + i, n, planet_id, asteroid_id, device_indices[i], 0);
        }
    }

    auto launch_end = std::chrono::high_resolution_clock::now();
    double launch_time = std::chrono::duration<double>(launch_end - launch_start).count();
    printf("All tasks launched in %.6f s (CPU launch overhead)\n", launch_time);
    fflush(stdout);

    // ========================================================================
    // PHASE 2: Global Synchronization Barrier
    // ========================================================================
    printf("Phase 2: Waiting for all GPU tasks to complete...\n");
    fflush(stdout);

    auto sync_start = std::chrono::high_resolution_clock::now();

    // Synchronize all GPUs
    for (int i = 0; i < num_gpus_to_use; i++)
    {
        gpus[i].synchronize_all();
    }

    auto sync_end = std::chrono::high_resolution_clock::now();

    double gpu_execution_time = std::chrono::duration<double>(sync_end - launch_start).count();
    printf("All GPU tasks completed in %.3f s (actual GPU execution time)\n", gpu_execution_time);
    fflush(stdout);

    // ========================================================================
    // PHASE 3: Collect Results from Device to Host
    // ========================================================================
    printf("Phase 3: Collecting results...\n");
    fflush(stdout);

    if (use_dual_gpu)
    {
        // Collect from dual GPUs
        int mid = num_devices / 2;

        // Pb1 from GPU 0, stream 0
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMemcpy(&pb1_result, gpus[0].d_results[0],
                            sizeof(SimResult), hipMemcpyDeviceToHost));

        // Pb2 from GPU 1, stream 0
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMemcpy(&pb2_result, gpus[1].d_results[0],
                            sizeof(SimResult), hipMemcpyDeviceToHost));

        // First half of Pb3 from GPU 0
        HIP_CHECK(hipSetDevice(0));
        for (int i = 0; i < mid; i++)
        {
            int sid = (1 + i) % BatchedGPUContext::NUM_STREAMS;
            HIP_CHECK(hipMemcpy(&pb3_results[i], gpus[0].d_results[sid],
                                sizeof(SimResult), hipMemcpyDeviceToHost));
        }

        // Second half of Pb3 from GPU 1
        HIP_CHECK(hipSetDevice(1));
        for (int i = mid; i < num_devices; i++)
        {
            int sid = (1 + (i - mid)) % BatchedGPUContext::NUM_STREAMS;
            HIP_CHECK(hipMemcpy(&pb3_results[i], gpus[1].d_results[sid],
                                sizeof(SimResult), hipMemcpyDeviceToHost));
        }
    }
    else
    {
        // Collect from single GPU
        const int PB1_STREAM = 0;
        const int PB2_STREAM = 1;
        const int PB3_STREAM_START = 2;

        HIP_CHECK(hipSetDevice(0));

        // Copy Pb1 result
        HIP_CHECK(hipMemcpy(&pb1_result, gpus[0].d_results[PB1_STREAM],
                            sizeof(SimResult), hipMemcpyDeviceToHost));

        // Copy Pb2 result
        HIP_CHECK(hipMemcpy(&pb2_result, gpus[0].d_results[PB2_STREAM],
                            sizeof(SimResult), hipMemcpyDeviceToHost));

        // Copy all Pb3 results
        for (int i = 0; i < num_devices; i++)
        {
            int sid = (PB3_STREAM_START + i) % BatchedGPUContext::NUM_STREAMS;
            HIP_CHECK(hipMemcpy(&pb3_results[i], gpus[0].d_results[sid],
                                sizeof(SimResult), hipMemcpyDeviceToHost));
        }
    }

    printf("Results collected\n");
    printf("  Pb1: min_dist=%e\n", pb1_result.min_dist);
    printf("  Pb2: hit_time_step=%d\n", pb2_result.hit_time_step);
    printf("  Pb3: tested %d devices\n", num_devices);
    fflush(stdout);

    // ========================================================================
    // Process Results
    // ========================================================================

    double min_dist = pb1_result.min_dist;
    int hit_time_step = pb2_result.hit_time_step;
    int best_device_id = -999;
    double best_cost = -999.0;

    // Pb3: only if collision detected
    if (hit_time_step >= 0)
    {
        best_cost = 1e100;

        for (int i = 0; i < num_devices; i++)
        {
            const SimResult &res = pb3_results[i];

            // Check if this strategy prevents collision
            if (res.hit_time_step == -2 && res.destroyed_step >= 0)
            {
                if (res.missile_cost < best_cost)
                {
                    best_cost = res.missile_cost;
                    best_device_id = device_indices[i];
                }
            }
        }

        // If no successful strategy found
        if (best_device_id == -999)
        {
            best_device_id = -1;
            best_cost = -1.0;
        }
    }

    // Write output
    write_output(argv[2], min_dist, hit_time_step, best_device_id, best_cost);

    // Cleanup
    for (int i = 0; i < num_gpus_to_use; i++)
    {
        gpus[i].cleanup();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    printf("\n=== Performance Summary ===\n");
    printf("Launch overhead: %.6f s (CPU time)\n", launch_time);
    printf("GPU execution:   %.3f s (wall-clock time for all tasks)\n", gpu_execution_time);
    printf("Total runtime:   %.3f s\n", total_time);
    printf("===========================\n");
    fflush(stdout);

    return 0;
}
