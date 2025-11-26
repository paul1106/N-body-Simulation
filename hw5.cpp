#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

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
    int padding;       // 手动补齐，凑满 64 bytes，避免 GPU 内存访问问题
};

struct SimResult
{
    double min_dist;
    int hit_time_step;
    int destroyed_step;
    double missile_cost;
};

// ============================================================================
// HIP Kernel - Single Block, All Steps Inside
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
                }
            }
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
// GPU Memory Management
// ============================================================================
struct GPUContext
{
    int device_id;
    Body *d_backup;      // Backup of initial state
    Body *d_working;     // Working buffer
    SimResult *d_result; // Result buffer
    hipStream_t stream;

    void allocate(int n)
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc((void **)&d_backup, n * sizeof(Body)));
        HIP_CHECK(hipMalloc((void **)&d_working, n * sizeof(Body)));
        HIP_CHECK(hipMalloc((void **)&d_result, sizeof(SimResult)));
        HIP_CHECK(hipStreamCreate(&stream));
    }

    void upload_initial_state(const std::vector<Body> &bodies)
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpy(d_backup, bodies.data(), bodies.size() * sizeof(Body),
                            hipMemcpyHostToDevice));
    }

    void reset_working_state(int n)
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpyAsync(d_working, d_backup, n * sizeof(Body),
                                 hipMemcpyDeviceToDevice, stream));
    }

    void launch_kernel(int n, int planet_id, int asteroid_id,
                       int target_device_id, int pb1_mode)
    {
        HIP_CHECK(hipSetDevice(device_id));
        reset_working_state(n);

        printf("  Launching kernel: n=%d, planet=%d, asteroid=%d, target=%d, pb1=%d\n",
               n, planet_id, asteroid_id, target_device_id, pb1_mode);
        fflush(stdout);

        // Always use 1024 threads to match shared memory size (strategy for N <= 1024)
        // Threads with tid >= n will be inactive
        nbody_kernel<<<1, 1024, 0, stream>>>(d_working, n, planet_id, asteroid_id,
                                             target_device_id, pb1_mode, d_result);
        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(launch_err));
            exit(EXIT_FAILURE);
        }

        printf("  Kernel launched successfully\n");
        fflush(stdout);
    }

    SimResult get_result()
    {
        HIP_CHECK(hipSetDevice(device_id));
        SimResult result;
        HIP_CHECK(hipMemcpy(&result, d_result, sizeof(SimResult), hipMemcpyDeviceToHost));
        return result;
    }

    void synchronize()
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    void cleanup()
    {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipFree(d_backup));
        HIP_CHECK(hipFree(d_working));
        HIP_CHECK(hipFree(d_result));
        HIP_CHECK(hipStreamDestroy(stream));
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

    // Initialize GPU context (single GPU)
    printf("Initializing GPU 0...\n");
    fflush(stdout);
    GPUContext gpu0;
    gpu0.device_id = 0;

    printf("Allocating GPU memory...\n");
    fflush(stdout);
    gpu0.allocate(n);
    printf("GPU 0 allocated\n");
    fflush(stdout);

    printf("Uploading initial state...\n");
    fflush(stdout);
    gpu0.upload_initial_state(bodies);
    printf("GPU 0 uploaded\n");
    fflush(stdout);

    // ========================================================================
    // Sequential Execution on Single GPU
    // ========================================================================

    printf("Launching Pb1 on GPU 0...\n");
    fflush(stdout);
    auto pb1_start = std::chrono::high_resolution_clock::now();
    // Pb1: compute min distance (all devices have mass 0)
    gpu0.launch_kernel(n, planet_id, asteroid_id, -1, 1); // pb1_mode = 1
    gpu0.synchronize();
    SimResult pb1_result = gpu0.get_result();
    auto pb1_end = std::chrono::high_resolution_clock::now();
    double pb1_time = std::chrono::duration<double>(pb1_end - pb1_start).count();
    printf("Pb1 completed: min_dist=%e (Time: %.3f s)\n", pb1_result.min_dist, pb1_time);
    fflush(stdout);

    printf("Launching Pb2 on GPU 0...\n");
    fflush(stdout);
    auto pb2_start = std::chrono::high_resolution_clock::now();
    // Pb2: detect collision (all devices active)
    gpu0.launch_kernel(n, planet_id, asteroid_id, -1, 0); // pb1_mode = 0
    gpu0.synchronize();
    SimResult pb2_result = gpu0.get_result();
    auto pb2_end = std::chrono::high_resolution_clock::now();
    double pb2_time = std::chrono::duration<double>(pb2_end - pb2_start).count();
    printf("Pb2 completed: hit_time_step=%d (Time: %.3f s)\n", pb2_result.hit_time_step, pb2_time);
    fflush(stdout);

    // Prepare Pb3 results storage
    std::vector<SimResult> pb3_results(num_devices);

    // Pb3: test each device destruction strategy
    printf("Launching Pb3 for %d devices on GPU 0...\n", num_devices);
    fflush(stdout);
    auto pb3_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_devices; i++)
    {
        printf("  Testing device %d (index %d)...\n", i, device_indices[i]);
        fflush(stdout);
        gpu0.launch_kernel(n, planet_id, asteroid_id, device_indices[i], 0);
        gpu0.synchronize();
        pb3_results[i] = gpu0.get_result();
    }
    auto pb3_end = std::chrono::high_resolution_clock::now();
    double pb3_time = std::chrono::duration<double>(pb3_end - pb3_start).count();
    printf("Pb3 completed (Time: %.3f s)\n", pb3_time);
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
    gpu0.cleanup();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    printf("\n=== Timing Summary ===\n");
    printf("Pb1 time: %.3f s\n", pb1_time);
    printf("Pb2 time: %.3f s\n", pb2_time);
    printf("Pb3 time: %.3f s\n", pb3_time);
    printf("Total time: %.3f s\n", total_time);
    printf("======================\n");
    fflush(stdout);

    return 0;
}
