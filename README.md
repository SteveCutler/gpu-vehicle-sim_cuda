# GPU Vehicle Simulation — CUDA
A CUDA simulation of independent 2D vehicles, built to investigate how memory layout and timestep batching affect simulation throughput.
The project compares three GPU implementations against a serial C++ reference: array of structs (AoS), structure of arrays (SoA), and SoA with the entire rollout inside a single kernel. Each vehicle follows a fixed feedback controller through a procedural disturbance field.
Measured result: 1,000,000 vehicles × 100 timesteps in 0.269 s, versus 20.55 s for the serial CPU reference—approximately 76.4× faster on the project's Tesla T4 / Xeon test setup, including GPU allocation and transfer costs. See the timing scopes and limitations below.

<img width="494" height="481" alt="VehicleSimulator_screencap" src="https://github.com/user-attachments/assets/771777e4-f909-40cd-aa7b-9686d54e0f96" />

[C++ reference and visualization](https://github.com/SteveCutler/gpu-vehicle-sim_cpp) · [Experiment notebook](https://colab.research.google.com/drive/1Oha_W-K_uN2omVKECxBUDF-z6V5Aaw0-?usp=sharing)


Implementation highlights
- One CUDA thread per vehicle, with 256 threads per block and an index bounds check.
- Three selectable execution variants for comparing layout and launch strategy.
- Fixed-step integration of linear and angular motion under feedback control and drag.
- A deterministic, time-varying disturbance field evaluated independently by each thread.
- CUDA-event GPU-loop timing alongside host wall-clock timing.
- Final-state CSV export for comparison against the CPU reference.
This is simulation infrastructure for future robot-learning experiments. The current implementation uses a hand-written controller; it does not train a policy.
Build and run
Requires:
- An NVIDIA GPU with a compatible driver.
- NVIDIA CUDA Toolkit, including nvcc.
- A CUDA-compatible C++17 host compiler.
- CMake 3.18 or newer.
git clone https://github.com/SteveCutler/gpu-vehicle-sim_cuda.git
cd gpu-vehicle-sim_cuda

cmake -S . -B out/release -DCMAKE_BUILD_TYPE=Release -DENABLE_RENDERER=OFF
cmake --build out/release --parallel

# 10,000 vehicles, 1,000 timesteps, batched SoA
./out/release/vehicle_sim 10000 1000 2 states_cuda.csv
If needed, pass -DCMAKE_CUDA_ARCHITECTURES=<your-GPU-architecture> at configuration time. The examples use a single-configuration build on a Unix-like shell.
Command-line arguments
vehicle_sim <vehicle_count> <step_count> <variant> [output_csv]
Argument	Meaning
vehicle_count	Number of independent vehicles; use a positive integer
step_count	Number of fixed simulation timesteps; use a non-negative integer
variant	0 = AoS, 1 = SoA, 2 = batched SoA
output_csv	Optional path for final-state export


Always supply all three numeric arguments. The current entry point leaves the variant variable uninitialized when it is omitted. Use only the documented variant values; other integers currently fall through to AoS.
./out/release/vehicle_sim 100000 100 0 aos.csv
./out/release/vehicle_sim 100000 100 1 soa.csv
./out/release/vehicle_sim 100000 100 2 batched.csv
The CUDA entry point is headless. Although renderer files and a CMake option remain in this repository, rendering is not wired into its simulation loop. Use the CPU repository for visualization.
Execution strategies
Variant	Device layout	Launches per rollout	State access
0: AoS	Contiguous VehicleState structs	One per timestep	Read and write global state each timestep
1: SoA	Separate arrays for each state component	One per timestep	Read and write global state each timestep
2: Batched SoA	Separate component arrays	One	Load initial state, advance locally, write final state


AoS versus SoA isolates the layout choice. SoA places neighboring vehicles' values for the same component next to each other, supporting coalesced accesses across threads.
Batching changes both launch frequency and repeated global-state access. Each thread keeps its vehicle's working state local while stepping through time. This reduces launches and intermediate state traffic, but it also removes the opportunity for host-side intervention between timesteps.
Vehicles are independent; timesteps within a vehicle are sequential. No inter-vehicle synchronization is required by this model. Every variant performs O(N × S) total work.
Simulation model
Each vehicle contains position, velocity, heading, angular velocity, and a goal position. A fixed controller produces bounded thrust and torque using PD heading control and alignment-aware speed control.
The dynamics combine heading-aligned thrust, linear drag relative to local environmental flow, and rotational drag. Semi-implicit Euler updates velocities before positions and heading. The disturbance field is computed from position and simulation time using sinusoidal functions; it is not a separately simulated fluid.
Defaults set in the source:
Parameter	Value
World domain	500 × 500 units
Timestep	0.02 seconds
Initialization seed	25
Initial linear/angular velocity	Zero
Threads per block	256


Vehicles have no collisions or shared interactions. Flow is zero outside the domain; boundaries do not constrain vehicle motion.
Measured performance
The project report records experiments on Google Colab, using an NVIDIA Tesla T4 and a single-threaded Intel Xeon CPU reported at 2.20 GHz, GNU C++ 13.3.0, and NVIDIA compiler 13.0.88. Rendering was disabled. Values below are medians of five measured runs after a discarded warmup.
Host elapsed time: 1,000,000 vehicles × 100 timesteps
Implementation	Elapsed time
Serial CPU reference	20.55 s
CUDA SoA	0.270 s
CUDA batched SoA	0.269 s


Batched SoA is approximately 76.4× faster than the serial reference for this workload. Ordinary and batched SoA have nearly identical host elapsed times here, despite a large difference in their GPU-loop timings.
GPU-loop time: same workload
Implementation	CUDA-event elapsed time
AoS	102.53 ms
SoA	90.35 ms
Batched SoA	16.95 ms


SoA reduced GPU-loop time by 11.9% relative to AoS. Batching provided a further 5.33× GPU-loop speedup over ordinary SoA.
What the timers include
- CPU host time: the simulation update loop.
- CUDA host time: the simulation routine, including device allocation, transfers, kernel execution, synchronization, cleanup, and layout conversion where applicable.
- GPU-loop time: the CUDA-event interval around the simulation launches, after initial transfer and before copying results back.
Initial vehicle generation and final CSV export are outside the host timers. GPU-loop speedups must not be presented as end-to-end speedups.
At 100 vehicles × 100 timesteps, the CPU took 1.52 ms while CUDA variants took approximately 168–238 ms in host time. Small workloads did not amortize GPU overhead. These measurements describe the recorded setup, not a hardware-independent speedup.
Nsight Compute profiling examined occupancy, compute and memory throughput, cache behavior, and local-memory spilling. Per-timestep variants were profiled on a sampled launch; the batched launch covers the entire rollout, so individual launch durations are not equivalent units of work.
See the experiment notebook for the experiment workflow.
Numerical validation
Final states can be exported with the same schema as the CPU reference:
id,x,y,vx,vy,heading,turnRate,goalx,goaly
Run each implementation with matching vehicle counts and step counts, then compare IDs, goals, finite values, and state differences. Verify matching zero-step initialization before interpreting floating-point differences, particularly across different compiler/standard-library environments.
The project report documents checks for:
- Matching initialization at zero timesteps.
- One vehicle for one timestep.
- 255, 256, and 257 vehicles for 100 timesteps.
- 10,000 vehicles for one and 1,000 timesteps.
For 10,000 vehicles × 1,000 timesteps, all three CUDA variants produced the same reported error summaries relative to the CPU reference:
Quantity	Maximum reported difference
Position component	0.002655 world units
Velocity component	0.00173 world units/s
Heading	0.000123561 radians


These results demonstrate numerical agreement in the tested cases. They do not establish physical accuracy or an error bound for arbitrary inputs.
Code map
File	Responsibility
src/main.cpp	Argument parsing, variant selection, host timing, and export
src/CudaSimulationAoS.cu	AoS kernel and device-memory lifecycle
src/CudaSimulationSoA.cu	SoA kernel with one launch per timestep
src/CudaSimulationSoA_batched.cu	SoA kernel with the rollout inside each thread
src/controller.cu	Device-side feedback control
src/dynamics.cu	Device-side dynamics and integration
src/environment.cu	Device-side procedural disturbance sampling
src/vehicleBatch.cpp	Host initialization and component arrays
include/stateExport.hpp	Final-state CSV export


Limitations and next steps
- Simplified 2D dynamics and a fixed controller; no learned policy, sensors, rewards, or episode resets yet.
- No obstacles, vehicle collisions, or shared environmental state.
- Full-rollout batching assumes no external intervention between timesteps. Policy inference and intermediate observations will require a different execution schedule or shorter batches.
- Allocation and transfers occur for each simulation invocation; persistent device buffers are a future optimization.
- CLI validation and CUDA error handling/resource cleanup need further hardening.
Planned work includes lidar observations, obstacle avoidance, rewards, independent resets, and PPO integration, followed by measurements of the complete observation–action–simulation pipeline.
