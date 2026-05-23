/*
 * pso_kernel.cu
 * ─────────────
 * CUDA kernel for parallel PSO fitness evaluation.
 * Each thread evaluates the fitness of ONE particle.
 *
 * Compile (standalone test):
 *   nvcc -o pso_kernel pso_kernel.cu
 *
 * This file is loaded at runtime by pso_cuda.py via PyCUDA SourceModule.
 */

#include <math.h>

/*
 * evaluate_fitness
 * ────────────────
 * Grid  : ceil(N_PARTICLES / BLOCK_SIZE) blocks
 * Block : BLOCK_SIZE threads  (recommended: 64)
 *
 * Each thread (pid) evaluates one particle's threshold vector against
 * the entire sensor dataset and writes its fitness to fitness_out[pid].
 */
__global__ void evaluate_fitness(
    float *positions,     /* [N_PARTICLES * N_DIMS]  particle positions   */
    float *X_data,        /* [N_SAMPLES   * N_DIMS]  sensor readings      */
    int   *y_labels,      /* [N_SAMPLES]             ground-truth labels  */
    float *fitness_out,   /* [N_PARTICLES]            output fitness       */
    int    n_particles,
    int    n_dims,
    int    n_samples,
    float  penalty        /* false-positive penalty weight (e.g. 0.3)     */
)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    float *thresholds = &positions[pid * n_dims];

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < n_samples; i++) {
        float *sample = &X_data[i * n_dims];
        int    label  = y_labels[i];
        int    y_bin  = (label > 0) ? 1 : 0;

        /* Predict anomaly if ANY feature exceeds its threshold */
        int pred = 0;
        for (int d = 0; d < n_dims; d++) {
            if (sample[d] > thresholds[d]) {
                pred = 1;
                break;
            }
        }

        if      (pred == 1 && y_bin == 1) tp++;
        else if (pred == 0 && y_bin == 0) tn++;
        else if (pred == 1 && y_bin == 0) fp++;
        else                               fn++;
    }

    float accuracy = (float)(tp + tn) / (float)n_samples;
    float fpr      = (float)fp / ((float)(fp + tn) + 1e-9f);
    float fit      = accuracy - penalty * fpr;

    /* Clamp fitness to [0, 1] */
    fitness_out[pid] = fmaxf(0.0f, fminf(1.0f, fit));
}


/*
 * velocity_update  (optional helper kernel)
 * ──────────────────────────────────────────
 * Parallelises velocity and position update on GPU.
 * Can be called instead of doing the update on CPU.
 */
__global__ void velocity_update(
    float *positions,
    float *velocities,
    float *personal_best,
    float *global_best,
    float *r1,
    float *r2,
    int    n_particles,
    int    n_dims,
    float  w,            /* inertia weight          */
    float  c1,           /* cognitive coefficient   */
    float  c2,           /* social coefficient      */
    float  v_max
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles * n_dims) return;

    int d   = idx % n_dims;   /* dimension index  */

    float v = w  * velocities[idx]
            + c1 * r1[idx] * (personal_best[idx] - positions[idx])
            + c2 * r2[idx] * (global_best[d]     - positions[idx]);

    /* Clamp velocity */
    v = fmaxf(-v_max, fminf(v_max, v));

    velocities[idx] = v;

    /* Update position and clamp to [0, 1] */
    float pos = positions[idx] + v;
    positions[idx] = fmaxf(0.0f, fminf(1.0f, pos));
}
