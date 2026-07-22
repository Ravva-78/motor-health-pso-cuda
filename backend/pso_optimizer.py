"""
pso_optimizer.py
────────────────
Particle Swarm Optimization for Random Forest hyperparameter tuning.

Each particle encodes 3 hyperparameters:
  [n_estimators, max_depth, min_samples_split]

Fitness = classification accuracy evaluated by CUDA kernel (or CPU fallback).
PSO uses standard velocity update:
  v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
  x = x + v   (clamped to bounds)
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from backend.cuda_module import gpu_accuracy
from backend.logger import get_logger
from backend.constants import PSO_DEFAULT_BOUNDS

logger = get_logger(__name__)


class PSOOptimizer:
    """
    PSO for Random Forest hyperparameter optimisation.

    Parameters
    ----------
    X, y         : training data
    n_particles  : swarm size
    max_iter     : number of PSO iterations
    w            : inertia weight
    c1           : cognitive coefficient
    c2           : social coefficient
    use_gpu      : if True, use CUDA kernel for fitness; else CPU
    """

    BOUNDS = PSO_DEFAULT_BOUNDS
    KEYS = list(BOUNDS.keys())
    DIM  = len(KEYS)

    def __init__(self, X, y, n_particles=20, max_iter=20,
                 w=0.7, c1=1.5, c2=1.5, use_gpu=True):
        self.X           = X
        self.y           = y
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.use_gpu     = use_gpu
        self.history     = []    # best accuracy per iteration
        self.best_params = None
        self.best_score  = 0.0

    # ── Initialise swarm ───────────────────────────────────────────────────
    def _init_swarm(self):
        lo = np.array([v[0] for v in self.BOUNDS.values()], dtype=float)
        hi = np.array([v[1] for v in self.BOUNDS.values()], dtype=float)
        pos = lo + np.random.uniform(0, 1, (self.n_particles, self.DIM)) * (hi - lo)
        vel = np.random.uniform(-0.5, 0.5, (self.n_particles, self.DIM))
        return pos, vel, lo, hi

    # ── Decode float vector → RF hyperparams ──────────────────────────────
    def _decode(self, pos):
        return {
            'n_estimators':      max(10, int(round(pos[0]))),
            'max_depth':         max(3,  int(round(pos[1]))),
            'min_samples_split': max(2,  int(round(pos[2]))),
        }

    # ── Evaluate particle fitness ─────────────────────────────────────────
    def _fitness(self, pos) -> float:
        params = self._decode(pos)
        Xtr, Xv, ytr, yv = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y
        )
        m = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight='balanced')
        m.fit(Xtr, ytr)
        preds = m.predict(Xv)

        if self.use_gpu:
            return gpu_accuracy(yv.astype(np.int32), preds.astype(np.int32))
        else:
            return float(np.mean(yv == preds))

    # ── Main PSO loop ──────────────────────────────────────────────────────
    def run(self, verbose=True):
        pos, vel, lo, hi = self._init_swarm()

        fit     = np.array([self._fitness(p) for p in pos])
        pbest   = pos.copy()
        pbest_f = fit.copy()
        gbest   = pbest[np.argmax(pbest_f)].copy()
        gbest_f = pbest_f.max()

        mode = "GPU (CUDA)" if self.use_gpu else "CPU"
        if verbose:
            logger.info(f'PSO Optimisation [{mode}] started')

        start_time = time.perf_counter()

        for it in range(self.max_iter):
            r1  = np.random.rand(self.n_particles, self.DIM)
            r2  = np.random.rand(self.n_particles, self.DIM)
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, lo, hi)

            fit = np.array([self._fitness(p) for p in pos])

            imp = fit > pbest_f
            pbest[imp]   = pos[imp]
            pbest_f[imp] = fit[imp]

            if pbest_f.max() > gbest_f:
                gbest_f = pbest_f.max()
                gbest   = pbest[np.argmax(pbest_f)].copy()

            self.history.append(float(gbest_f))

            if verbose:
                logger.info(f'Iter {it+1:02d}/{self.max_iter} | Best Accuracy [{mode}]: {gbest_f:.4f} | Params: {self._decode(gbest)}')

        elapsed = time.perf_counter() - start_time
        self.best_params = self._decode(gbest)
        self.best_score  = float(gbest_f)
        self.elapsed     = elapsed

        if verbose:
            logger.info(f'PSO done in {elapsed:.2f}s')
            logger.info(f'Best params : {self.best_params}')
            logger.info(f'Best val acc: {self.best_score:.4f}')

        return self.best_params
