"""Particle Swarm Optimization (PSO) for Autoencoder Hyperparameter Tuning."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Tuple
import numpy as np

from training.metaheuristics.search_space import (
    DIMENSION,
    Hyperparameters,
    decode_vector,
)

logger = logging.getLogger("pso")


def run_pso(
    eval_fn: Callable[[Hyperparameters], Tuple[float, Dict]],
    n_particles: int = 6,
    n_iterations: int = 5,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Run Particle Swarm Optimization over the continuous [0, 1]^D search space."""
    rng = np.random.default_rng(seed)
    dim = DIMENSION

    # Initialize positions in [0, 1] and velocities
    positions = rng.uniform(0.05, 0.95, size=(n_particles, dim))
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, dim))

    # Evaluate initial swarm
    pbests = positions.copy()
    pbest_fitnesses = np.full(n_particles, -float("inf"))
    eval_history: List[Dict] = []
    iteration_history: List[Dict] = []

    gbest_pos = None
    gbest_fitness = -float("inf")
    gbest_hp = None

    eval_count = 0
    start_time = time.perf_counter()

    logger.info("Initializing PSO swarm (N=%d, T=%d, Dim=%d)...", n_particles, n_iterations, dim)

    for i in range(n_particles):
        eval_count += 1
        pos = positions[i]
        hp = decode_vector(pos)
        fitness, info = eval_fn(hp)

        pbest_fitnesses[i] = fitness
        pbests[i] = pos.copy()

        if fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest_pos = pos.copy()
            gbest_hp = hp

        eval_history.append({
            "eval_id": eval_count,
            "iteration": 0,
            "agent_idx": i,
            "position": pos.tolist(),
            "hyperparameters": hp.to_dict(),
            "fitness": float(fitness),
            "gbest_fitness": float(gbest_fitness),
            "info": info,
        })
        logger.info(
            "PSO Init [%d/%d] | val_AUPRC=%.5f | gbest=%.5f | lr=%.5f, h=%s, latent=%d",
            i + 1, n_particles, fitness, gbest_fitness, hp.lr, hp.hidden_dims, hp.latent_dim
        )

    iteration_history.append({
        "iteration": 0,
        "best_fitness": float(gbest_fitness),
        "mean_fitness": float(np.mean(pbest_fitnesses)),
    })

    # Optimization loop
    for it in range(1, n_iterations + 1):
        w = w_max - (it / n_iterations) * (w_max - w_min)
        iter_fitnesses = []

        for i in range(n_particles):
            # Update velocity
            r1 = rng.uniform(0, 1, size=dim)
            r2 = rng.uniform(0, 1, size=dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbests[i] - positions[i])
                + c2 * r2 * (gbest_pos - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Update position
            positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)

            # Evaluate candidate
            eval_count += 1
            pos = positions[i]
            hp = decode_vector(pos)
            fitness, info = eval_fn(hp)
            iter_fitnesses.append(fitness)

            # Update personal best
            if fitness > pbest_fitnesses[i]:
                pbest_fitnesses[i] = fitness
                pbests[i] = pos.copy()

            # Update global best
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_pos = pos.copy()
                gbest_hp = hp

            eval_history.append({
                "eval_id": eval_count,
                "iteration": it,
                "agent_idx": i,
                "position": pos.tolist(),
                "hyperparameters": hp.to_dict(),
                "fitness": float(fitness),
                "gbest_fitness": float(gbest_fitness),
                "info": info,
            })

            logger.info(
                "PSO Iter %d [%d/%d] | val_AUPRC=%.5f | gbest=%.5f",
                it, i + 1, n_particles, fitness, gbest_fitness
            )

        iteration_history.append({
            "iteration": it,
            "best_fitness": float(gbest_fitness),
            "mean_fitness": float(np.mean(iter_fitnesses)),
        })

    elapsed_time = time.perf_counter() - start_time
    logger.info("PSO finished in %.2fs. Best val_AUPRC=%.5f", elapsed_time, gbest_fitness)

    return {
        "optimizer": "PSO",
        "best_position": gbest_pos.tolist(),
        "best_hyperparameters": gbest_hp.to_dict(),
        "best_fitness": float(gbest_fitness),
        "total_evaluations": eval_count,
        "n_particles": n_particles,
        "n_iterations": n_iterations,
        "elapsed_time_s": float(elapsed_time),
        "eval_history": eval_history,
        "iteration_history": iteration_history,
    }
