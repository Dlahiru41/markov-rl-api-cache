"""
Standalone Prometheus metrics exporter for the Markov-RL API Cache.

This module:
  1. Creates a MetricsCollector and starts the HTTP /metrics server.
  2. Patches the IntegrationController's training/eval loops so every
     episode, training step, and Markov prediction is automatically
     recorded.
  3. Exposes a /metrics endpoint on the FastAPI REST API as well
     (generates latest Prometheus exposition format).

Run directly::

    python -m src.monitoring.exporter --port 9200 --api-port 8080

Or import and call ``setup_metrics(controller)`` to attach metrics
to an existing IntegrationController instance.
"""

import argparse
import logging
import threading
import time
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attach metrics to an IntegrationController
# ---------------------------------------------------------------------------

def setup_metrics(controller, metrics_port: int = 9200):
    """
    Attach a MetricsCollector to *controller* and start the HTTP server.

    This function monkey-patches the controller's Trainer callbacks and
    environment step recording so that all metrics are captured without
    requiring changes to the core training code.

    Args:
        controller:    An initialised IntegrationController instance.
        metrics_port:  TCP port for the Prometheus /metrics HTTP endpoint.

    Returns:
        The created MetricsCollector instance.
    """
    from src.monitoring.metrics import MetricsCollector, start_metrics_server

    collector = MetricsCollector(service="api-cache")
    start_metrics_server(port=metrics_port, registry=collector.registry)

    # ── Patch Trainer callbacks ──────────────────────────────────────────
    if controller.trainer is not None:
        _patch_trainer(controller.trainer, collector)

    # ── Patch Environment step ───────────────────────────────────────────
    if controller.env is not None:
        _patch_env(controller.env, collector)

    # ── Background thread: poll controller.get_metrics() every 10 s ─────
    _start_poll_thread(controller, collector)

    # ── Expose metrics via FastAPI if API is enabled ──────────────────────
    if controller.api_server is not None:
        _add_metrics_endpoint(controller.api_server, collector)

    # Store on the public attribute (IntegrationController already declares it)
    controller.metrics_collector = collector
    logger.info(f"Prometheus metrics exporter attached (port={metrics_port})")
    return collector


def _patch_trainer(trainer, collector):
    """Wrap Trainer._run_episode to capture per-episode metrics."""
    original_run_episode = getattr(trainer, "_run_episode", None)
    if original_run_episode is None:
        logger.debug("Trainer._run_episode not found – skipping patch")
        return

    def patched_run_episode(*args, **kwargs):
        result = original_run_episode(*args, **kwargs)
        # result is expected to be a dict with keys:
        #   reward, length, cache_hit_rate, cascade_occurred, [epsilon, loss]
        if isinstance(result, dict):
            collector.record_episode(
                reward=result.get("reward", result.get("episode_reward", 0.0)),
                length=result.get("length", result.get("episode_length", 0)),
                hit_rate=result.get("cache_hit_rate", 0.0),
                cascade_occurred=result.get("cascade_occurred", False),
                cascade_prevented=result.get("cascade_prevented", False),
                reward_breakdown=result.get("reward_breakdown"),
            )
            if "epsilon" in result:
                collector.update_epsilon(result["epsilon"])
            if "loss" in result and result["loss"] is not None:
                collector.record_training_step(
                    loss=result["loss"],
                    epsilon=result.get("epsilon", 0.0),
                    q_mean=result.get("q_mean"),
                    buffer_size=result.get("buffer_size"),
                )
            if "cascade_risk" in result:
                collector.update_cascade_risk(result["cascade_risk"])
        return result

    trainer._run_episode = patched_run_episode
    logger.debug("Trainer._run_episode patched for metrics collection")


def _patch_env(env, collector):
    """Wrap CachingEnv.step to capture per-step metrics."""
    original_step = env.step

    def patched_step(action):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = original_step(action)
        latency = time.perf_counter() - t0  # noqa: F841  (available for future use)

        # Action name
        try:
            from src.rl.actions import CacheAction
            action_name = CacheAction.get_name(action)
        except Exception:
            action_name = str(action)

        collector.record_env_step(action_name=action_name)

        # Record hit/miss from info dict
        if info.get("cache_hit"):
            collector.record_cache_hit(endpoint=info.get("endpoint", "unknown"))
        elif info.get("cache_miss"):
            collector.record_cache_miss(endpoint=info.get("endpoint", "unknown"))

        # Cascade risk
        if "cascade_risk" in info:
            collector.update_cascade_risk(info["cascade_risk"])

        return obs, reward, terminated, truncated, info

    env.step = patched_step
    logger.debug("CachingEnv.step patched for metrics collection")


def _start_poll_thread(controller, collector):
    """Start a background daemon thread that polls controller metrics."""

    def _poll():
        while True:
            try:
                metrics = controller.get_metrics()
                collector.update_from_metrics_dict(metrics)

                # System-level (optional – requires psutil)
                try:
                    import psutil
                    collector.update_system_metrics(
                        cpu=psutil.cpu_percent(interval=None) / 100.0,
                        memory=psutil.virtual_memory().percent / 100.0,
                    )
                except ImportError:
                    psutil = None  # type: ignore[assignment]

            except Exception as exc:
                logger.debug(f"Metrics poll error (non-fatal): {exc}")
            time.sleep(10)

    t = threading.Thread(target=_poll, daemon=True, name="metrics-poll")
    t.start()
    logger.debug("Metrics background poll thread started")


def _add_metrics_endpoint(app, collector):
    """Add GET /metrics (Prometheus exposition) to the FastAPI app."""
    try:
        from fastapi import Response
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics():
            data = generate_latest(collector.registry)
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)

        logger.info("Added GET /metrics endpoint to FastAPI app")
    except Exception as exc:
        logger.warning(f"Could not add /metrics endpoint: {exc}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Markov-RL Prometheus metrics exporter"
    )
    parser.add_argument(
        "--port", type=int, default=9200,
        help="Prometheus /metrics HTTP port (default: 9200)"
    )
    parser.add_argument(
        "--api-port", type=int, default=8080,
        help="FastAPI control API port (default: 8080)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to controller YAML config (optional)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Add project root to sys.path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.integration.controller import IntegrationController, ControllerConfig
        from src.integration.gym_environment import CacheEnvConfig
    except ImportError as exc:
        IntegrationController = None   # type: ignore[assignment,misc]
        ControllerConfig = None        # type: ignore[assignment,misc]
        CacheEnvConfig = None          # type: ignore[assignment,misc]
        logger.error(f"Failed to import controller: {exc}")
        sys.exit(1)

    # Build config
    config = ControllerConfig(
        mode="training",
        enable_monitoring=True,
        enable_api=True,
        api_port=args.api_port,
        output_dir="results/metrics_export",
        env_config=CacheEnvConfig(),
    )

    controller = IntegrationController(config)
    if not controller.setup():
        logger.error("Controller setup failed")
        sys.exit(1)

    # Attach metrics collector
    setup_metrics(controller, metrics_port=args.port)

    # Start controller
    controller.start()

    logger.info(
        f"Markov-RL metrics exporter running\n"
        f"  Prometheus: http://0.0.0.0:{args.port}/metrics\n"
        f"  REST API:   http://0.0.0.0:{args.api_port}\n"
        f"  Press Ctrl+C to stop."
    )

    # Run training in background so metrics populate
    def _train():
        try:
            controller.train()
        except Exception as exc:
            logger.error(f"Training error: {exc}")

    train_thread = threading.Thread(target=_train, daemon=True, name="training")
    train_thread.start()

    # If FastAPI is set up, serve it (blocks)
    if controller.api_server is not None:
        try:
            import uvicorn
            uvicorn.run(
                controller.api_server,
                host="0.0.0.0",
                port=args.api_port,
                log_level=args.log_level.lower(),
            )
        except ImportError:
            uvicorn = None  # type: ignore[assignment]
            logger.warning("uvicorn not installed – API not served, running metrics-only mode")
            _block_until_interrupted()
    else:
        _block_until_interrupted()

    controller.stop()
    logger.info("Exporter stopped.")


def _block_until_interrupted():
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

