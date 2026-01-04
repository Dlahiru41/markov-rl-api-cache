from src.utils.logging import setup_logger, MetricsLogger, log_cache_event, log_training_step


def main():
    setup_logger(level="DEBUG")
    ml = MetricsLogger()
    ml.log_metric("test", 1.0, step=0)
    ml.log_episode_summary(episode=1, reward=10.0, loss=0.5, epsilon=0.9)
    log_cache_event(hit=True, api="/api/v1/items", latency_ms=12.3)
    log_training_step(episode=1, step=2, action=0, reward=1.0)
    print("Metrics rows:", len(ml._rows))
    for r in ml._rows:
        print(r)
    # Export to logs/metrics.csv
    ml.export_csv("logs/metrics.csv")


if __name__ == "__main__":
    main()

