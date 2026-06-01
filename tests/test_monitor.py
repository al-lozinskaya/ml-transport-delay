from src.monitor import SystemMonitor


def test_system_monitor_summary_aggregates_samples():
    monitor = SystemMonitor()
    monitor.samples = [
        {"cpu_percent": 10.0, "memory_percent": 40.0},
        {"cpu_percent": 30.0, "memory_percent": 60.0},
    ]

    summary = monitor.summary()

    assert summary == {
        "cpu_percent_mean": 20.0,
        "cpu_percent_max": 30.0,
        "memory_percent_mean": 50.0,
        "memory_percent_max": 60.0,
    }


def test_system_monitor_summary_returns_zeroes_without_samples():
    summary = SystemMonitor().summary()

    assert summary == {
        "cpu_percent_mean": 0.0,
        "cpu_percent_max": 0.0,
        "memory_percent_mean": 0.0,
        "memory_percent_max": 0.0,
    }
