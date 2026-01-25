"""
Traffic Generator - Practical Usage Example

Demonstrates how to use the traffic generator with e-commerce services.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulator.traffic.generator import TrafficGenerator, TrafficProfile


async def example_basic_usage():
    """Basic usage example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70 + "\n")

    # Load profile
    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')
    print(f"Loaded profile: {profile.name}")
    print(f"  - Target rate: {profile.requests_per_second} RPS")
    print(f"  - Duration: {profile.duration_seconds}s")
    print()

    # Override for demo (shorter duration)
    profile.duration_seconds = 15
    profile.requests_per_second = 10

    # Configure service URLs
    service_urls = {
        'user': 'http://localhost:8001',
        'auth': 'http://localhost:8002',
        'product': 'http://localhost:8003',
        'cart': 'http://localhost:8004',
        'order': 'http://localhost:8005',
        'payment': 'http://localhost:8006',
        'inventory': 'http://localhost:8007',
    }

    # Create generator
    generator = TrafficGenerator(profile, service_urls)
    print("Generator created")
    print()

    # Start generating traffic
    print("Starting traffic generation...")
    await generator.start()

    # Monitor progress
    print("\nMonitoring (services may not be running - errors are expected):")
    for i in range(15):
        await asyncio.sleep(1)
        stats = generator.get_stats()
        print(f"  [{i+1:2d}s] Requests: {stats['total']:3d}, "
              f"Active users: {stats['active_users']:2d}, "
              f"Success: {stats['success_rate']:5.1%}")

    # Get final statistics
    print("\nFinal Statistics:")
    stats = generator.get_stats()
    print(f"  Total requests: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    if stats['total'] > 0:
        print(f"  Avg latency: {stats['latency_mean']:.0f}ms")
        print(f"  p95 latency: {stats['latency_p95']:.0f}ms")
        print(f"  Throughput: {stats['throughput_rps']:.1f} RPS")

    print("\n  Requests by workflow:")
    for workflow, count in sorted(stats['requests_by_workflow'].items()):
        print(f"    - {workflow}: {count}")

    # Stop generator
    await generator.stop()
    print("\n[OK] Generator stopped\n")


async def example_different_profiles():
    """Compare different traffic profiles."""
    print("=" * 70)
    print("EXAMPLE 2: Different Traffic Profiles")
    print("=" * 70 + "\n")

    profiles = [
        ('normal', 'Typical weekday traffic'),
        ('peak', 'Evening/weekend high traffic'),
        ('degraded', 'Testing with slow services'),
        ('burst', 'Flash sale spike'),
    ]

    for profile_name, description in profiles:
        profile = TrafficProfile.from_yaml(f'simulator/traffic/profiles/{profile_name}.yaml')

        print(f"📊 {profile.name.upper()}: {description}")
        print(f"   Rate: {profile.requests_per_second} RPS")
        print(f"   Duration: {profile.duration_seconds}s")
        print(f"   Ramp: {profile.ramp_up_seconds}s up, {profile.ramp_down_seconds}s down")
        print(f"   User mix: Premium {profile.user_type_distribution.get('premium', 0):.0%}, "
              f"Free {profile.user_type_distribution.get('free', 0):.0%}, "
              f"Guest {profile.user_type_distribution.get('guest', 0):.0%}")
        print(f"   Top workflow: {max(profile.workflow_distribution.items(), key=lambda x: x[1])[0]}")
        print()


async def example_monitoring():
    """Example with detailed monitoring."""
    print("=" * 70)
    print("EXAMPLE 3: Monitoring Traffic Generation")
    print("=" * 70 + "\n")

    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')
    profile.duration_seconds = 20
    profile.requests_per_second = 20
    profile.ramp_up_seconds = 5
    profile.ramp_down_seconds = 5

    service_urls = {
        'user': 'http://localhost:8001',
        'product': 'http://localhost:8003',
        'cart': 'http://localhost:8004',
    }

    generator = TrafficGenerator(profile, service_urls)

    print("Starting with ramp-up/down...")
    await generator.start()

    print("\nReal-time monitoring:")
    print(f"{'Time':<6} {'RPS':<8} {'Total':<8} {'Success':<8} {'Latency p95':<12}")
    print("-" * 50)

    last_total = 0
    for i in range(20):
        await asyncio.sleep(1)
        stats = generator.get_stats()

        # Calculate current RPS
        current_rps = stats['total'] - last_total
        last_total = stats['total']

        print(f"{i+1:4d}s  {current_rps:6d}   {stats['total']:6d}   "
              f"{stats['success_rate']:6.1%}   {stats['latency_p95']:8.0f}ms")

    print()
    await generator.stop()
    print("[OK] Monitoring complete\n")


async def example_pause_resume():
    """Example demonstrating pause/resume functionality."""
    print("=" * 70)
    print("EXAMPLE 4: Pause and Resume Control")
    print("=" * 70 + "\n")

    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')
    profile.duration_seconds = 30
    profile.requests_per_second = 15

    service_urls = {'user': 'http://localhost:8001'}
    generator = TrafficGenerator(profile, service_urls)

    print("Starting traffic...")
    await generator.start()

    # Run for 5 seconds
    print("Running normally...")
    for i in range(5):
        await asyncio.sleep(1)
        stats = generator.get_stats()
        print(f"  {i+1}s: {stats['total']} requests")

    # Pause
    print("\n|| Pausing traffic...")
    generator.pause()
    requests_at_pause = generator.get_stats()['total']

    for i in range(3):
        await asyncio.sleep(1)
        stats = generator.get_stats()
        print(f"  While paused: {stats['total']} requests (should stay at {requests_at_pause})")

    # Resume
    print("\n> Resuming traffic...")
    generator.resume()

    for i in range(5):
        await asyncio.sleep(1)
        stats = generator.get_stats()
        print(f"  {i+1}s after resume: {stats['total']} requests")

    await generator.stop()
    print("\n[OK] Pause/resume demo complete\n")


async def example_workflow_analysis():
    """Analyze workflow distribution."""
    print("=" * 70)
    print("EXAMPLE 5: Workflow Distribution Analysis")
    print("=" * 70 + "\n")

    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/peak.yaml')
    profile.duration_seconds = 15
    profile.requests_per_second = 30

    service_urls = {
        'user': 'http://localhost:8001',
        'product': 'http://localhost:8003',
        'cart': 'http://localhost:8004',
        'order': 'http://localhost:8005',
    }

    generator = TrafficGenerator(profile, service_urls)

    print(f"Profile: {profile.name}")
    print(f"Expected workflow distribution:")
    for workflow, fraction in sorted(profile.workflow_distribution.items()):
        print(f"  - {workflow}: {fraction:.0%}")
    print()

    print("Generating traffic...")
    await generator.start()

    # Let it run
    await asyncio.sleep(15)

    # Analyze results
    stats = generator.get_stats()
    print("\nActual workflow distribution:")
    total = sum(stats['requests_by_workflow'].values())
    for workflow, count in sorted(stats['requests_by_workflow'].items()):
        actual_pct = count / total if total > 0 else 0
        expected_pct = profile.workflow_distribution.get(workflow, 0)
        diff = actual_pct - expected_pct
        print(f"  - {workflow}: {actual_pct:.0%} (expected {expected_pct:.0%}, "
              f"diff: {diff:+.0%})")

    await generator.stop()
    print("\n[OK] Workflow analysis complete\n")


async def main():
    """Run all examples."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 15 + "TRAFFIC GENERATOR EXAMPLES" + " " * 27 + "|")
    print("=" + "=" * 68 + "╝")

    print("\nNOTE: These examples will attempt to connect to e-commerce services.")
    print("If services are not running, you will see connection errors.")
    print("This is expected - the generator handles errors gracefully.")
    print()

    try:
        await example_basic_usage()
        await example_different_profiles()
        await example_monitoring()
        await example_pause_resume()
        await example_workflow_analysis()

        print("=" * 70)
        print("[SUCCESS] ALL EXAMPLES COMPLETED")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  1. Load profiles from YAML files")
        print("  2. Configure service URLs")
        print("  3. Start generator and monitor progress")
        print("  4. Use pause/resume for control")
        print("  5. Analyze statistics and workflow distribution")
        print()
        print("To use with live services:")
        print("  1. Start e-commerce services:")
        print("     python -m simulator.services.ecommerce.orchestrator")
        print("  2. Run this script again")
        print("  3. Watch traffic flow to services!")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

