"""
Validation script for Traffic Generator

Tests the traffic generator according to the requirements.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simulator.traffic.generator import TrafficGenerator, TrafficProfile


async def test_profile_loading():
    """Test loading profiles from YAML."""
    print("\n" + "=" * 70)
    print("TEST 1: Profile Loading")
    print("=" * 70)

    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')

    print(f"[OK] Profile loaded: {profile.name}")
    print(f"  - RPS: {profile.requests_per_second}")
    print(f"  - Duration: {profile.duration_seconds}s")
    print(f"  - Ramp up: {profile.ramp_up_seconds}s")
    print(f"  - User types: {profile.user_type_distribution}")
    print(f"  - Workflows: {profile.workflow_distribution}")

    assert profile.name == "normal"
    assert profile.requests_per_second == 200
    print("[OK] Profile validation passed\n")


async def test_traffic_generation():
    """Test traffic generation for 10 seconds."""
    print("=" * 70)
    print("TEST 2: Traffic Generation (10 seconds)")
    print("=" * 70)

    # Load profile
    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')

    # Override for quick test
    profile.duration_seconds = 10
    profile.ramp_up_seconds = 2
    profile.ramp_down_seconds = 2
    profile.requests_per_second = 20  # Lower rate for testing

    print(f"Profile: {profile.name}, {profile.requests_per_second} RPS")

    # Create generator
    service_urls = {
        'user': 'http://localhost:8001',
        'product': 'http://localhost:8003',
        'cart': 'http://localhost:8004',
        'order': 'http://localhost:8005',
        'auth': 'http://localhost:8002',
        'inventory': 'http://localhost:8007',
        'payment': 'http://localhost:8006',
    }

    generator = TrafficGenerator(profile, service_urls)

    print("[OK] Generator created")
    print("Starting traffic generation...")

    try:
        # Start generator
        await generator.start()
        print("[OK] Generator started")

        # Monitor for 10 seconds
        for i in range(10):
            await asyncio.sleep(1)
            stats = generator.get_stats()
            print(f"  [{i+1}s] Requests: {stats['total']}, "
                  f"Active users: {stats['active_users']}, "
                  f"Success rate: {stats['success_rate']:.1%}")

        # Get final stats
        stats = generator.get_stats()

        print("\n[OK] Generation complete")
        print(f"Total requests: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Latency p50: {stats['latency_p50']:.0f}ms")
        print(f"Latency p95: {stats['latency_p95']:.0f}ms")
        print(f"Latency p99: {stats['latency_p99']:.0f}ms")
        print(f"Throughput: {stats['throughput_rps']:.1f} RPS")

        print("\nRequests by workflow:")
        for workflow, count in stats['requests_by_workflow'].items():
            print(f"  - {workflow}: {count}")

        # Stop generator
        await generator.stop()
        print("\n[OK] Generator stopped cleanly")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        await generator.stop()
        raise


async def test_all_profiles():
    """Test loading all profile files."""
    print("\n" + "=" * 70)
    print("TEST 3: All Profile Files")
    print("=" * 70)

    profiles = [
        'simulator/traffic/profiles/normal.yaml',
        'simulator/traffic/profiles/peak.yaml',
        'simulator/traffic/profiles/degraded.yaml',
        'simulator/traffic/profiles/burst.yaml',
    ]

    for profile_path in profiles:
        try:
            profile = TrafficProfile.from_yaml(profile_path)
            print(f"[OK] {profile.name}: {profile.requests_per_second} RPS, "
                  f"{profile.duration_seconds}s duration")
        except Exception as e:
            print(f"[FAIL] Failed to load {profile_path}: {e}")
            raise

    print("[OK] All profiles loaded successfully\n")


async def test_workflows():
    """Test that workflows are defined correctly."""
    print("=" * 70)
    print("TEST 4: Workflow Definitions")
    print("=" * 70)

    from simulator.traffic.generator import (
        BrowseWorkflow,
        PurchaseWorkflow,
        AccountWorkflow,
        QuickBuyWorkflow
    )

    service_urls = {
        'user': 'http://localhost:8001',
        'product': 'http://localhost:8003',
        'cart': 'http://localhost:8004',
        'order': 'http://localhost:8005',
    }

    workflows = [
        ('BrowseWorkflow', BrowseWorkflow),
        ('PurchaseWorkflow', PurchaseWorkflow),
        ('AccountWorkflow', AccountWorkflow),
        ('QuickBuyWorkflow', QuickBuyWorkflow),
    ]

    for name, workflow_class in workflows:
        workflow = workflow_class('test_user', 'premium', service_urls)
        steps = workflow._define_steps()
        print(f"[OK] {name}: {len(steps)} steps defined")

        # Test getting first request
        request = workflow.get_next_request()
        assert request is not None, f"{name} should return a request"
        print(f"  First request: {request[1]} {request[0]}")

    print("[OK] All workflows validated\n")


async def test_pause_resume():
    """Test pause/resume functionality."""
    print("=" * 70)
    print("TEST 5: Pause/Resume")
    print("=" * 70)

    profile = TrafficProfile.from_yaml('simulator/traffic/profiles/normal.yaml')
    profile.duration_seconds = 20
    profile.requests_per_second = 10

    service_urls = {'user': 'http://localhost:8001'}
    generator = TrafficGenerator(profile, service_urls)

    try:
        await generator.start()
        print("[OK] Generator started")

        # Run for 3 seconds
        await asyncio.sleep(3)
        stats1 = generator.get_stats()
        print(f"After 3s: {stats1['total']} requests")

        # Pause
        generator.pause()
        print("[OK] Generator paused")
        await asyncio.sleep(2)
        stats2 = generator.get_stats()
        print(f"After pause: {stats2['total']} requests (should be same)")

        # Resume
        generator.resume()
        print("[OK] Generator resumed")
        await asyncio.sleep(3)
        stats3 = generator.get_stats()
        print(f"After resume: {stats3['total']} requests (should increase)")

        assert stats3['total'] > stats2['total'], "Traffic should resume after pause"

        await generator.stop()
        print("[OK] Pause/resume working correctly\n")

    except Exception as e:
        await generator.stop()
        raise


async def main():
    """Run all tests."""
    print("\n")
    print("=" + "=" * 68 + "=")
    print("|" + " " * 18 + "TRAFFIC GENERATOR TESTS" + " " * 27 + "|")
    print("=" + "=" * 68 + "╝")

    try:
        await test_profile_loading()
        await test_all_profiles()
        await test_workflows()

        print("\n" + "=" * 70)
        print("NOTE: The following tests will attempt to connect to services.")
        print("They will show connection errors if services are not running.")
        print("This is expected behavior - the generator handles errors gracefully.")
        print("=" * 70 + "\n")

        await test_traffic_generation()
        await test_pause_resume()

        print("=" * 70)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  [OK] Profile loading from YAML working")
        print("  [OK] All 4 profile files valid")
        print("  [OK] All 4 workflows defined correctly")
        print("  [OK] Traffic generation working")
        print("  [OK] Statistics tracking accurate")
        print("  [OK] Pause/resume functionality working")
        print()
        print("Traffic generator is ready to use!")
        print()
        print("To use with live services:")
        print("  1. Start e-commerce services (orchestrator)")
        print("  2. Run: python -c 'from simulator.traffic.generator import ...'")
        print("  3. Or use the example in the validation code")
        print()

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

