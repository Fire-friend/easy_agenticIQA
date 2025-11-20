#!/usr/bin/env python3
"""
Example client for AgenticIQA REST API.

Demonstrates how to call the API endpoints using Python requests library.

Requirements:
    pip install requests pillow

Usage:
    # Make sure API server is running first:
    python scripts/run_api.py &

    # Then run this example:
    python examples/api_client_example.py
"""

import requests
import tempfile
import os
from PIL import Image


# API server configuration
API_BASE_URL = "http://localhost:8000"


def create_test_image(path: str, size=(512, 512), color=(100, 150, 200)):
    """Create a test image at specified path."""
    img = Image.new('RGB', size, color=color)
    img.save(path, 'JPEG')


def example_health_check():
    """Example: Check API server health."""
    print("\n" + "="*60)
    print("Example 1: Health Check")
    print("="*60)

    response = requests.get(f"{API_BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ API server is healthy")
        print(f"  Status: {data['status']}")
        print(f"  Version: {data['version']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")


def example_path_based_assessment():
    """Example: Assess image via file path (No-Reference mode)."""
    print("\n" + "="*60)
    print("Example 2: Path-Based Assessment (No-Reference)")
    print("="*60)

    # Create a temporary test image
    # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
    #     create_test_image(tmp.name, color=(128, 128, 128))
    #     image_path = tmp.name
    image_path = '/data/wujiawei/CVPR/data/all_in_one/LOL/test/low/493.png'
    print(f"Using image path: {image_path}")

    # Prepare the request
    payload = {
        'query': 'How is the overall quality of this image?',
        'image_path': image_path,
        'max_replan_iterations': 2
    }

    try:
        print(f"Sending request...")
        print(f"  Query: {payload['query']}")

        response = requests.post(
            f"{API_BASE_URL}/assess",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Assessment successful!")
            print(f"  Final Answer: {result['final_answer']}")
            print(f"  Quality Score: {result.get('quality_score', 'N/A')}")
            print(f"  Quality Reasoning: {result['quality_reasoning'][:150]}...")
            print(f"  Execution Time: {result['execution_metadata']['execution_time_seconds']}s")
            print(f"  Iterations: {result['execution_metadata']['iteration_count']}")
        else:
            print(f"\n✗ Request failed: {response.status_code}")
            print(f"  Error: {response.json()}")

    except requests.exceptions.Timeout:
        print("✗ Request timed out")
    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        # Note: Files are preserved (not deleted) after processing
        print(f"\nNote: Test image preserved at: {image_path}")


def example_full_reference_assessment():
    """Example: Assess with reference image (Full-Reference mode)."""
    print("\n" + "="*60)
    print("Example 3: Path-Based Assessment (Full-Reference)")
    print("="*60)

    # Create temporary test images
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
        create_test_image(tmp1.name, color=(80, 80, 80))
        distorted_path = tmp1.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
        create_test_image(tmp2.name, color=(200, 200, 200))
        reference_path = tmp2.name

    print(f"Distorted image: {distorted_path}")
    print(f"Reference image: {reference_path}")

    # Prepare the request
    payload = {
        'query': 'Compare the distorted image quality to the reference image',
        'image_path': distorted_path,
        'reference_path': reference_path,
        'max_replan_iterations': 2
    }

    try:
        print(f"Sending request...")
        print(f"  Query: {payload['query']}")

        response = requests.post(
            f"{API_BASE_URL}/assess",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Assessment successful!")
            print(f"  Final Answer: {result['final_answer']}")
            print(f"  Quality Score: {result.get('quality_score', 'N/A')}")
            print(f"  Reasoning: {result['quality_reasoning'][:150]}...")
        else:
            print(f"\n✗ Request failed: {response.status_code}")
            print(f"  Error: {response.json()}")

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        # Note: Files are preserved (not deleted) after processing
        print(f"\nNote: Test images preserved at:")
        print(f"  Distorted: {distorted_path}")
        print(f"  Reference: {reference_path}")


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60)

    # Example 4a: Missing required parameter
    print("\n4a. Missing required parameter:")
    response = requests.post(f"{API_BASE_URL}/assess", json={})
    print(f"  Status Code: {response.status_code}")
    if response.status_code == 422:
        print(f"  Error: {response.json()['detail'][0]['msg']}")

    # Example 4b: Invalid file path
    print("\n4b. Invalid file path:")
    payload = {
        'query': 'Test query',
        'image_path': '/nonexistent/path/image.jpg'
    }
    response = requests.post(f"{API_BASE_URL}/assess", json=payload)
    print(f"  Status Code: {response.status_code}")
    if response.status_code == 500:
        error_detail = response.json()['detail']
        if isinstance(error_detail, dict):
            print(f"  Error Type: {error_detail.get('error_type')}")
            print(f"  Error Message: {error_detail.get('message')}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print(" AgenticIQA API Client Examples")
    print("="*60)
    print(f"\nAPI Server: {API_BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python scripts/run_api.py")
    print("="*60)

    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✓ API server is reachable")
        else:
            print("✗ API server returned unexpected status")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server")
        print("\nPlease start the server first:")
        print("  python scripts/run_api.py")
        return

    # Run examples
    # example_health_check()
    # example_error_handling()

    # Test pipeline execution (with real images)
    print("\n" + "="*60)
    print(" Testing Pipeline Execution")
    print("="*60)
    print("\nNote: These tests will execute the full AgenticIQA pipeline")
    print("and may take 30-60 seconds per request.\n")

    example_path_based_assessment()
    # example_full_reference_assessment()

    print("\n" + "="*60)
    print(" Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
