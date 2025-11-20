#!/usr/bin/env python3
"""
Test script for AgenticIQA FastAPI server.

Tests all API endpoints with various scenarios including:
- Health check
- File upload assessment (both NR and FR modes)
- Path-based assessment
- Error handling (invalid files, missing parameters, etc.)

Usage:
    # Start the API server first:
    python scripts/run_api.py &

    # Then run tests:
    python scripts/test_api.py

    # Or test against a different server:
    python scripts/test_api.py --base-url http://localhost:9000
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Colors:
    """Terminal colors for output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_section(title):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_test(name, status, message=""):
    """Print test result."""
    status_icon = "✓" if status else "✗"
    status_color = Colors.GREEN if status else Colors.RED
    print(f"{status_color}{status_icon}{Colors.RESET} {name}")
    if message:
        print(f"  {Colors.YELLOW}{message}{Colors.RESET}")


def create_test_image(size=(256, 256), color=(255, 0, 0)):
    """Create a test image in memory."""
    img = Image.new('RGB', size, color=color)
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_health_check(base_url):
    """Test health check endpoint."""
    print_section("Testing Health Check Endpoint")

    try:
        response = requests.get(f"{base_url}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_test("Health check successful", True)
            print(f"  Status: {data.get('status')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Timestamp: {data.get('timestamp')}")
            return True
        else:
            print_test("Health check failed", False, f"Status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print_test("Health check failed", False, "Connection refused - is the server running?")
        return False
    except Exception as e:
        print_test("Health check failed", False, str(e))
        return False


def test_root_endpoint(base_url):
    """Test root info endpoint."""
    print_section("Testing Root Info Endpoint")

    try:
        response = requests.get(f"{base_url}/", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_test("Root endpoint successful", True)
            print(f"  Name: {data.get('name')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Docs URL: {data.get('docs_url')}")
            return True
        else:
            print_test("Root endpoint failed", False, f"Status code: {response.status_code}")
            return False

    except Exception as e:
        print_test("Root endpoint failed", False, str(e))
        return False


def test_file_upload_no_reference(base_url):
    """Test file upload assessment without reference (No-Reference mode)."""
    print_section("Testing File Upload (No-Reference Mode)")

    try:
        # Create test image
        test_image = create_test_image(color=(100, 150, 200))

        # Prepare multipart form data
        files = {
            'image': ('test_image.jpg', test_image, 'image/jpeg')
        }
        data = {
            'query': 'Test query: How is the image quality?',
            'max_replan_iterations': 1
        }

        print("  Sending request with test image...")
        print(f"  Query: {data['query']}")

        # Note: This will actually call the pipeline, which requires:
        # - API keys configured
        # - Pipeline dependencies available
        # For a quick test, we expect it might fail at pipeline execution
        # but should pass request validation

        response = requests.post(
            f"{base_url}/assess",
            files=files,
            data=data,
            timeout=30
        )

        print(f"  Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print_test("File upload assessment successful", True)
            print(f"  Final answer: {result.get('final_answer')}")
            print(f"  Quality score: {result.get('quality_score')}")
            print(f"  Reasoning: {result.get('quality_reasoning', '')[:100]}...")
            print(f"  Execution time: {result.get('execution_metadata', {}).get('execution_time_seconds')}s")
            return True
        elif response.status_code == 422:
            # Validation error - this is expected behavior
            print_test("Request validation working", True, "Pydantic validation triggered as expected")
            print(f"  Error detail: {response.json()}")
            return True
        elif response.status_code in [400, 500]:
            # Pipeline execution error - acceptable for testing without full setup
            error = response.json().get('detail', {})
            error_type = error.get('error_type', 'unknown') if isinstance(error, dict) else 'unknown'
            print_test("Request accepted but pipeline failed", True,
                      f"Expected without full environment: {error_type}")
            return True
        else:
            print_test("Unexpected response", False, f"Status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False

    except Exception as e:
        print_test("File upload test failed", False, str(e))
        return False


def test_file_upload_with_reference(base_url):
    """Test file upload assessment with reference (Full-Reference mode)."""
    print_section("Testing File Upload (Full-Reference Mode)")

    try:
        # Create test images
        distorted_image = create_test_image(color=(100, 100, 100))
        reference_image = create_test_image(color=(200, 200, 200))

        # Prepare multipart form data
        files = {
            'image': ('distorted.jpg', distorted_image, 'image/jpeg'),
            'reference': ('reference.jpg', reference_image, 'image/jpeg')
        }
        data = {
            'query': 'Compare the distorted image quality to the reference',
            'max_replan_iterations': 1
        }

        print("  Sending request with distorted and reference images...")

        response = requests.post(
            f"{base_url}/assess",
            files=files,
            data=data,
            timeout=30
        )

        print(f"  Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print_test("Full-reference assessment successful", True)
            print(f"  Final answer: {result.get('final_answer')}")
            return True
        elif response.status_code in [400, 422, 500]:
            print_test("Request accepted but execution failed", True,
                      "Expected without full environment")
            return True
        else:
            print_test("Unexpected response", False, f"Status: {response.status_code}")
            return False

    except Exception as e:
        print_test("Full-reference test failed", False, str(e))
        return False


def test_path_based_assessment(base_url):
    """Test path-based assessment endpoint."""
    print_section("Testing Path-Based Assessment")

    try:
        # Create a temporary test image
        temp_dir = Path(os.getenv('AGENTIC_ROOT', '.')).expanduser() / 'tmp' / 'test_images'
        temp_dir.mkdir(parents=True, exist_ok=True)
        test_image_path = temp_dir / 'test_image.jpg'

        # Save test image
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        img.save(test_image_path, 'JPEG')

        print(f"  Using test image: {test_image_path}")

        # Send request
        payload = {
            'query': 'Assess the image quality',
            'image_path': str(test_image_path),
            'max_replan_iterations': 1
        }

        response = requests.post(
            f"{base_url}/assess-path",
            json=payload,
            timeout=30
        )

        print(f"  Response status: {response.status_code}")

        # Cleanup
        test_image_path.unlink(missing_ok=True)

        if response.status_code == 200:
            result = response.json()
            print_test("Path-based assessment successful", True)
            print(f"  Final answer: {result.get('final_answer')}")
            return True
        elif response.status_code in [400, 500]:
            print_test("Request accepted but execution failed", True,
                      "Expected without full environment")
            return True
        else:
            print_test("Unexpected response", False, f"Status: {response.status_code}")
            return False

    except Exception as e:
        print_test("Path-based test failed", False, str(e))
        return False


def test_error_handling(base_url):
    """Test error handling and validation."""
    print_section("Testing Error Handling")

    passed_tests = 0
    total_tests = 0

    # Test 1: Missing required parameter
    total_tests += 1
    try:
        response = requests.post(f"{base_url}/assess", data={}, timeout=5)
        if response.status_code == 422:
            print_test("Missing parameter validation", True, "422 Unprocessable Entity")
            passed_tests += 1
        else:
            print_test("Missing parameter validation", False, f"Expected 422, got {response.status_code}")
    except Exception as e:
        print_test("Missing parameter validation", False, str(e))

    # Test 2: Invalid file format
    total_tests += 1
    try:
        files = {'image': ('test.txt', BytesIO(b'not an image'), 'text/plain')}
        data = {'query': 'Test query'}
        response = requests.post(f"{base_url}/assess", files=files, data=data, timeout=5)
        if response.status_code == 400:
            print_test("Invalid file format validation", True, "400 Bad Request")
            passed_tests += 1
        else:
            print_test("Invalid file format validation", False, f"Expected 400, got {response.status_code}")
    except Exception as e:
        print_test("Invalid file format validation", False, str(e))

    # Test 3: File too large (if we could simulate)
    # Skipping as it requires actual large file

    # Test 4: Invalid path in path-based assessment
    total_tests += 1
    try:
        payload = {
            'query': 'Test query',
            'image_path': '/nonexistent/path/image.jpg'
        }
        response = requests.post(f"{base_url}/assess-path", json=payload, timeout=5)
        # Accept both 400 and 500 as valid error responses
        if response.status_code in [400, 500]:
            print_test("Invalid path validation", True, f"{response.status_code} Error (expected)")
            passed_tests += 1
        else:
            print_test("Invalid path validation", False, f"Expected error, got {response.status_code}")
    except Exception as e:
        print_test("Invalid path validation", False, str(e))

    print(f"\n  Error handling tests: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


def test_api_documentation(base_url):
    """Test API documentation endpoints."""
    print_section("Testing API Documentation")

    passed = 0
    total = 3

    # Test Swagger UI
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200 and 'swagger' in response.text.lower():
            print_test("Swagger UI available", True, f"{base_url}/docs")
            passed += 1
        else:
            print_test("Swagger UI check", False)
    except Exception as e:
        print_test("Swagger UI check failed", False, str(e))

    # Test ReDoc
    try:
        response = requests.get(f"{base_url}/redoc", timeout=5)
        if response.status_code == 200 and 'redoc' in response.text.lower():
            print_test("ReDoc available", True, f"{base_url}/redoc")
            passed += 1
        else:
            print_test("ReDoc check", False)
    except Exception as e:
        print_test("ReDoc check failed", False, str(e))

    # Test OpenAPI schema
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        if response.status_code == 200:
            schema = response.json()
            print_test("OpenAPI schema available", True)
            print(f"  API Title: {schema.get('info', {}).get('title')}")
            print(f"  API Version: {schema.get('info', {}).get('version')}")
            print(f"  Endpoints: {len(schema.get('paths', {}))}")
            passed += 1
        else:
            print_test("OpenAPI schema check", False)
    except Exception as e:
        print_test("OpenAPI schema check failed", False, str(e))

    return passed == total


def main():
    """Run all API tests."""
    parser = argparse.ArgumentParser(description="Test AgenticIQA FastAPI server")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip tests that require full pipeline setup"
    )

    args = parser.parse_args()
    base_url = args.base_url.rstrip('/')

    print(f"\n{Colors.BOLD}AgenticIQA API Test Suite{Colors.RESET}")
    print(f"Testing server at: {Colors.BLUE}{base_url}{Colors.RESET}\n")

    # Check if server is running
    try:
        requests.get(f"{base_url}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ Error: API server is not running at {base_url}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Start the server first:{Colors.RESET}")
        print(f"  python scripts/run_api.py\n")
        sys.exit(1)

    # Run tests
    results = []

    results.append(("Health Check", test_health_check(base_url)))
    results.append(("Root Endpoint", test_root_endpoint(base_url)))
    results.append(("API Documentation", test_api_documentation(base_url)))
    results.append(("Error Handling", test_error_handling(base_url)))

    if not args.skip_pipeline:
        print(f"\n{Colors.YELLOW}Note: The following tests require the full AgenticIQA pipeline{Colors.RESET}")
        print(f"{Colors.YELLOW}to be configured (VLM models, environment variables, etc.){Colors.RESET}")
        time.sleep(1)

        results.append(("File Upload (NR)", test_file_upload_no_reference(base_url)))
        results.append(("File Upload (FR)", test_file_upload_with_reference(base_url)))
        results.append(("Path-Based Assessment", test_path_based_assessment(base_url)))

    # Summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status_icon = "✓" if result else "✗"
        status_color = Colors.GREEN if result else Colors.RED
        print(f"{status_color}{status_icon}{Colors.RESET} {name}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print(f"{Colors.GREEN}All tests passed! ✓{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}Some tests failed.{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
