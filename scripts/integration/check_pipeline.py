#!/usr/bin/env python3
"""
Pipeline Connectivity Test Script

Tests all endpoints and pipeline components of the Voice AI system
to ensure end-to-end connectivity from frontend → backend → all layers.
"""

import argparse
import sys
import json
import requests
import time
from urllib.parse import urlparse


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test(name):
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing:{Colors.RESET} {name}")


def print_success(msg):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def print_fail(msg):
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def print_warn(msg):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")


def test_server_health(server_url):
    """Test /healthz endpoint"""
    print_test("Server Health Check")
    try:
        response = requests.get(f"{server_url}/healthz", timeout=5)
        data = response.json()
        
        if data.get("status") == "ok":
            print_success(f"Server is online (device: {data.get('device')})")
            
            if data.get("whisper_ready"):
                print_success("Whisper STT model loaded")
            else:
                print_warn("Whisper STT model NOT loaded")
            
            if data.get("lid_ready"):
                print_success("Language detection model loaded")
            else:
                print_warn("Language detection model NOT loaded")
            
            if data.get("redis"):
                print_success("Redis connected")
            else:
                print_warn("Redis NOT connected")
            
            ollama = data.get("ollama", {})
            if ollama.get("server_running"):
                print_success("Ollama LLM server running")
                if ollama.get("llm_model_ready"):
                    print_success("LLM model ready")
                else:
                    print_warn("LLM model NOT ready")
            else:
                print_warn("Ollama server NOT running")
            
            return True
        else:
            print_fail(f"Server status: {data.get('status')}")
            return False
    except requests.exceptions.RequestException as e:
        print_fail(f"Failed to connect: {e}")
        return False


def test_ready_endpoint(server_url):
    """Test /ready endpoint for model readiness"""
    print_test("Model Readiness Check")
    try:
        response = requests.get(f"{server_url}/ready", timeout=5)
        data = response.json()
        
        if data.get("ready"):
            print_success("All models loaded and ready to accept calls")
            return True
        else:
            print_warn("Models still loading:")
            if not data.get("whisper_ready"):
                print_warn("  - Whisper STT not ready")
            if not data.get("lid_ready"):
                print_warn("  - Language detection not ready")
            return False
    except requests.exceptions.RequestException as e:
        print_fail(f"Failed to check readiness: {e}")
        return False


def test_cors_headers(server_url):
    """Test CORS headers on /offer endpoint"""
    print_test("CORS Headers (cross-origin support)")
    try:
        response = requests.options(
            f"{server_url}/offer",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            timeout=5
        )
        
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        cors_methods = response.headers.get("Access-Control-Allow-Methods")
        
        if cors_origin == "*" or cors_origin == "http://localhost:3000":
            print_success(f"CORS Origin: {cors_origin}")
        else:
            print_fail(f"CORS Origin not set correctly: {cors_origin}")
            return False
        
        if cors_methods and "POST" in cors_methods:
            print_success(f"CORS Methods: {cors_methods}")
        else:
            print_warn(f"CORS Methods may not include POST: {cors_methods}")
        
        return True
    except requests.exceptions.RequestException as e:
        print_fail(f"CORS check failed: {e}")
        return False


def test_ice_config(server_url):
    """Test /ice-config endpoint"""
    print_test("ICE Server Configuration")
    try:
        response = requests.get(f"{server_url}/ice-config", timeout=5)
        data = response.json()
        
        ice_servers = data.get("iceServers", [])
        if not ice_servers:
            print_fail("No ICE servers configured")
            return False
        
        has_stun = False
        has_turn = False
        
        for server in ice_servers:
            urls = server.get("urls", "")
            if "stun:" in urls:
                has_stun = True
                print_success(f"STUN server: {urls}")
            elif "turn:" in urls:
                has_turn = True
                username = server.get("username", "N/A")
                print_success(f"TURN server: {urls} (username: {username})")
        
        if not has_stun:
            print_warn("No STUN servers found")
        
        if not has_turn:
            print_warn("No TURN servers found (may have issues with NAT)")
        
        return has_stun or has_turn
    except requests.exceptions.RequestException as e:
        print_fail(f"Failed to get ICE config: {e}")
        return False


def test_sdp_offer(server_url):
    """Test /offer endpoint with mock SDP"""
    print_test("WebRTC SDP Offer Exchange")
    
    # Minimal valid SDP offer
    mock_sdp = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test
a=ice-pwd:testtesttesttest
a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
a=setup:actpass
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
"""
    
    try:
        response = requests.post(
            f"{server_url}/offer",
            json={
                "sdp": mock_sdp,
                "type": "offer",
                "language": "eng",
                "auto_detect": False
            },
            timeout=10
        )
        
        if response.status_code == 503:
            print_warn("Server models not ready yet (503)")
            return False
        
        if response.status_code != 200:
            print_fail(f"HTTP {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        
        if "error" in data:
            print_fail(f"Error in response: {data['error']}")
            return False
        
        if data.get("type") == "answer" and data.get("sdp"):
            print_success("SDP offer accepted, answer received")
            print_success(f"Answer SDP length: {len(data['sdp'])} bytes")
            return True
        else:
            print_fail("Invalid answer format")
            return False
            
    except requests.exceptions.RequestException as e:
        print_fail(f"SDP offer failed: {e}")
        return False


def test_metrics(server_url):
    """Test /metrics endpoint"""
    print_test("Metrics Endpoint")
    try:
        response = requests.get(f"{server_url}/metrics", timeout=5)
        data = response.json()
        
        print_success(f"Total calls: {data.get('calls_total', 0)}")
        print_success(f"Active calls: {data.get('active_calls', 0)}")
        print_success(f"Utterances processed: {data.get('utterances_total', 0)}")
        
        latency_stats = data.get("latency_stats", {})
        if latency_stats:
            print_success("Latency statistics available:")
            for metric, stats in latency_stats.items():
                print(f"    {metric}: avg={stats.get('avg_ms', 0):.0f}ms, p95={stats.get('p95_ms', 0):.0f}ms")
        else:
            print_warn("No latency statistics yet (no calls processed)")
        
        return True
    except requests.exceptions.RequestException as e:
        print_fail(f"Failed to get metrics: {e}")
        return False


def run_all_tests(server_url):
    """Run all tests and return overall success"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Voice AI Pipeline Connectivity Test{Colors.RESET}")
    print(f"{Colors.BOLD}Server: {server_url}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    results = []
    
    # Critical tests
    results.append(("Server Health", test_server_health(server_url)))
    results.append(("Model Readiness", test_ready_endpoint(server_url)))
    results.append(("CORS Support", test_cors_headers(server_url)))
    results.append(("ICE Configuration", test_ice_config(server_url)))
    results.append(("SDP Offer/Answer", test_sdp_offer(server_url)))
    
    # Optional tests
    results.append(("Metrics", test_metrics(server_url)))
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All {total} tests passed!{Colors.RESET}")
        print(f"\n{Colors.GREEN}The pipeline is ready for end-to-end calls.{Colors.RESET}")
        return 0
    else:
        failed = total - passed
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {passed}/{total} tests passed ({failed} failed){Colors.RESET}")
        
        if not results[0][1]:  # Server health failed
            print(f"\n{Colors.RED}CRITICAL: Server is not accessible or not running.{Colors.RESET}")
        elif not results[1][1]:  # Model readiness failed
            print(f"\n{Colors.YELLOW}Models are still loading. Wait ~30-60s and try again.{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}Some tests failed. Review errors above.{Colors.RESET}")
        
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test Voice AI pipeline connectivity and readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --server http://localhost:8080
  %(prog)s --server http://165.245.140.34:8080
        """
    )
    parser.add_argument(
        '--server',
        default='http://localhost:8080',
        help='Server URL (default: http://localhost:8080)'
    )
    
    args = parser.parse_args()
    
    # Normalize URL
    server_url = args.server.rstrip('/')
    if not server_url.startswith('http://') and not server_url.startswith('https://'):
        server_url = f'http://{server_url}'
    
    exit_code = run_all_tests(server_url)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
