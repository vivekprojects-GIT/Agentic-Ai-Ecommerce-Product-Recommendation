#!/usr/bin/env python3
"""
Comprehensive test suite for RAG implementation and image descriptions.
Tests all possible scenarios to verify proper RAG patterns and image functionality.
"""

import sys
import os
import json
import base64
import io
import requests
from PIL import Image
from typing import Dict, List, Any
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.vector_store.catalog import CatalogStore
from src.tools.vision import VisionTool

class RAGTester:
    """Comprehensive tester for RAG implementation and image descriptions."""
    
    def __init__(self):
        self.config = get_config()
        self.catalog = CatalogStore(
            persist_dir=self.config.chroma_persist_dir,
            embedding_model=self.config.embedding_model
        )
        self.vision_tool = VisionTool(self.config.google_api_key)
        self.api_base_url = f"http://{self.config.api_host}:{self.config.api_port}"
        self.test_results = {
            "rag_tests": {},
            "image_tests": {},
            "integration_tests": {},
            "summary": {}
        }
    
    def create_test_images(self) -> Dict[str, str]:
        """Create various test images for comprehensive testing."""
        test_images = {}
        
        # Test 1: Red square (simple color test)
        red_img = Image.new('RGB', (100, 100), color='red')
        test_images['red_square'] = self._image_to_base64(red_img)
        
        # Test 2: Blue circle-like (color and shape)
        blue_img = Image.new('RGB', (100, 100), color='blue')
        test_images['blue_square'] = self._image_to_base64(blue_img)
        
        # Test 3: Green rectangle
        green_img = Image.new('RGB', (150, 100), color='green')
        test_images['green_rectangle'] = self._image_to_base64(green_img)
        
        # Test 4: Multi-color gradient
        gradient_img = Image.new('RGB', (100, 100))
        for x in range(100):
            for y in range(100):
                gradient_img.putpixel((x, y), (x*2, y*2, 100))
        test_images['gradient'] = self._image_to_base64(gradient_img)
        
        return test_images
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def test_direct_catalog_search(self) -> Dict[str, Any]:
        """Test direct catalog search to verify RAG retrieval."""
        print("🔍 Testing Direct Catalog Search (RAG Retrieval)...")
        
        test_queries = [
            "red shirt",
            "running shoes",
            "electronics",
            "clothing under $50",
            "Nike products",
            "blue items"
        ]
        
        results = {}
        for query in test_queries:
            try:
                # Direct catalog search (this is the RAG retrieval step)
                products = self.catalog.search(query, top_k=5)
                results[query] = {
                    "success": True,
                    "products_found": len(products),
                    "products": [
                        {
                            "name": p.get('name', 'N/A'),
                            "price": p.get('price', 0),
                            "brand": p.get('attributes', {}).get('brand', 'N/A')
                        } for p in products
                    ]
                }
                print(f"  ✅ '{query}': {len(products)} products found")
            except Exception as e:
                results[query] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  ❌ '{query}': {str(e)}")
        
        return results
    
    def test_vision_descriptions(self) -> Dict[str, Any]:
        """Test image description quality and accuracy."""
        print("\n📷 Testing Image Descriptions...")
        
        test_images = self.create_test_images()
        results = {}
        
        for image_name, image_base64 in test_images.items():
            try:
                description = self.vision_tool.analyze_image(image_base64)
                results[image_name] = {
                    "success": True,
                    "description": description,
                    "description_length": len(description),
                    "has_colors": any(color in description.lower() for color in ['red', 'blue', 'green', 'color']),
                    "is_detailed": len(description) > 20
                }
                print(f"  ✅ {image_name}: {description[:50]}...")
            except Exception as e:
                results[image_name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  ❌ {image_name}: {str(e)}")
        
        return results
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test API endpoints for proper RAG implementation."""
        print("\n🌐 Testing API Endpoints...")
        
        # Test health endpoint
        health_result = self._test_endpoint("GET", "/health", expected_status=200)
        
        # Test text-only queries
        text_tests = [
            {"query": "red shirts", "expected_intent": "product_search"},
            {"query": "shoes under $100", "expected_intent": "product_search"},
            {"query": "hello", "expected_intent": "general_chat"}
        ]
        
        text_results = {}
        for test in text_tests:
            result = self._test_endpoint(
                "POST", 
                "/api/v1/simple-rag/ask",
                data={"text_input": test["query"]},
                expected_status=200
            )
            text_results[test["query"]] = result
        
        # Test image + text queries
        test_images = self.create_test_images()
        image_results = {}
        for image_name, image_base64 in test_images.items():
            result = self._test_endpoint(
                "POST",
                "/api/v1/simple-rag/ask", 
                data={
                    "text_input": "find products similar to this",
                    "image_base64": image_base64
                },
                expected_status=200
            )
            image_results[image_name] = result
        
        return {
            "health": health_result,
            "text_queries": text_results,
            "image_queries": image_results
        }
    
    def _test_endpoint(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single API endpoint."""
        try:
            url = f"{self.api_base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
            
            return {
                "success": response.status_code == expected_status,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expected_status": expected_status
            }
    
    def test_rag_quality(self) -> Dict[str, Any]:
        """Test RAG quality by analyzing retrieval relevance."""
        print("\n🎯 Testing RAG Quality...")
        
        test_cases = [
            {
                "query": "red athletic shoes",
                "expected_attributes": ["red", "shoes", "athletic"],
                "min_products": 1
            },
            {
                "query": "cheap t-shirts",
                "expected_attributes": ["shirt", "t-shirt"],
                "price_filter": lambda p: p.get('price', 0) < 50,
                "min_products": 1
            },
            {
                "query": "Nike running gear",
                "expected_attributes": ["nike"],
                "min_products": 1
            }
        ]
        
        results = {}
        for test_case in test_cases:
            query = test_case["query"]
            try:
                products = self.catalog.search(query, top_k=5)
                
                # Analyze relevance
                relevance_score = 0
                for product in products:
                    product_text = f"{product.get('name', '')} {product.get('description', '')}".lower()
                    
                    # Check if expected attributes are found
                    for attr in test_case["expected_attributes"]:
                        if attr.lower() in product_text:
                            relevance_score += 1
                
                # Calculate relevance percentage
                total_possible = len(test_case["expected_attributes"]) * len(products)
                relevance_percentage = (relevance_score / total_possible * 100) if total_possible > 0 else 0
                
                # Check price filter if specified
                price_filter_passed = True
                if "price_filter" in test_case:
                    price_filter_passed = any(test_case["price_filter"](p) for p in products)
                
                results[query] = {
                    "success": len(products) >= test_case["min_products"],
                    "products_found": len(products),
                    "min_required": test_case["min_products"],
                    "relevance_score": relevance_score,
                    "relevance_percentage": relevance_percentage,
                    "price_filter_passed": price_filter_passed,
                    "products": [
                        {
                            "name": p.get('name', 'N/A'),
                            "price": p.get('price', 0),
                            "relevance_indicators": [
                                attr for attr in test_case["expected_attributes"] 
                                if attr.lower() in f"{p.get('name', '')} {p.get('description', '')}".lower()
                            ]
                        } for p in products
                    ]
                }
                
                print(f"  ✅ '{query}': {len(products)} products, {relevance_percentage:.1f}% relevance")
                
            except Exception as e:
                results[query] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  ❌ '{query}': {str(e)}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🚀 Starting Comprehensive RAG and Image Testing...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        self.test_results["rag_tests"] = self.test_direct_catalog_search()
        self.test_results["image_tests"] = self.test_vision_descriptions()
        self.test_results["api_tests"] = self.test_api_endpoints()
        self.test_results["quality_tests"] = self.test_rag_quality()
        
        # Generate summary
        self.test_results["summary"] = self._generate_summary()
        
        total_time = time.time() - start_time
        self.test_results["summary"]["total_test_time"] = total_time
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        self._print_summary()
        
        return self.test_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "rag_quality": "unknown",
            "image_quality": "unknown",
            "api_functionality": "unknown"
        }
        
        # Count RAG tests
        for test_name, result in self.test_results["rag_tests"].items():
            summary["total_tests"] += 1
            if result.get("success", False):
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1
        
        # Count image tests
        for test_name, result in self.test_results["image_tests"].items():
            summary["total_tests"] += 1
            if result.get("success", False):
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1
        
        # Count API tests
        api_tests = self.test_results.get("api_tests", {})
        for test_name, result in api_tests.items():
            if isinstance(result, dict) and "success" in result:
                summary["total_tests"] += 1
                if result["success"]:
                    summary["passed_tests"] += 1
                else:
                    summary["failed_tests"] += 1
        
        # Determine quality ratings
        if summary["passed_tests"] / summary["total_tests"] > 0.8:
            summary["rag_quality"] = "excellent"
        elif summary["passed_tests"] / summary["total_tests"] > 0.6:
            summary["rag_quality"] = "good"
        elif summary["passed_tests"] / summary["total_tests"] > 0.4:
            summary["rag_quality"] = "fair"
        else:
            summary["rag_quality"] = "poor"
        
        return summary
    
    def _print_summary(self):
        """Print formatted test summary."""
        summary = self.test_results["summary"]
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Success Rate: {(summary['passed_tests']/summary['total_tests']*100):.1f}%")
        print(f"Overall Quality: {summary['rag_quality'].upper()}")
        print(f"Test Duration: {summary['total_test_time']:.2f}s")
        
        # Specific recommendations
        print("\n📋 RECOMMENDATIONS:")
        if summary["rag_quality"] == "poor":
            print("  🔧 RAG implementation needs significant improvements")
        elif summary["rag_quality"] == "fair":
            print("  🔧 RAG implementation needs some improvements")
        elif summary["rag_quality"] == "good":
            print("  ✅ RAG implementation is working well")
        else:
            print("  🎉 RAG implementation is excellent!")
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file."""
        filepath = os.path.join("tests", filename)
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Test results saved to: {filepath}")

def main():
    """Run comprehensive RAG and image testing."""
    tester = RAGTester()
    results = tester.run_all_tests()
    tester.save_results()
    
    # Return exit code based on test results
    if results["summary"]["rag_quality"] in ["poor", "fair"]:
        sys.exit(1)  # Tests failed
    else:
        sys.exit(0)  # Tests passed

if __name__ == "__main__":
    main()
