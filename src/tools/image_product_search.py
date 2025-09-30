"""
Image-Based Product Search Tool
Handles product recommendations and search based on uploaded images.
"""

import logging
from typing import Dict, Any, List, Optional
from src.tools.text_product_search import TextProductSearchTool
from src.config.dynamic_config import get_config

logger = logging.getLogger(__name__)

class ImageProductSearchTool:
    """Tool for image-based product search and recommendations."""
    
    def __init__(self, catalog, text_search_tool: TextProductSearchTool):
        """Initialize the image product search tool."""
        self.catalog = catalog
        self.text_search_tool = text_search_tool
        self.config = get_config()
        
    def search_products_by_image(self, image_base64: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products based on uploaded image."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            logger.info("Starting image-based product search")
            
            # Step 1: Analyze the image (simplified: no external vision dependency)
            image_analysis = self._analyze_image(image_base64)
            
            if image_analysis.get("error"):
                return {
                    "response": image_analysis["response"],
                    "products": [],
                    "intent": "image_search",
                    "confidence": 0.3,
                    "metadata": {
                        "tool_used": "image_product_search",
                        "error": image_analysis["error"]
                    }
                }
            
            # Step 2: Extract search query from image analysis
            search_query = self._extract_search_query_from_analysis(image_analysis["description"])
            
            # Step 3: Search for products using the extracted query
            search_result = self.text_search_tool.search_products(search_query, top_k)
            
            # Step 4: Enhance response with image context
            enhanced_response = self._enhance_response_with_image_context(
                search_result["response"], 
                image_analysis["description"]
            )
            
            return {
                "response": enhanced_response,
                "products": search_result["products"],
                "intent": "image_search",
                "confidence": search_result["confidence"],
                "metadata": {
                    "tool_used": "image_product_search",
                    "image_analysis": image_analysis["description"],
                    "search_query": search_query,
                    "products_found": len(search_result["products"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image product search: {e}")
            return {
                "response": "I encountered an error while analyzing your image. Please try uploading a different image or describe what you're looking for in text.",
                "products": [],
                "intent": "image_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "tool_used": "image_product_search"}
            }
    
    def _analyze_image(self, image_base64: str) -> Dict[str, Any]:
        """Analyze the uploaded image."""
        try:
            # Without the external vision tool, return a graceful fallback
            return {
                "description": "",
                "response": "📷 I can't analyze images right now, but I can still help! Please describe the item in text (e.g., 'red running shoes').",
                "error": "analysis_unavailable"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "description": "",
                "response": "📷 I'm having trouble analyzing the image right now. Please describe what you see or what you're looking for.",
                "error": str(e)
            }
    
    def _extract_search_query_from_analysis(self, analysis: str) -> str:
        """Extract a search query from image analysis."""
        if not analysis:
            return "clothing apparel"
        
        # Convert analysis to search-friendly terms
        analysis_lower = analysis.lower()
        
        # Extract key product terms
        product_terms = []
        
        # Clothing items
        clothing_keywords = {
            "shirt": ["shirt", "top", "blouse", "tee", "t-shirt"],
            "pants": ["pants", "trousers", "jeans", "shorts"],
            "dress": ["dress", "gown", "frock"],
            "jacket": ["jacket", "coat", "blazer", "hoodie"],
            "shoes": ["shoes", "sneakers", "boots", "sandals", "footwear"],
            "accessories": ["bag", "hat", "cap", "belt", "watch", "jewelry"]
        }
        
        for category, keywords in clothing_keywords.items():
            if any(keyword in analysis_lower for keyword in keywords):
                product_terms.append(category)
        
        # Colors
        color_keywords = ["red", "blue", "green", "black", "white", "yellow", "pink", "purple", "orange", "brown", "gray", "grey"]
        colors_found = [color for color in color_keywords if color in analysis_lower]
        
        # Styles
        style_keywords = ["casual", "formal", "sport", "athletic", "elegant", "vintage", "modern", "classic"]
        styles_found = [style for style in style_keywords if style in analysis_lower]
        
        # Build search query
        query_parts = []
        
        if product_terms:
            query_parts.extend(product_terms)
        else:
            query_parts.append("clothing")
        
        if colors_found:
            query_parts.extend(colors_found)
        
        if styles_found:
            query_parts.extend(styles_found)
        
        # Add some general terms if nothing specific found
        if not query_parts:
            query_parts = ["apparel", "clothing"]
        
        return " ".join(query_parts)
    
    def _enhance_response_with_image_context(self, base_response: str, image_description: str) -> str:
        """Enhance the response with image analysis context."""
        if not image_description:
            return base_response
        
        enhanced_response = f"Based on your image, I can see: {image_description}\n\n"
        enhanced_response += base_response
        
        return enhanced_response
    
    def get_supported_image_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return ["JPEG", "PNG", "JPG", "WEBP"]
    
    def validate_image_format(self, image_base64: str) -> Dict[str, Any]:
        """Validate if the uploaded image is in a supported format."""
        try:
            # Basic validation - check if it's a valid base64 image
            if not image_base64 or len(image_base64) < 100:
                return {
                    "valid": False,
                    "error": "Invalid or empty image data"
                }
            
            # Check for common image format headers
            if image_base64.startswith("data:image/"):
                # Extract format from data URL
                format_part = image_base64.split(";")[0].split("/")[-1].upper()
                if format_part in self.get_supported_image_formats():
                    return {"valid": True, "format": format_part}
                else:
                    return {
                        "valid": False,
                        "error": f"Unsupported format: {format_part}. Supported formats: {', '.join(self.get_supported_image_formats())}"
                    }
            
            # If no data URL prefix, assume it's raw base64
            return {"valid": True, "format": "Unknown"}
            
        except Exception as e:
            logger.error(f"Error validating image format: {e}")
            return {
                "valid": False,
                "error": f"Error validating image: {str(e)}"
            }
