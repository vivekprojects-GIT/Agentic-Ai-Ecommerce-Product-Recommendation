"""
Image-Based Product Search Tool
Generates a compact, generalizable description from the image using an LLM,
then reuses the text search pipeline. No static product vocabularies.
"""

import logging
from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
import google.generativeai as genai
import json
import re
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
        # Configure Google Generative AI client once
        if self.config.google_api_key:
            try:
                genai.configure(api_key=self.config.google_api_key)
            except Exception:
                pass
        
    def search_products_by_image(self, image_base64: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products based on uploaded image."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            logger.info("Starting image-based product search")
            
            # Step 1: Ask the LLM to describe the image
            image_analysis = self._describe_image_with_llm(image_base64)
            
            if image_analysis.get("error"):
                return {
                    "response": image_analysis.get("response") or "📷 I'm having trouble analyzing the image. Please try another image or add a short description.",
                    "products": [],
                    "intent": "image_search",
                    "confidence": 0.3,
                    "metadata": {
                        "tool_used": "image_product_search",
                        "error": image_analysis["error"]
                    }
                }
            
            # Step 2: Turn that description/attributes into a neutral search query (no static lists)
            search_query = self._extract_search_query_from_analysis(image_analysis.get("description", ""))
            
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
                    "image_analysis": image_analysis,
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
    
    def _describe_image_with_llm(self, image_base64: str) -> Dict[str, Any]:
        """Use the configured LLM to generate a compact, product-focused description.

        The prompt is intentionally generic so it works for any product domain.
        """
        try:
            if not self.config.google_api_key:
                return {"description": "apparel clothing", "response": "", "error": "missing_api_key"}

            # Prepare image part
            b64 = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
            image_bytes = base64.b64decode(b64)
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}

            model_name = getattr(self.config, "llm_model", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            prompt = (
                "You are an e-commerce vision expert. Analyze the image and return STRICT JSON only, no prose.\n"
                "Fields: {\n"
                "  \"item_type\": string,  \n"
                "  \"category\": string,   \n"
                "  \"color\": string,      \n"
                "  \"material\": string,   \n"
                "  \"pattern\": string,    \n"
                "  \"style\": string,      \n"
                "  \"keywords\": [string]  \n"
                "}\n"
                "Rules: be concise, guess only when reasonably visible; never include brand names or prices; lowercase values."
            )
            resp = model.generate_content([prompt, image_part])
            raw = (getattr(resp, "text", None) or "").strip()
            # Extract JSON block if model added formatting
            data = self._safe_parse_json(raw)
            if not isinstance(data, dict) or not data.get("item_type"):
                # Fallback to using raw text if JSON parse fails
                text = raw if raw else "apparel clothing"
                return {"description": text, "response": "", "error": None}
            # Build a compact description string from fields
            parts = [
                data.get("color"),
                data.get("item_type"),
                data.get("category"),
                data.get("material"),
                data.get("pattern"),
                data.get("style"),
            ]
            desc = " ".join([p for p in parts if p]).strip()
            return {"description": desc, "attributes": data, "response": "", "error": None}
        except Exception as e:
            logger.error(f"LLM image description failed: {e}")
            # Fallback to very generic query so pipeline still works
            return {"description": "apparel clothing", "response": "", "error": str(e)}

    def _safe_parse_json(self, text: str) -> Any:
        try:
            if not text:
                return {}
            # Remove code fences if present
            m = re.search(r"\{[\s\S]*\}", text)
            blob = m.group(0) if m else text
            return json.loads(blob)
        except Exception:
            return {}
    
    def _extract_search_query_from_analysis(self, analysis: str) -> str:
        """Extract a search query from image analysis."""
        if not analysis:
            return "clothing apparel"
        
        # Convert analysis to search-friendly terms
        analysis_lower = analysis.lower()
        
        # Extract key product terms
        product_terms = []
        
        # Infer category terms directly from the LLM sentence without static maps
        # by extracting likely nouns (very lightweight approach)
        for token in analysis_lower.split():
            if token.endswith('s'):
                token = token[:-1]
            if token.isalpha() and len(token) >= 3:
                product_terms.append(token)
        
        # Colors (extract words that match common color tokens from the sentence)
        color_tokens = [
            "red","blue","green","black","white","yellow","pink","purple","orange","brown","gray","grey","beige","navy","teal"
        ]
        colors_found = [c for c in color_tokens if c in analysis_lower]
        
        # Styles (simple signal words)
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
        # Keep the enhancement concise
        hint = "Based on your image, I searched similar apparel." if image_description else ""
        return (hint + "\n\n" + base_response) if hint else base_response
    
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
