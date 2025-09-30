"""
Text-Based Product Search Tool
Handles product recommendations and search based on text descriptions.
"""

import logging
from typing import Dict, Any, List, Optional
from src.vector_store.catalog import CatalogStore
from src.config.dynamic_config import get_config

logger = logging.getLogger(__name__)

class TextProductSearchTool:
    """Tool for text-based product search and recommendations."""
    
    def __init__(self, catalog: CatalogStore):
        """Initialize the text product search tool."""
        self.catalog = catalog
        self.config = get_config()
        self.hybrid_search = None  # Deprecated: hybrid search removed
        
    def search_products(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products based on text query."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            logger.info(f"Text product search for query: '{query}'")
            
            # Direct catalog semantic search only (hybrid removed)
            products = self.catalog.search(query, top_k=top_k)
            
            if not products:
                return {
                    "response": "I couldn't find any products matching your request. Please try different search terms or be more specific about what you're looking for.",
                    "products": [],
                    "intent": "product_search",
                    "confidence": 0.3,
                    "metadata": {
                        "tool_used": "text_product_search",
                        "query": query,
                        "search_type": "catalog"
                    }
                }
            
            # Generate response with products
            response = self._generate_product_response(products, query)
            
            return {
                "response": response,
                "products": products,
                "intent": "product_search",
                "confidence": 0.8,
                "metadata": {
                    "tool_used": "text_product_search",
                    "query": query,
                    "products_found": len(products),
                    "search_type": "catalog"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in text product search: {e}")
            return {
                "response": "I encountered an error while searching for products. Please try again with a different query.",
                "products": [],
                "intent": "product_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "tool_used": "text_product_search"}
            }
    
    # Hybrid search removed; direct catalog search is used instead
    
    def _generate_product_response(self, products: List[Dict[str, Any]], query: str) -> str:
        """Generate a response with product recommendations."""
        # Limit to top 3 products only
        top_products = products[:3]
        
        if not top_products:
            return "No relevant products found matching your request."
        
        response = "**Top Recommendations:**\n\n"
        
        for i, product in enumerate(top_products, 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            description = product.get('description', '')
            
            # Get one key relevant feature from description
            key_feature = description[:50] + "..." if len(description) > 50 else description
            
            response += f"{i}. **{name}** - ${price:.2f}\n"
            response += f"   {key_feature}\n\n"
        
        return response
    
    def get_product_categories(self) -> List[str]:
        """Get available product categories."""
        try:
            all_products = self.catalog.get_all_products()
            categories = set()
            
            for product in all_products:
                product_categories = product.get('category', [])
                if isinstance(product_categories, list):
                    categories.update(product_categories)
                elif isinstance(product_categories, str):
                    categories.add(product_categories)
            
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting product categories: {e}")
            return []
    
    def get_brands(self) -> List[str]:
        """Get available product brands."""
        try:
            all_products = self.catalog.get_all_products()
            brands = set()
            
            for product in all_products:
                attributes = product.get('attributes', {})
                brand = attributes.get('brand', '')
                if brand:
                    brands.add(brand)
            
            return sorted(list(brands))
            
        except Exception as e:
            logger.error(f"Error getting brands: {e}")
            return []
    
    def search_by_category(self, category: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products by category."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            query = f"category:{category}"
            return self.search_products(query, top_k)
            
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return {
                "response": f"I couldn't find any products in the {category} category. Please try a different category.",
                "products": [],
                "intent": "product_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "tool_used": "text_product_search"}
            }
    
    def search_by_brand(self, brand: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products by brand."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            query = f"brand:{brand}"
            return self.search_products(query, top_k)
            
        except Exception as e:
            logger.error(f"Error searching by brand: {e}")
            return {
                "response": f"I couldn't find any products from {brand}. Please try a different brand.",
                "products": [],
                "intent": "product_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "tool_used": "text_product_search"}
            }
