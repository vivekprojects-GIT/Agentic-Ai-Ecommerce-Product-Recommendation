"""
Text-Based Product Search Tool
Handles product recommendations and search based on text descriptions.
"""

import logging
from typing import Dict, Any, List, Optional
from src.vector_store.catalog import CatalogStore
from src.config.dynamic_config import get_config
import re
import string

logger = logging.getLogger(__name__)

class TextProductSearchTool:
    """Tool for text-based product search and recommendations."""
    
    def __init__(self, catalog: CatalogStore):
        """Initialize the text product search tool."""
        self.catalog = catalog
        self.config = get_config()
        self.hybrid_search = None  # Deprecated: hybrid search removed
        self._vocab_cache: Dict[str, List[str]] = {}

    def _build_dynamic_vocab(self) -> Dict[str, List[str]]:
        """Build vocab (colors, brands, categories) from catalog at runtime.

        This avoids relying on any static lists or config. Results are cached
        for the lifetime of the tool instance.
        """
        if self._vocab_cache:
            return self._vocab_cache

        all_products = self.catalog.get_all_products()
        colors, brands, categories = set(), set(), set()
        for p in all_products:
            attrs = p.get("attributes", {}) or {}
            c = attrs.get("color_family")
            if isinstance(c, str) and c:
                colors.add(c.strip().lower())
            b = attrs.get("brand")
            if isinstance(b, str) and b:
                brands.add(b.strip().lower())
            cats = p.get("category") or []
            for cat in cats if isinstance(cats, list) else [cats]:
                if isinstance(cat, str) and cat:
                    categories.add(cat.strip().lower())

        self._vocab_cache = {
            "colors": sorted(colors),
            "brands": sorted(brands),
            "categories": sorted(categories),
        }
        return self._vocab_cache

    def _normalize(self, s: str) -> str:
        table = str.maketrans('', '', string.punctuation)
        return (s or "").lower().translate(table)

    def _simple_keyword_match(self, query: str, limit: int) -> List[Dict[str, Any]]:
        qn = self._normalize(query)
        q_tokens = [t for t in qn.split() if t]
        if not q_tokens:
            return []
        all_products = self.catalog.get_all_products()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for p in all_products:
            name = self._normalize(p.get("name", ""))
            desc = self._normalize(p.get("description", ""))
            text = f"{name} {desc} {self._normalize(p.get('search_text',''))}"
            # simple token overlap score
            hits = sum(1 for t in q_tokens if t in text)
            if hits:
                scored.append((hits / len(q_tokens), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]
        
    def search_products(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Search for products based on text query."""
        try:
            if top_k is None:
                top_k = self.config.search_top_k
                
            logger.info(f"Text product search for query: '{query}'")
            
            # Parse lightweight structured constraints (price, color)
            constraints: Dict[str, Any] = {}
            q_lower = (query or "").lower()

            # price: "under 10", "below $15", "<= 20"
            m = re.search(r"(?:under|below|<=|less than)\s*\$?\s*(\d+)", q_lower)
            if m:
                try:
                    constraints["price_max"] = float(m.group(1))
                    constraints.setdefault("price_min", 0.0)
                except Exception:
                    pass

            # color detection (dynamic from catalog metadata)
            vocab = self._build_dynamic_vocab()
            for c in vocab.get("colors", []):
                if re.search(fr"\b{re.escape(c)}\b", q_lower):
                    constraints["color_family"] = c.title()
                    break

            # Direct catalog semantic search with optional metadata filters
            products = self.catalog.search(query, top_k=top_k * 2, filters=constraints if constraints else None)

            # Additional post-filter to enforce price/color if present in free text but not metadata
            def passes(p: Dict[str, Any]) -> bool:
                if "price_max" in constraints and p.get("price", 0) > constraints["price_max"]:
                    return False
                if "color_family" in constraints:
                    meta_color = p.get("attributes", {}).get("color_family", "")
                    name_desc = (p.get("name","") + " " + p.get("description",""))
                    if constraints["color_family"].lower() not in (meta_color or "").lower() \
                       and constraints["color_family"].lower() not in name_desc.lower():
                        return False
                return True

            products = [p for p in products if passes(p)][:top_k]

            # Fallback: exact/substring keyword match if semantic returns none
            if not products:
                fallback = self._simple_keyword_match(query, top_k)
                if fallback:
                    products = fallback
            
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
            url = product.get('url', '')
            
            # Get one key relevant feature from description
            key_feature = description[:50] + "..." if len(description) > 50 else description
            
            display_name = f"[{name}]({url})" if url else name
            response += f"{i}. **{display_name}** - ${price:.2f}\n"
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
