"""
Dynamic configuration system using environment variables.
All static values are replaced with configurable environment variables.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class DynamicConfig:
    """Dynamic configuration class that reads all values from environment variables."""
    
    # API Configuration
    api_port: int = int(os.getenv("API_PORT", "8080"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
    gradio_host: str = os.getenv("GRADIO_HOST", "0.0.0.0")
    
    # Model Configuration
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    # Database Configuration
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_force_reload: bool = os.getenv("CHROMA_FORCE_RELOAD", "0") == "1"
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "commerce_products")
    
    # Search Configuration
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "3"))
    search_similarity_threshold: float = float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.7"))
    candidate_multiplier: int = int(os.getenv("CANDIDATE_MULTIPLIER", "5"))
    max_catalog_products: int = int(os.getenv("MAX_CATALOG_PRODUCTS", "10000"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    ann_weight: float = float(os.getenv("ANN_WEIGHT", "0.7"))
    
    # Agent Configuration
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    conversation_context_limit: int = int(os.getenv("CONVERSATION_CONTEXT_LIMIT", "3"))
    reasoning_enabled: bool = os.getenv("REASONING_ENABLED", "1") == "1"
    tool_calling_enabled: bool = os.getenv("TOOL_CALLING_ENABLED", "1") == "1"
    
    # UI Configuration
    ui_title: str = os.getenv("UI_TITLE", "Shopping Assistant")
    ui_description: str = os.getenv("UI_DESCRIPTION", "Your AI-powered shopping companion")
    ui_theme: str = os.getenv("UI_THEME", "soft")
    ui_max_width: int = int(os.getenv("UI_MAX_WIDTH", "3200"))
    ui_chat_height: int = int(os.getenv("UI_CHAT_HEIGHT", "1800"))
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Performance Configuration
    enable_caching: bool = os.getenv("ENABLE_CACHING", "1") == "1"
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Feature Flags
    enable_image_analysis: bool = os.getenv("ENABLE_IMAGE_ANALYSIS", "1") == "1"
    enable_product_comparison: bool = os.getenv("ENABLE_PRODUCT_COMPARISON", "1") == "1"
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "1") == "1"
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "1") == "1"
    
    # Dynamic Keyword Lists (comma-separated)
    gym_keywords: str = os.getenv("GYM_KEYWORDS", "gym,workout,fitness,exercise,training,athletic,sports,running,jogging")
    casual_keywords: str = os.getenv("CASUAL_KEYWORDS", "casual,everyday,daily,relaxed,comfort")
    formal_keywords: str = os.getenv("FORMAL_KEYWORDS", "formal,business,office,professional,dress")
    outdoor_keywords: str = os.getenv("OUTDOOR_KEYWORDS", "outdoor,hiking,camping,adventure,nature")
    price_keywords: str = os.getenv("PRICE_KEYWORDS", "under,budget,cheap,affordable,dollar,price,cost")
    search_keywords: str = os.getenv("SEARCH_KEYWORDS", "find,search,look for,need,want,recommend,show me,suggest")
    apparel_keywords: str = os.getenv("APPAREL_KEYWORDS", "shirt,t-shirt,shorts,pants,jacket,sweater,hoodie,cap,hat")
    sports_keywords: str = os.getenv("SPORTS_KEYWORDS", "shoes,sneakers,cleats,baseball,football,basketball,soccer")
    comparison_keywords: str = os.getenv("COMPARISON_KEYWORDS", "compare,vs,versus,difference")
    details_keywords: str = os.getenv("DETAILS_KEYWORDS", "details,info,about,specs")
    image_keywords: str = os.getenv("IMAGE_KEYWORDS", "image,photo")
    
    # Dynamic Intent Types
    product_search_intent: str = os.getenv("PRODUCT_SEARCH_INTENT", "product_search")
    product_comparison_intent: str = os.getenv("PRODUCT_COMPARISON_INTENT", "product_comparison")
    product_details_intent: str = os.getenv("PRODUCT_DETAILS_INTENT", "product_details")
    image_search_intent: str = os.getenv("IMAGE_SEARCH_INTENT", "image_search")
    general_chat_intent: str = os.getenv("GENERAL_CHAT_INTENT", "general_chat")
    
    # Dynamic Response Messages
    agent_name: str = os.getenv("AGENT_NAME", "AI shopping assistant")
    agent_capabilities: str = os.getenv("AGENT_CAPABILITIES", "find products, analyze images, and answer questions about our catalog")
    agent_greeting: str = os.getenv("AGENT_GREETING", "Hello! I'm your AI shopping assistant. I'm here to help you find amazing products, answer your questions, and make your shopping experience great.")
    
    # Dynamic Material Descriptions
    active_wear_material: str = os.getenv("ACTIVE_WEAR_MATERIAL", "Perfect for active wear")
    professional_material: str = os.getenv("PROFESSIONAL_MATERIAL", "Professional quality")
    outdoor_material: str = os.getenv("OUTDOOR_MATERIAL", "Durable for outdoor use")
    
    # Dynamic Context Responses
    gym_intro: str = os.getenv("GYM_INTRO", "💪 Perfect for your gym workouts!")
    running_intro: str = os.getenv("RUNNING_INTRO", "🏃‍♂️ Great running gear!")
    casual_intro: str = os.getenv("CASUAL_INTRO", "👕 Perfect for everyday wear!")
    formal_intro: str = os.getenv("FORMAL_INTRO", "👔 Professional attire!")
    outdoor_intro: str = os.getenv("OUTDOOR_INTRO", "🏔️ Adventure-ready gear!")
    price_intro: str = os.getenv("PRICE_INTRO", "💰 Budget-friendly options!")
    
    # Dynamic Error Messages
    api_quota_error: str = os.getenv("API_QUOTA_ERROR", "I apologize, but I've reached my daily API quota limit. Please try again later or contact support.")
    image_analysis_error: str = os.getenv("IMAGE_ANALYSIS_ERROR", "I'm unable to analyze images at the moment. Please describe what you're looking for in text.")
    search_error: str = os.getenv("SEARCH_ERROR", "I encountered an error while searching for products. Please try again with different terms.")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary."""
        return {
            "model": self.llm_model,
            "google_api_key": self.google_api_key,
            "temperature": self.llm_temperature,
            "max_output_tokens": self.llm_max_tokens
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration as dictionary."""
        return {
            "top_k": self.search_top_k,
            "similarity_threshold": self.search_similarity_threshold,
            "bm25_weight": self.bm25_weight,
            "ann_weight": self.ann_weight
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration as dictionary."""
        return {
            "title": self.ui_title,
            "description": self.ui_description,
            "theme": self.ui_theme,
            "max_width": self.ui_max_width,
            "chat_height": self.ui_chat_height
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration as dictionary."""
        return {
            "level": self.log_level,
            "format": self.log_format
        }
    
    def get_keyword_lists(self) -> Dict[str, List[str]]:
        """Get all keyword lists as dictionaries."""
        return {
            "gym": [k.strip() for k in self.gym_keywords.split(",")],
            "casual": [k.strip() for k in self.casual_keywords.split(",")],
            "formal": [k.strip() for k in self.formal_keywords.split(",")],
            "outdoor": [k.strip() for k in self.outdoor_keywords.split(",")],
            "price": [k.strip() for k in self.price_keywords.split(",")],
            "search": [k.strip() for k in self.search_keywords.split(",")],
            "apparel": [k.strip() for k in self.apparel_keywords.split(",")],
            "sports": [k.strip() for k in self.sports_keywords.split(",")],
            "comparison": [k.strip() for k in self.comparison_keywords.split(",")],
            "details": [k.strip() for k in self.details_keywords.split(",")],
            "image": [k.strip() for k in self.image_keywords.split(",")]
        }
    
    def get_intent_types(self) -> Dict[str, str]:
        """Get all intent types as dictionary."""
        return {
            "product_search": self.product_search_intent,
            "product_comparison": self.product_comparison_intent,
            "product_details": self.product_details_intent,
            "image_search": self.image_search_intent,
            "general_chat": self.general_chat_intent
        }
    
    def get_response_messages(self) -> Dict[str, str]:
        """Get all response messages as dictionary."""
        return {
            "agent_name": self.agent_name,
            "agent_capabilities": self.agent_capabilities,
            "agent_greeting": self.agent_greeting,
            "gym_intro": self.gym_intro,
            "running_intro": self.running_intro,
            "casual_intro": self.casual_intro,
            "formal_intro": self.formal_intro,
            "outdoor_intro": self.outdoor_intro,
            "price_intro": self.price_intro,
            "active_wear_material": self.active_wear_material,
            "professional_material": self.professional_material,
            "outdoor_material": self.outdoor_material
        }
    
    def get_error_messages(self) -> Dict[str, str]:
        """Get all error messages as dictionary."""
        return {
            "api_quota_error": self.api_quota_error,
            "image_analysis_error": self.image_analysis_error,
            "search_error": self.search_error
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")
        
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("API_PORT must be between 1 and 65535")
        
        if self.gradio_port < 1 or self.gradio_port > 65535:
            errors.append("GRADIO_PORT must be between 1 and 65535")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            errors.append("LLM_TEMPERATURE must be between 0 and 2")
        
        if self.max_retries < 0:
            errors.append("MAX_RETRIES must be non-negative")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        return errors

# Global configuration instance
config = DynamicConfig()

def get_config() -> DynamicConfig:
    """Get the global configuration instance."""
    return config

def reload_config() -> DynamicConfig:
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)  # Override existing environment variables
    config = DynamicConfig()
    return config

def setup_logging():
    """Setup logging based on configuration."""
    logging_config = config.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config["level"].upper()),
        format=logging_config["format"]
    )
