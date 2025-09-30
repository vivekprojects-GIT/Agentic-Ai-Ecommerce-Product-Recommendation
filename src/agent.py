"""
Modular Agentic AI Commerce Agent - Fully dynamic and configurable design.
"""

import logging
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_config, setup_logging
# Removed duplicate imports - functionality is now integrated directly
# Simplified search approach - no separate search tool needed
from src.vector_store.catalog import CatalogStore
from src.tools.general_conversation import GeneralConversationTool
from src.tools.text_product_search import TextProductSearchTool
from src.tools.image_product_search import ImageProductSearchTool

# Setup logging with dynamic configuration
setup_logging()
logger = logging.getLogger(__name__)

class ModularAgenticAgent:
    """
    Modular Agentic AI Commerce Agent with clean separation of concerns.
    
    Features:
    - Modular tool system
    - State management
    - Reasoning engine
    - Error recovery
    - Goal-oriented behavior
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the modular agentic agent with dynamic configuration."""
        # Get dynamic configuration
        self.config = get_config()
        
        # Use provided API key or get from config
        self.api_key = api_key or self.config.google_api_key
        
        # Validate configuration
        errors = self.config.validate_config()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Initialize LLM with dynamic configuration
        llm_config = self.config.get_llm_config()
        self.llm = ChatGoogleGenerativeAI(**llm_config)
        
        # Initialize tools with dynamic configuration
        self.catalog = CatalogStore(
            persist_dir=self.config.chroma_persist_dir,
            embedding_model=self.config.embedding_model
        )
        # Initialize specialized tools
        self.general_conversation_tool = GeneralConversationTool()
        self.text_product_search_tool = TextProductSearchTool(self.catalog)
        self.image_product_search_tool = ImageProductSearchTool(
            self.catalog,
            self.text_product_search_tool
        )
        
        # Initialize modular components with dynamic configuration
        self.max_retries = self.config.max_retries
        
        # Create tools for the agent (only if tool calling is enabled)
        if self.config.tool_calling_enabled:
            self.tools = ["general_conversation", "text_product_search", "image_product_search"]
        else:
            self.tools = []
        
        logger.info(f"Modular Agentic Commerce Agent initialized with config: {self.config.llm_model}, retries: {self.config.max_retries}")
    
    def _classify_intent(self, message: str) -> str:
        """Classify user intent using LLM router first, then fallback to rules."""
        try:
            intent = self._route_with_llm(message)
            if intent:
                return intent
        except Exception as e:
            logger.warning(f"LLM routing unavailable, using fallback: {e}")
        # Fallback for reliability
        logger.info("Using fallback intent classification")
        return self._fallback_intent_classification(message)

    def _route_with_llm(self, message: str) -> Optional[str]:
        """Use the LLM to decide which tool to route to based on the user's message.

        Returns one of: "general_chat", "product_search", "image_search".
        """
        router_system = (
            "You are PalonaAI's routing controller. Decide which tool should handle the user's message. "
            "Available tools: general_conversation, text_product_search, image_product_search. "
            "Rules: \n"
            "- Use image_product_search if the user provided an image or asks to analyze a photo/picture.\n"
            "- Use text_product_search for any shopping/product-related intent described in text.\n"
            "- Use general_conversation for greetings, questions about PalonaAI (name/capabilities), or non-shopping chat.\n"
            "- General conversation MUST NOT access the database or include product metadata.\n"
            "Output strict JSON with keys: route (one of the three tools), rationale (short)."
        )
        router_user = f"User message: {message}\nReturn JSON only."

        prompt = ChatPromptTemplate.from_messages([
            ("system", router_system),
            ("user", router_user),
        ])

        chain = prompt | self.llm
        resp = chain.invoke({})
        content = getattr(resp, "content", "") or getattr(resp, "text", "")
        try:
            data = json.loads(content) if isinstance(content, str) else content
            route = (data or {}).get("route", "").strip()
        except Exception:
            # Try to extract a simple token from free-form content
            text = content if isinstance(content, str) else json.dumps(content)
            text_lower = text.lower()
            if "image_product_search" in text_lower:
                route = "image_product_search"
            elif "text_product_search" in text_lower or "product_search" in text_lower:
                route = "text_product_search"
            elif "general_conversation" in text_lower or "general" in text_lower:
                route = "general_conversation"
            else:
                route = ""

        mapping = {
            "general_conversation": "general_chat",
            "text_product_search": "product_search",
            "image_product_search": "image_search",
        }
        return mapping.get(route)
    
    def _fallback_intent_classification(self, message: str) -> str:
        """Fallback rule-based intent classification when LLM is unavailable."""
        message_lower = message.lower().strip()
        logger.info(f"Classifying message: '{message}' -> '{message_lower}'")
        
        # Check for simple responses first
        simple_responses = ["yes", "no", "ok", "okay", "sure", "alright", "thanks", "thank you", "bye", "goodbye"]
        if message_lower in simple_responses:
            logger.info("Classified as general_chat (simple response)")
            return "general_chat"
        
        # Check for specific greeting patterns (exact matches or full phrases)
        greeting_patterns = ["hello", "hi", "hey", "what can you do", "help", "capabilities", "what's your name", "who are you"]
        if any(pattern == message_lower or message_lower.startswith(pattern + " ") for pattern in greeting_patterns):
            logger.info("Classified as general_chat (greeting)")
            return "general_chat"
        
        # Check for product-related keywords (more specific and comprehensive)
        product_keywords = [
            "buy", "shop", "find", "search", "show me", "show", "recommend", "recommendation", "suggest", "suggestion",
            "product", "item", "price", "brand", "color", "size", "clothing", "clothes", "apparel", "wear",
            "shoes", "sneakers", "boots", "sandals", "footwear", "electronics", "gadget", "device",
            "t-shirt", "tshirt", "shirt", "blouse", "top", "jeans", "pants", "trousers", "shorts", "skirt",
            "hoodie", "sweater", "jacket", "coat", "blazer", "dress", "gown", "suit", "uniform",
            "need", "want", "looking for", "looking", "same", "like", "similar", "comparable",
            "athletic", "sport", "sports", "gym", "workout", "fitness", "running", "training",
            "casual", "formal", "business", "party", "wedding", "everyday", "daily"
        ]
        matched_keywords = [keyword for keyword in product_keywords if keyword in message_lower]
        if matched_keywords:
            logger.info(f"Classified as product_search (matched keywords: {matched_keywords})")
            return "product_search"
        
        # Check for image-related keywords
        image_keywords = ["image", "photo", "picture", "upload", "uploaded", "see this", "analyze this", "what's in this"]
        if any(keyword in message_lower for keyword in image_keywords):
            logger.info("Classified as image_search")
            return "image_search"
        
        # Check for comparison keywords
        comparison_keywords = ["compare", "vs", "versus", "better", "difference", "which is better", "which one"]
        if any(keyword in message_lower for keyword in comparison_keywords):
            logger.info("Classified as product_comparison")
            return "product_comparison"
        
        # Check for general questions (likely general chat)
        question_words = ["what", "how", "when", "where", "why", "can you", "do you", "are you", "tell me about"]
        if any(word in message_lower for word in question_words):
            logger.info("Classified as general_chat (question)")
            return "general_chat"
        
        # Default to general chat for unclear queries
        logger.info("Classified as general_chat (default)")
        return "general_chat"
    
    def _hybrid_search_products(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform hybrid search using BM25 + Semantic + LLM validation for optimal results."""
        try:
            # Get all products from catalog for hybrid search
            all_products = self.catalog.get_all_products()
            
            if not all_products:
                logger.warning("No products available for hybrid search")
                return []
            
            # Initialize hybrid search system
            from src.tools.hybrid_search import HybridSearchSystem
            hybrid_search = HybridSearchSystem(
                api_key=self.config.google_api_key,
                embedding_model=self.catalog.embedding_model
            )
            
            # Load products into hybrid search
            hybrid_search.load_products(all_products)
            
            # Perform hybrid search with optimal parameters
            products = hybrid_search.hybrid_search(query, top_k=top_k)
            
            logger.info(f"Hybrid search found {len(products)} products for query: '{query}'")
            return products
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to direct catalog search
            return self.catalog.search(query, top_k=top_k)
    
    def _create_plan(self, message: str, intent: str) -> Dict[str, Any]:
        """Create a plan based on user input and intent."""
        
        if intent == "product_search":
            return {
                "goal": "Find and recommend products",
                "steps": ["search_products", "present_results"],
                "tools_needed": ["search_products"],
                "intent": intent
            }
        elif intent == "image_search":
            return {
                "goal": "Analyze image and find similar products",
                "steps": ["analyze_image", "search_products", "present_results"],
                "tools_needed": ["analyze_image", "search_products"],
                "intent": intent
            }
        elif intent == "product_comparison":
            return {
                "goal": "Compare products",
                "steps": ["search_products", "present_results"],
                "tools_needed": ["search_products"],
                "intent": intent
            }
        elif intent == "product_details":
            return {
                "goal": "Get detailed product information",
                "steps": ["search_products", "present_results"],
                "tools_needed": ["search_products"],
                "intent": intent
            }
        else:
            return {
                "goal": "Provide helpful response",
                "steps": ["generate_response"],
                "tools_needed": [],
                "intent": intent
            }
    
    def process_request(self, message: str, image_base64: Optional[str] = None, conversation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request using the modular agentic system with conversation memory."""
        try:
            # Create initial state with conversation context
            state = {
                "messages": [{"role": "user", "content": message}],
                "current_goal": None,
                "reasoning_steps": [],
                "tools_used": [],
                "products_found": [],
                "error_count": 0,
                "max_retries": self.max_retries,
                "user_intent": None,
                "confidence": 0.0,
                "metadata": {},
                "conversation_context": conversation_context or {}
            }
            
            # Add conversation history to messages if available
            if conversation_context and "conversation_history" in conversation_context:
                recent_history = conversation_context["conversation_history"]
                if recent_history:
                    # Add recent conversation context to state
                    state["messages"] = []
                    context_limit = getattr(self.config, 'conversation_context_limit', 3)
                    for interaction in recent_history[-context_limit:]:  # Configurable context limit
                        state["messages"].extend([
                            {"role": "user", "content": interaction.get("user_input", "")},
                            {"role": "assistant", "content": interaction.get("agent_response", "")}
                        ])
                    # Add current message
                    state["messages"].append({"role": "user", "content": message})
            
            # Classify intent
            intent = self._classify_intent(message)
            state["user_intent"] = intent
            
            # Create plan
            plan = self._create_plan(message, intent)
            state["current_goal"] = plan["goal"]
            
            # Execute plan
            result = self._execute_plan(state, plan, message, image_base64)
            
            return result
            
        except Exception as e:
            logger.error(f"Modular agentic processing error: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "products": [],
                "intent": "error",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _execute_plan(self, state: Dict[str, Any], plan: Dict[str, Any], message: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """Execute the planned actions using specialized tools."""
        try:
            intent = state.get("user_intent", "general_chat")
            goal = plan.get("goal", "Provide helpful response")
            
            # Route to appropriate specialized tool based on intent
            if intent == "general_chat":
                return self._handle_general_conversation(state, message)
            elif intent == "product_search":
                return self._handle_text_product_search(state, message)
            elif intent == "image_search":
                return self._handle_image_product_search(state, message, image_base64)
            elif intent == "product_comparison":
                return self._handle_product_comparison(state, message)
            else:
                # Default to general conversation
                return self._handle_general_conversation(state, message)
            
        except Exception as e:
            logger.error(f"Plan execution error: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "products": [],
                "intent": state.get("user_intent", "error"),
                "confidence": 0.3,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _handle_general_conversation(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Handle general conversation using the specialized tool."""
        try:
            conversation_context = state.get("conversation_context", {})
            result = self.general_conversation_tool.handle_conversation(message, conversation_context)
            
            # Update state
            state["tools_used"] = state.get("tools_used", []) + ["general_conversation"]
            
            return {
                "response": result["response"],
                "products": [],
                "intent": "general_chat",
                "confidence": result["confidence"],
                # General chat should not include DB-derived metadata; keep minimal
                "metadata": {}
            }
            
        except Exception as e:
            logger.error(f"Error in general conversation: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your message. Could you please try again?",
                "products": [],
                "intent": "general_chat",
                "confidence": 0.3,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _handle_text_product_search(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Handle text-based product search using the specialized tool."""
        try:
            result = self.text_product_search_tool.search_products(message, self.config.search_top_k)
            
            # Update state
            state["tools_used"] = state.get("tools_used", []) + ["text_product_search"]
            state["products_found"] = result["products"]
            
            return {
                "response": result["response"],
                "products": result["products"],
                "intent": "product_search",
                "confidence": result["confidence"],
                "metadata": {
                    "goal": "Find and recommend products",
                    "tools_used": state.get("tools_used", []),
                    "reasoning_steps": state.get("reasoning_steps", []),
                    "agentic": True,
                    "modular": True,
                    "tool_metadata": result.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in text product search: {e}")
            return {
                "response": "I encountered an error while searching for products. Please try again with a different query.",
                "products": [],
                "intent": "product_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _handle_image_product_search(self, state: Dict[str, Any], message: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """Handle image-based product search using the specialized tool."""
        try:
            if not image_base64:
                return {
                    "response": "I need an image to perform image-based product search. Please upload an image or describe what you're looking for in text.",
                    "products": [],
                    "intent": "image_search",
                    "confidence": 0.3,
                    "metadata": {"error": "No image provided", "agentic": True, "modular": True}
                }
            
            result = self.image_product_search_tool.search_products_by_image(image_base64, self.config.search_top_k)
            
            # Update state
            state["tools_used"] = state.get("tools_used", []) + ["image_product_search"]
            state["products_found"] = result["products"]
            
            return {
                "response": result["response"],
                "products": result["products"],
                "intent": "image_search",
                "confidence": result["confidence"],
                "metadata": {
                    "goal": "Analyze image and find similar products",
                    "tools_used": state.get("tools_used", []),
                    "reasoning_steps": state.get("reasoning_steps", []),
                    "agentic": True,
                    "modular": True,
                    "tool_metadata": result.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image product search: {e}")
            return {
                "response": "I encountered an error while analyzing your image. Please try uploading a different image or describe what you're looking for in text.",
                "products": [],
                "intent": "image_search",
                "confidence": 0.3,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _handle_product_comparison(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Handle product comparison requests."""
        try:
            # For now, treat comparison as a text search with comparison context
            result = self.text_product_search_tool.search_products(message, self.config.search_top_k)
            
            # Enhance response for comparison
            if result["products"]:
                comparison_response = f"Here are some products for comparison based on your request:\n\n{result['response']}\n\nI can help you compare specific features, prices, or other aspects of these products. What would you like to know more about?"
            else:
                comparison_response = "I couldn't find products to compare. Please try a more specific search term or describe what you'd like to compare."
            
            # Update state
            state["tools_used"] = state.get("tools_used", []) + ["text_product_search"]
            state["products_found"] = result["products"]
            
            return {
                "response": comparison_response,
                "products": result["products"],
                "intent": "product_comparison",
                "confidence": result["confidence"],
                "metadata": {
                    "goal": "Compare products",
                    "tools_used": state.get("tools_used", []),
                    "reasoning_steps": state.get("reasoning_steps", []),
                    "agentic": True,
                    "modular": True,
                    "tool_metadata": result.get("metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in product comparison: {e}")
            return {
                "response": "I encountered an error while searching for products to compare. Please try again with a different query.",
                "products": [],
                "intent": "product_comparison",
                "confidence": 0.3,
                "metadata": {"error": str(e), "agentic": True, "modular": True}
            }
    
    def _generate_response(self, state: Dict[str, Any], products: List[Dict[str, Any]], response_parts: List[str], goal: str) -> str:
        """Generate a response using proper RAG: Retrieved Products + Query → LLM → Beautiful Response."""
        try:
            # Handle general conversation
            if goal == "Provide helpful response":
                return self._generate_general_response(state)
            
            # Handle no products found
            if not products:
                return "No products found in our catalog matching your request. Please try different search terms."
            
            # Get the original query
            messages = state.get("messages", [])
            original_query = messages[0].get("content", "") if messages else ""
            
            # Format products for LLM
            products_text = self._format_products_for_llm(products)
            
            # Limit to top 3 most relevant products only
            top_products = products[:3]
            products_text = self._format_products_for_llm(top_products)
            
            # Create optimized RAG prompt for precise recommendations
            rag_prompt = f"""
You are a professional shopping assistant. Provide EXACTLY the top 3 most relevant product recommendations.

User Query: "{original_query}"

Top 3 Products:
{products_text}

CRITICAL INSTRUCTIONS:
1. Show EXACTLY 3 products - no more, no less
2. Only recommend products that are HIGHLY RELEVANT to the user's query
3. If fewer than 3 relevant products exist, show only the relevant ones
4. Do NOT include unrelated products
5. Be concise and direct
6. Focus on product name, price, and key relevant features only
7. No extra information, tips, or general advice

Format your response as:
**Top Recommendations:**

1. [Product Name] - $[Price]
   [One key relevant feature]

2. [Product Name] - $[Price]  
   [One key relevant feature]

3. [Product Name] - $[Price]
   [One key relevant feature]

Response:
"""
            
            # Skip LLM generation due to quota limits - use fallback directly
            logger.info("Using fallback response generation (LLM quota exceeded)")
            # Ensure we only use top 3 products
            limited_products = top_products[:3]
            return self._generate_simple_product_response(limited_products, response_parts)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I found some products but had trouble formatting the response. Please try again."
    
    def _format_products_for_llm(self, products: List[Dict[str, Any]]) -> str:
        """Format products for LLM consumption in RAG."""
        if not products:
            return "No products found."
        
        formatted_products = []
        for product in products:
            name = product.get('name', 'N/A')
            brand = product.get('attributes', {}).get('brand', 'N/A')
            price = product.get('price', 0)
            color = product.get('attributes', {}).get('color_family', 'N/A')
            description = product.get('description', 'N/A')
            
            formatted_products.append(
                f"- Name: {name}, Brand: {brand}, Price: ${price:.2f}, Color: {color}, Description: {description}"
            )
        
        return "\n".join(formatted_products)
    
    def _generate_simple_product_response(self, products: List[Dict[str, Any]], response_parts: List[str]) -> str:
        """Fallback simple product response if LLM fails - shows only top 3."""
        # Limit to top 3 products only
        top_products = products[:3]
        
        if not top_products:
            return "No relevant products found matching your request."
        
        response = "**Top Recommendations:**\n\n"
        
        for i, product in enumerate(top_products, 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            brand = product.get('attributes', {}).get('brand', 'Unknown Brand')
            color = product.get('attributes', {}).get('color_family', 'Unknown Color')
            description = product.get('description', '')
            
            # Get one key relevant feature from description
            key_feature = description[:50] + "..." if len(description) > 50 else description
            
            response += f"{i}. **{name}** - ${price:.2f}\n"
            response += f"   {key_feature}\n\n"
        
        return response
    
    def _generate_general_response(self, state: Dict[str, Any]) -> str:
        """Generate responses for general conversation with enhanced prompt engineering."""
        messages = state.get("messages", [])
        if not messages:
            return f"Hello! I'm your {self.config.agent_name}. How can I help you today?"
        
        # Get the latest user message
        user_message = messages[0].get("content", "") if messages else ""
        
        # Get dynamic response messages
        response_messages = self.config.get_response_messages()
        
        # Simple responses
        if any(word in user_message.lower() for word in ["name", "who are you"]):
            return f"""Hi! I'm your AI shopping assistant. I can help you find products from our catalog using text search or image analysis. What would you like to find?"""
        
        elif any(word in user_message.lower() for word in ["what can you do", "help", "capabilities"]):
            return """I can help you find products from our catalog. You can:
• Search for products using text descriptions
• Upload an image to find similar products
• Ask questions about our products

Just tell me what you're looking for or upload an image to get started!"""
        
        elif any(word in user_message.lower() for word in ["hello", "hi", "hey"]):
            return """Hello! 👋 I'm your AI shopping assistant. I'm here to help you find amazing products, answer your questions, and make your shopping experience great. 

I use advanced AI technology to understand exactly what you're looking for, whether you describe it in words or show me a picture. What can I help you with today?"""
        
        elif any(word in user_message.lower() for word in ["yes", "ok", "okay", "sure", "alright"]):
            # Check if we recently showed products
            recent_products = state.get("products_found", [])
            if recent_products:
                return "Great! Those are some excellent options. Would you like me to help you with anything else, or do you have questions about any of those products?"
            else:
                return "Perfect! What would you like to explore today? I can help you find products, answer questions, or assist with your shopping needs."
        
        elif any(word in user_message.lower() for word in ["no", "nope"]):
            return "No problem! Is there something else I can help you with? I'm here to assist with any shopping questions or product searches you might have."
        
        elif any(word in user_message.lower() for word in self.config.get_keyword_lists()["price"]):
            return """I'd be happy to help you find products within your budget! 💰
            
            Please specify what type of product you're looking for along with your price range. For example:
            • "Show me gym t-shirts under $15"
            • "Find running shoes under $50"
            • "I need a water bottle under $10"
            • "Casual wear under $20"
            
            Just tell me the product type and your budget, and I'll find the best options for you!"""
        
        elif any(word in user_message.lower() for word in self.config.get_keyword_lists()["gym"]):
            return """💪 Perfect! I can help you find great gym and fitness gear!
            
            I have athletic wear, workout clothes, sports equipment, and fitness accessories. Just tell me what you need:
            • "Show me gym t-shirts"
            • "I need workout shorts"
            • "Find running shoes"
            • "Athletic wear for women"
            • "Gym accessories under $20"
            
            What specific gym gear are you looking for?"""
        
        elif any(word in user_message.lower() for word in self.config.get_keyword_lists()["casual"]):
            return """👕 Great choice! I can help you find comfortable casual wear!
            
            I have a great selection of everyday clothes perfect for relaxed style:
            • "Show me casual t-shirts"
            • "I need everyday jeans"
            • "Find comfortable sneakers"
            • "Casual wear for summer"
            • "Relaxed fit clothing"
            
            What type of casual wear are you looking for?"""
        
        else:
            return """I'm your AI shopping assistant! I can help you find products, answer questions about our catalog, and assist with your shopping needs. 

Here are some ways I can help:
• **Search for products**: "Show me athletic wear" or "Find me a water bottle"
• **Upload an image**: Take a photo to find similar products
• **Get recommendations**: Tell me what you need and I'll suggest the best options
• **Compare products**: I can help you choose between different items

What would you like to explore today?"""
    
    def process_request_stream(self, message: str, image_base64: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Process request with streaming (simplified for now)."""
        result = self.process_request(message, image_base64)
        
        # Yield the result as a single chunk
        yield {
            "type": "response",
            "response": result["response"],
            "products": result["products"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "metadata": result["metadata"]
        }
    
