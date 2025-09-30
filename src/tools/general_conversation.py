"""
General Conversation Tool for PalonaAI
Handles general questions, greetings, and conversational interactions.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class GeneralConversationTool:
    """Tool for handling general conversation with PalonaAI."""
    
    def __init__(self):
        """Initialize the general conversation tool."""
        self.agent_name = "PalonaAI"
        self.capabilities = [
            "Product recommendations and search",
            "Image-based product analysis", 
            "General shopping assistance",
            "Product comparisons",
            "Answering questions about our catalog"
        ]
        
        # Predefined Q&A pairs for common questions
        self.qa_pairs = {
            "what's your name": f"I'm {self.agent_name}, your AI shopping assistant! I'm here to help you find the perfect products.",
            "who are you": f"I'm {self.agent_name}, your intelligent shopping companion. I specialize in helping you discover amazing products through text descriptions or image analysis.",
            "what can you do": f"I can help you with several things:\n• **Product Search**: Find specific items like 'red t-shirts' or 'running shoes'\n• **Image Analysis**: Upload photos to find similar products\n• **Recommendations**: Get personalized suggestions based on your needs\n• **Product Information**: Learn about features, prices, and availability\n• **Comparisons**: Help you choose between different products",
            "how do you work": "I use advanced AI technology to understand exactly what you're looking for. You can describe products in words or show me pictures, and I'll search through our catalog to find the best matches for you.",
            "what products do you have": "I have access to a wide range of products including clothing, shoes, electronics, and accessories. I can help you find specific items or browse by category.",
            "can you help me shop": "Absolutely! I'm here to make your shopping experience great. Just tell me what you're looking for or upload an image, and I'll help you find the perfect products.",
            "what's your specialty": f"My specialty is understanding your shopping needs and finding the best products for you. Whether you describe what you want or show me a picture, I'll use my AI capabilities to provide accurate recommendations.",
            "how accurate are your recommendations": "I use advanced semantic search and AI validation to ensure my recommendations are highly relevant to what you're looking for. I focus on finding products that truly match your needs.",
            "can you compare products": "Yes! I can help you compare different products, explain their features, and help you make informed decisions about which items best suit your needs.",
            "do you remember our conversation": "I maintain context throughout our conversation to provide better, more personalized recommendations as we chat.",
            "what if i don't like the recommendations": "No problem! Just let me know what you'd like to change - maybe different colors, styles, or price ranges. I'll adjust my search to find better options for you.",
            "can you help with sizing": "While I can show you available sizes for products, I'd recommend checking the specific sizing charts for each item to ensure the best fit for you.",
            "what about returns or exchanges": "For specific return and exchange policies, I'd recommend checking with the retailer directly, as policies can vary by product and seller.",
            "how do i place an order": "I can help you find the perfect products, but for placing orders, you'll need to visit the retailer's website or store directly.",
            "are your recommendations personalized": "Yes! I learn from our conversation to provide more relevant recommendations. The more you tell me about your preferences, the better I can help you find what you're looking for."
        }
    
    def handle_conversation(self, message: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle general conversation and return appropriate response."""
        try:
            message_lower = message.lower().strip()
            
            # Handle greetings
            if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                return self._handle_greeting(conversation_context)
            
            # Handle simple responses
            elif message_lower in ["yes", "ok", "okay", "sure", "alright"]:
                return self._handle_affirmative_response(conversation_context)
            
            elif message_lower in ["no", "nope"]:
                return self._handle_negative_response(conversation_context)
            
            # Handle thanks
            elif any(thanks in message_lower for thanks in ["thanks", "thank you", "appreciate it"]):
                return self._handle_thanks()
            
            # Handle goodbye
            elif any(goodbye in message_lower for goodbye in ["bye", "goodbye", "see you", "talk to you later"]):
                return self._handle_goodbye()
            
            # Check for predefined Q&A
            for question, answer in self.qa_pairs.items():
                if question in message_lower:
                    return {
                        "response": answer,
                        "intent": "general_chat",
                        "confidence": 0.9,
                        "metadata": {
                            "tool_used": "general_conversation",
                            "qa_match": question
                        }
                    }
            
            # Handle general questions
            if any(word in message_lower for word in ["what", "how", "when", "where", "why", "can you", "do you", "are you", "tell me about"]):
                return self._handle_general_question(message, conversation_context)
            
            # Default response
            return self._handle_default_response(message, conversation_context)
            
        except Exception as e:
            logger.error(f"Error in general conversation: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your message. Could you please try again?",
                "intent": "general_chat",
                "confidence": 0.5,
                "metadata": {"error": str(e), "tool_used": "general_conversation"}
            }
    
    def _handle_greeting(self, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle greeting messages."""
        recent_products = conversation_context.get("products_found", []) if conversation_context else []
        
        if recent_products:
            response = f"Hello! 👋 Welcome back! I see we were looking at some products earlier. Would you like to continue exploring those options, or is there something new you'd like to find today?"
        else:
            response = f"Hello! 👋 I'm {self.agent_name}, your AI shopping assistant. I'm here to help you find amazing products, answer your questions, and make your shopping experience great.\n\nI use advanced AI technology to understand exactly what you're looking for, whether you describe it in words or show me a picture. What can I help you with today?"
        
        return {
            "response": response,
            "intent": "general_chat",
            "confidence": 0.9,
            "metadata": {"tool_used": "general_conversation", "type": "greeting"}
        }
    
    def _handle_affirmative_response(self, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle affirmative responses like 'yes', 'ok', etc."""
        recent_products = conversation_context.get("products_found", []) if conversation_context else []
        
        if recent_products:
            response = "Great! Those are some excellent options. Would you like me to help you with anything else, or do you have questions about any of those products?"
        else:
            response = "Perfect! What would you like to explore today? I can help you find products, answer questions, or assist with your shopping needs."
        
        return {
            "response": response,
            "intent": "general_chat",
            "confidence": 0.8,
            "metadata": {"tool_used": "general_conversation", "type": "affirmative"}
        }
    
    def _handle_negative_response(self, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle negative responses like 'no', 'nope', etc."""
        return {
            "response": "No problem! Is there something else I can help you with? I'm here to assist with any shopping questions or product searches you might have.",
            "intent": "general_chat",
            "confidence": 0.8,
            "metadata": {"tool_used": "general_conversation", "type": "negative"}
        }
    
    def _handle_thanks(self) -> Dict[str, Any]:
        """Handle thank you messages."""
        return {
            "response": "You're welcome! 😊 Is there anything else I can help you with today?",
            "intent": "general_chat",
            "confidence": 0.9,
            "metadata": {"tool_used": "general_conversation", "type": "thanks"}
        }
    
    def _handle_goodbye(self) -> Dict[str, Any]:
        """Handle goodbye messages."""
        return {
            "response": f"Goodbye! 👋 Thanks for shopping with {self.agent_name} today. Feel free to come back anytime you need help finding the perfect products!",
            "intent": "general_chat",
            "confidence": 0.9,
            "metadata": {"tool_used": "general_conversation", "type": "goodbye"}
        }
    
    def _handle_general_question(self, message: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle general questions not covered by Q&A pairs."""
        return {
            "response": f"I'm here to help with your shopping needs! I can assist you with finding products, answering questions about our catalog, or helping you make decisions. What specific shopping question can I help you with?",
            "intent": "general_chat",
            "confidence": 0.7,
            "metadata": {"tool_used": "general_conversation", "type": "general_question"}
        }
    
    def _handle_default_response(self, message: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle messages that don't match specific patterns."""
        return {
            "response": f"I'm here to help with your shopping needs! You can ask me to find products, upload images for recommendations, or ask questions about our catalog. What would you like to explore today?",
            "intent": "general_chat",
            "confidence": 0.6,
            "metadata": {"tool_used": "general_conversation", "type": "default"}
        }
