"""
General Conversation Tool for PalonaAI
Handles general questions, greetings, and conversational interactions.
"""

import logging
from typing import Dict, Any, List
import re
from src.config.dynamic_config import get_config

logger = logging.getLogger(__name__)

class GeneralConversationTool:
    """Tool for handling general conversation with PalonaAI."""
    
    def __init__(self):
        """Initialize the general conversation tool."""
        self.config = get_config()
        self.agent_name = self.config.agent_name
        msgs = self.config.get_response_messages()
        # Build lightweight QA map from config messages to avoid hardcoded text
        self.qa_map = {
            "what's your name": f"I'm {self.agent_name}, your AI shopping assistant!",
            "who are you": f"I'm {self.agent_name}, your shopping companion.",
            "what can you do": (
                "I can help you with product search, image analysis, recommendations, "
                "product information, and comparisons."
            ),
            "how do you work": "I understand your request and search the catalog for the best matches.",
            "can you compare products": "Yes, I can compare items and explain key differences.",
            "do you remember our conversation": "Yes, I keep context to improve recommendations."
        }
    
    def handle_conversation(self, message: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle general conversation and return appropriate response."""
        try:
            message_lower = message.lower().strip()
            # Capture: "my name is <name>" and acknowledge + return metadata to persist
            m = re.search(r"\bmy\s+name\s+is\s+([a-zA-Z][\w\s'-]{1,40})$", message_lower)
            if m:
                name = m.group(1).strip().title()
                return {
                    "response": f"Nice to meet you, {name}! I’ll remember your name while we shop.",
                    "intent": "general_chat",
                    "confidence": 0.95,
                    "metadata": {"tool_used": "general_conversation", "set_user_name": name}
                }
            
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
            
            # Check for predefined Q&A (config-driven map)
            for question, answer in self.qa_map.items():
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
        # If user asks about earlier context, summarize last interactions
        msg = (message or "").lower()
        if "what did i ask" in msg or "what did we talk" in msg or "earlier" in msg:
            history = []
            if conversation_context and isinstance(conversation_context, dict):
                history = (conversation_context.get("conversation_history") or [])
            if history:
                last = history[-5:]
                bullets = []
                for it in last:
                    intent = it.get("intent", "unknown")
                    txt = (it.get("user_input") or "").strip().replace("\n"," ")
                    if len(txt) > 70:
                        txt = txt[:70] + "…"
                    bullets.append(f"• {intent}: {txt}")
                summary = "Here’s what you asked recently:\n\n" + "\n".join(bullets)
            else:
                summary = "I don’t have earlier messages yet in this session."
            return {
                "response": summary,
                "intent": "general_chat",
                "confidence": 0.9,
                "metadata": {"tool_used": "general_conversation", "type": "context_summary"}
            }

        return {
            "response": "I’m here to help with your shopping. Ask me to find items, set a budget or color, or upload a photo for similar products.",
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
