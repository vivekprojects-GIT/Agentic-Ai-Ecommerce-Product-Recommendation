"""
Unified Gradio UI for Commerce Agent with proper conversational memory.
Combines the best features from both UI files and fixes memory issues.
"""

import gradio as gr
import requests
import json
import base64
import io
import os
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from src.config import get_config, setup_logging

# Setup dynamic logging
setup_logging()

# Get dynamic configuration
config = get_config()
API_BASE_URL = f"http://{config.api_host}:{config.api_port}"
API_ENDPOINT = f"{API_BASE_URL}/api/v1/simple-rag/ask"

class ConversationMemory:
    """Enhanced conversation memory with state tracking."""
    
    def __init__(self):
        self.conversation_history = []
        self.current_state = "idle"
        self.session_metadata = {
            "session_id": None,
            "start_time": datetime.now().isoformat(),
            "total_interactions": 0,
            "user_preferences": {},
            "product_interests": []
        }
    
    def add_interaction(self, user_input: str, agent_response: str, image_data: Optional[str] = None, intent: str = "unknown"):
        """Add interaction to memory with state tracking."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "image_data": image_data,
            "intent": intent,
            "state": self.current_state
        }
        
        self.conversation_history.append(interaction)
        self.session_metadata["total_interactions"] += 1
        
        # Update state based on intent
        self._update_state(intent)
    
    def _update_state(self, intent: str):
        """Update conversation state based on intent."""
        state_transitions = {
            "idle": {"product_search": "searching", "image_search": "analyzing", "general_chat": "chatting"},
            "searching": {"product_search": "searching", "general_chat": "chatting", "image_search": "analyzing"},
            "analyzing": {"image_search": "analyzing", "product_search": "searching", "general_chat": "chatting"},
            "chatting": {"general_chat": "chatting", "product_search": "searching", "image_search": "analyzing"}
        }
        
        if intent in state_transitions.get(self.current_state, {}):
            self.current_state = state_transitions[self.current_state][intent]
    
    def get_context_for_api(self) -> Dict[str, Any]:
        """Get conversation context for API requests."""
        # Get configurable conversation history limit
        history_limit = getattr(config, 'conversation_context_limit', 5)
        return {
            "conversation_history": self.conversation_history[-history_limit:],
            "current_state": self.current_state,
            "session_metadata": self.session_metadata
        }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.conversation_history = []
        self.current_state = "idle"
        self.session_metadata = {
            "session_id": None,
            "start_time": datetime.now().isoformat(),
            "total_interactions": 0,
            "user_preferences": {},
            "product_interests": []
        }

class UnifiedCommerceUI:
    """Unified Gradio interface for the Commerce Agent with proper memory."""
    
    def __init__(self):
        self.conversation_memory = ConversationMemory()
        self.api_available = self._check_api_health()
        self.config = config
    
    def _check_api_health(self) -> bool:
        """Check if the API is running and accessible."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _process_image(self, image) -> Optional[str]:
        """Convert PIL image to base64 string."""
        if image is None:
            return None
        
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 1024px on longest side)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def _format_response_with_products(self, agent_response: str, products: List[Dict[str, Any]]) -> str:
        """Format response with product cards using markdown."""
        if not products:
            return agent_response
        
        # Create product cards using markdown format
        product_cards = []
        for i, product in enumerate(products[:5], 1):  # Limit to 5 products for display
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            brand = product.get('attributes', {}).get('brand', 'Unknown Brand')
            color = product.get('attributes', {}).get('color_family', 'Unknown Color')
            description = product.get('description', '')
            url = product.get('url', '')
            
            # Truncate description if too long
            if len(description) > 100:
                description = description[:100] + '...'
            
            display_name = f"[{name}]({url})" if url else name
            product_card = f"""
**{i}. {display_name}** - ${price:.2f}
- **Brand:** {brand}
- **Color:** {color}
- **Description:** {description}

---
"""
            product_cards.append(product_card)
        
        return agent_response + f"""

🛍️ **Recommended Products:**

{''.join(product_cards)}
"""
    
    def chat_with_agent(self, message: str, image: Optional[Image.Image], history: List) -> Tuple[str, List, Optional[Image.Image]]:
        """Process chat message with proper conversation memory."""
        if not self.api_available:
            return "❌ **API Error**: The commerce agent API is not available. Please make sure the server is running on port 8080.", history, None
        
        if not message.strip() and image is None:
            return "Please enter a message or upload an image.", history, None
        
        try:
            # Process image if provided
            image_base64 = self._process_image(image)
            
            # Get conversation context from memory
            conversation_context = self.conversation_memory.get_context_for_api()
            
            # Prepare request data with conversation context
            request_data = {
                "text_input": message.strip() if message else None,
                "image_base64": image_base64,
                "conversation_context": conversation_context  # This is the key fix!
            }
            
            # Make API request
            response = requests.post(API_ENDPOINT, json=request_data, timeout=30)
            
            if response.status_code != 200:
                return f"❌ **API Error**: {response.status_code} - {response.text}", history, None
            
            # Parse response
            result = response.json()
            agent_response = result.get('response', 'No response received')
            products = result.get('products', [])
            intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # Format response with products
            formatted_response = self._format_response_with_products(agent_response, products)
            
            # Add interaction to memory
            self.conversation_memory.add_interaction(
                user_input=message,
                agent_response=agent_response,
                image_data=image_base64,
                intent=intent
            )
            
            # Update conversation history for UI
            history.append([message, formatted_response])
            
            return "", history, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ **Connection Error**: Unable to connect to the API. Please check if the server is running.\n\nError: {str(e)}"
            history.append([message, error_msg])
            return "", history, None
        except Exception as e:
            error_msg = f"❌ **Unexpected Error**: {str(e)}"
            history.append([message, error_msg])
            return "", history, None
    
    def clear_chat(self) -> Tuple[str, List]:
        """Clear the conversation history and memory."""
        self.conversation_memory.clear_memory()
        return "", []
    
    def create_interface(self):
        """Create the unified Gradio interface with proper memory management."""
        ui_config = self.config.get_ui_config()
        
        # Custom CSS for beautiful, clean styling
        custom_css = f"""
        .gradio-container {{
            max-width: {ui_config['max_width']}px !important;
            margin: auto !important;
        }}
        .chat-message {{
            padding: 16px !important;
            margin: 12px 0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }}
        .user-message {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }}
        .assistant-message {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
            color: white !important;
        }}
        .main-container {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
            padding: 20px !important;
            border-radius: 16px !important;
        }}
        """
        
        # Get theme safely
        try:
            theme = getattr(gr.themes, ui_config['theme'].title())()
        except:
            theme = gr.themes.Soft()
        
        with gr.Blocks(css=custom_css, title=f"🛍️ {ui_config['title']}", theme=theme) as interface:
            
            # Dynamic header
            gr.Markdown(f"""
            # 🛍️ {ui_config['title']}
            
            *{ui_config['description']}*
            
            **Features:**
            - 💬 **Conversational Memory**: Remembers our conversation
            - 🖼️ **Image Analysis**: Upload images for product recommendations  
            - 🔍 **Smart Search**: Find products with natural language
            - 🎯 **Context-Aware**: Understands your preferences over time
            """)
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=1):
                    # Chat history
                    chatbot = gr.Chatbot(
                        value=[],
                        height=ui_config['chat_height'],
                        show_label=False,
                        elem_classes=["chat-message"],
                        container=True
                    )
                    
                    # Input area with inline image uploader
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me about products or upload an image below...",
                            show_label=False,
                            lines=1,
                            max_lines=3,
                            scale=5
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    # Image upload (full-width, clearly visible)
                    image_input = gr.Image(
                        type="pil",
                        sources=["upload", "clipboard", "webcam"],
                        label="Upload Image (Optional)",
                        height=240,
                        interactive=True
                    )
                    
                    # Action buttons
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        memory_info_btn = gr.Button("🧠 Memory Info", variant="secondary")
            
            # Memory info panel (initially hidden)
            with gr.Row(visible=False) as memory_panel:
                with gr.Column():
                    memory_info = gr.Markdown("Memory information will appear here...")
            
            # Event handlers
            def handle_send(message, image, history):
                return self.chat_with_agent(message, image, history)
            
            def handle_clear():
                return self.clear_chat()
            
            def show_memory_info():
                ctx = self.conversation_memory.get_context_for_api()
                state = (ctx.get('current_state') or 'idle').lower()
                meta = ctx.get('session_metadata', {})
                history = ctx.get('conversation_history', [])

                # Build compact state line with highlighted current state
                state_order = ["idle", "searching", "analyzing", "chatting"]
                def fmt(s):
                    return f"<span style='padding:2px 8px;border-radius:10px;background:#2a3942;color:#e9edef'>{s}</span>" if s != state \
                        else f"<span style='padding:2px 8px;border-radius:10px;background:#00a884;color:#111b21;font-weight:700'>{s}</span>"
                state_graph = " → ".join(fmt(s) for s in state_order)

                # Build recent interactions (most recent first, max 5)
                lines = []
                recent = history[-5:][::-1]
                for it in recent:
                    intent = it.get('intent', 'unknown')
                    text = (it.get('user_input') or '').strip().replace('\n', ' ')
                    if len(text) > 70:
                        text = text[:70] + '…'
                    badge = f"<span style='padding:2px 6px;border:1px solid #2a3942;border-radius:6px;color:#e9edef'>{intent}</span>"
                    lines.append(f"<div style='margin:6px 0'><span style='opacity:.85'>{badge}</span> <span style='color:#cfd8dc'>{text}</span></div>")

                recent_html = "".join(lines) if lines else "<div style='opacity:.7'>No prior interactions.</div>"

                # Compose panel
                html = f"""
<div style='line-height:1.6'>
  <div style='margin-bottom:8px;font-weight:700;font-size:16px'>Conversation Memory</div>
  <div style='margin-bottom:6px'>State: {state_graph}</div>
  <div style='margin-bottom:12px;color:#cfd8dc'>
    <span style='margin-right:12px'>Checkpoints: <b>{meta.get('total_interactions',0)}</b></span>
    <span>Started: {meta.get('start_time','')}</span>
  </div>
  <div style='margin:6px 0;font-weight:600'>Recent</div>
  {recent_html}
</div>
                """
                return html
            
            # Connect event handlers
            send_btn.click(
                handle_send,
                inputs=[msg_input, image_input, chatbot],
                outputs=[msg_input, chatbot, image_input]
            )
            
            msg_input.submit(
                handle_send,
                inputs=[msg_input, image_input, chatbot],
                outputs=[msg_input, chatbot, image_input]
            )
            
            clear_btn.click(
                handle_clear,
                outputs=[msg_input, chatbot]
            )
            
            memory_info_btn.click(
                show_memory_info,
                outputs=[memory_info]
            ).then(
                lambda: gr.Row(visible=True),
                outputs=[memory_panel]
            )
            
            # Footer
            gr.Markdown(f"""
            ---
            **Powered by**: {self.config.llm_model} | **Memory**: ✅ Enabled | **API**: {'🟢 Connected' if self.api_available else '🔴 Disconnected'}
            """)
        
        return interface

def main():
    """Main function to launch the unified UI."""
    print("🚀 Starting Unified Commerce Agent UI...")
    print("✅ Features: Conversational Memory, Image Analysis, Context-Aware Responses")
    
    # Create unified UI
    ui = UnifiedCommerceUI()
    interface = ui.create_interface()
    
    # Launch interface
    print(f"🌐 Launching interface on {config.api_host}:7860...")
    interface.launch(
        server_name=config.api_host,
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
