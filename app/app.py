import gradio as gr
import requests

# The URL pointing directly to your local FastAPI backend endpoint
API_URL = "http://127.0.0.1:8000/api/v1/support"

def process_customer_ticket(user_message):
    if not user_message.strip():
        # Default neutral alert if empty
        neutral_html = """
        <div style="background-color: #f3f4f6; border-left: 6px solid #9ca3af; padding: 15px; border-radius: 4px; color: #1f2937; font-weight: bold;">
            ⚪ No Active Request
        </div>
        """
        return "⚠️ Please enter a message.", "N/A", "0.0%", "N/A", neutral_html
        
    try:
        # 1. Send the user's text to your FastAPI backend
        response = requests.post(API_URL, json={"text": user_message})
        data = response.json()
        
        # If the API returned an internal processing error
        if "error" in data:
            error_html = """
            <div style="background-color: #fde8e8; border-left: 6px solid #e11d48; padding: 15px; border-radius: 4px; color: #9f1239; font-weight: bold;">
                🔴 Backend Error
            </div>
            """
            return "Error", "N/A", "0.0%", data["details"], error_html
            
        # 2. Extract the variables sent back from your FastAPI engine
        intent = data["intent"]
        confidence_pct = f"{data['confidence'] * 100:.2f}%"
        reply = data["suggested_reply"]
        hitl_flag = data["human_in_the_loop_required"]
        
        # 3. Dynamically generate HTML and CSS based on the HITL flag status
        if hitl_flag:
            # ORANGE ALERT STATE
            hitl_html = """
            <div style="background-color: #fff7ed; border: 2px solid #ea580c; border-left: 8px solid #ea580c; padding: 15px; border-radius: 6px; color: #c2410c; font-weight: bold; font-size: 16px;">
                ⚠️ REQUIRES HUMAN REVIEW (Low Confidence Calibration)
            </div>
            """
        else:
            # GREEN SAFE STATE
            hitl_html = """
            <div style="background-color: #f0fdf4; border: 2px solid #16a34a; border-left: 8px solid #16a34a; padding: 15px; border-radius: 6px; color: #15803d; font-weight: bold; font-size: 16px;">
                ✅ AUTOMATICALLY ROUTED (Safe to Send)
            </div>
            """
            
        return user_message, intent, confidence_pct, reply, hitl_html
        
    except requests.exceptions.ConnectionError:
        offline_html = """
        <div style="background-color: #fef2f2; border-left: 6px solid #dc2626; padding: 15px; border-radius: 4px; color: #991b1b; font-weight: bold;">
            🔴 Backend Offline
        </div>
        """
        return "Error", "N/A", "0.0%", "Could not connect to FastAPI backend. Is uvicorn running?", offline_html

# --- Building the Dual-Pane Dashboard Interface Layout ---
with gr.Blocks(title="AI Support Routing Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 E-Commerce Smart Support & Escalation Hub")
    gr.Markdown("This system combines a fine-tuned DistilBERT intent classifier with deterministic routing and automated fallback orchestration.")
    
    with gr.Row():
        # LEFT PANE: Customer Simulator
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Customer Simulator View")
            user_input = gr.Textbox(
                label="Type your support request here:", 
                placeholder="e.g., I want to return my broken shoes for a full refund...",
                lines=4
            )
            submit_btn = gr.Button("Send to AI Routing Switchboard", variant="primary")
            
        # RIGHT PANE: Enterprise Agent Dashboard
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Agent Control Room (Behind the Scenes)")
            
            # CHANGED: Using gr.HTML instead of gr.Textbox for dynamic, colored styling injection
            with gr.Row():
                output_hitl = gr.HTML(
                    value="""
                    <div style="background-color: #f3f4f6; border-left: 6px solid #9ca3af; padding: 15px; border-radius: 4px; color: #1f2937; font-weight: bold;">
                        ⚪ Awaiting Input Sequence...
                    </div>
                    """
                )
                
            with gr.Row():
                output_intent = gr.Textbox(label="Detected Intent Category", interactive=False)
                output_confidence = gr.Textbox(label="Classification Confidence Score", interactive=False)
                
            output_query = gr.Textbox(label="Processed Raw Text", interactive=False)
            output_reply = gr.Textbox(label="Generated/Suggested Customer Response", lines=4, interactive=False)

    # Connect the button click event to our python function
    submit_btn.click(
        fn=process_customer_ticket,
        inputs=[user_input],
        outputs=[output_query, output_intent, output_confidence, output_reply, output_hitl]
    )

# Launch the frontend app locally
if __name__ == "__main__":
    demo.launch()