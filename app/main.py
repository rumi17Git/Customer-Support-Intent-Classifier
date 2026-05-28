from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory

# Ensure language detection is reproducible across boots
DetectorFactory.seed = 42

app = FastAPI(title="E-commerce Intent, Translation & Escalation Engine")

# 1. Load Intent Classifier on CPU
print("Loading Intent Classifier...")
classifier = pipeline(
    "text-classification", 
    model="RummanJ17/distilbert-ecommerce-intents",
    tokenizer="RummanJ17/distilbert-ecommerce-intents",
    device=-1
)

# 2. Native Initialization of Translation Engines (Bypassing Pipeline entirely)
print("Loading Native Marian Translation Engines (FR <-> EN)...")

# French to English Engine
fr_to_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name)

# English to French Engine
en_to_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name)

print("All engines successfully initialized locally!")

STATIC_RESPONSES = {
    "WHERE_IS_MY_ORDER": "Hello! You can track your package in real-time by visiting the 'My Orders' section on our website.",
    "REFUND_REQUEST": "Hi there. To start a return, please package your item securely and print your free return label from our portal.",
    "PRODUCT_FEEDBACK": "Thank you for sharing your experience. We have logged this with our Quality Assurance team to improve our next batch.",
    "CANCEL_ORDER": "If your order has not shipped yet, you can cancel it instantly by clicking 'Cancel Order' in your confirmation email."
}

class UserQuery(BaseModel):
    text: str

def call_mock_genai_llm(user_text):
    return f"[GenAI Draft]: We received your request: '{user_text}'. A support representative is reviewing your account to provide an exact resolution."

# Helper translation function to handle native tokenization tensors
def native_translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

@app.post("/api/v1/support")
def process_support_ticket(query: UserQuery):
    try:
        raw_text = query.text
        
        # --- LANGUAGE DETECTION LAYER ---
        try:
            detected_lang = detect(raw_text)
        except:
            detected_lang = "en"
            
        # If input is French, translate it to English natively
        if detected_lang == "fr":
            processing_text = native_translate(raw_text, fr_to_en_model, fr_to_en_tokenizer)
            print(f"🇫🇷 French detected. Translated to EN: '{processing_text}'")
        else:
            processing_text = raw_text

        # --- CLASSIFICATION LAYER ---
        model_outputs = classifier(processing_text)[0]
        detected_intent = model_outputs.get('label')
        confidence_score = model_outputs.get('score', 0.0)
        
        # --- BUSINESS LOGIC & ORCHESTRATION LAYER ---
        if confidence_score >= 0.80:
            final_reply = STATIC_RESPONSES.get(detected_intent, "How can we assist you today?")
            human_in_the_loop = False
        else:
            final_reply = call_mock_genai_llm(processing_text)
            human_in_the_loop = True
            
        # --- REVERSE TRANSLATION LAYER ---
        # If original text was French, send the answer back in French natively
        if detected_lang == "fr":
            final_reply = native_translate(final_reply, en_to_fr_model, en_to_fr_tokenizer)

        return {
            "customer_query": raw_text,
            "detected_language": detected_lang,
            "intent": detected_intent,
            "confidence": round(float(confidence_score), 4),
            "suggested_reply": final_reply,
            "human_in_the_loop_required": human_in_the_loop
        }
        
    except Exception as e:
        print(f"❌ CRITICAL BACKEND ERROR: {str(e)}")
        return {"error": "Internal processing issue", "details": str(e)}