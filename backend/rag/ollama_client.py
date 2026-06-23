import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

class CardioVisionRAG:
    def __init__(self):
        # Initialize Vector DB
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chromadb'))
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="cardiovision_kb")
        
        # Load embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_explanation(self, predictions: dict, key_features: dict, shap_importance: dict, lead_importance: dict = {}):
        """
        Retrieves context and asks Phi3 to generate an explanation.
        predictions: {"Normal Sinus Rhythm": 0.85, ...}
        key_features: {"Heart_Rate": 75, ...}
        shap_importance: {"Normal Sinus Rhythm": {"Top Feature 1": 0.23, ...}}
        lead_importance: {"II": 30.5, "V1": 12.0, ...}
        """
        
        # 1. Build a search query based on predicted conditions
        predicted_classes = list(predictions.keys())
        if not predicted_classes:
            query_text = "Normal Sinus Rhythm and healthy ECG characteristics."
        else:
            query_text = " ".join(predicted_classes) + " ECG abnormalities, causes, and characteristics."
            
        # 2. Retrieve top 3 relevant medical documents from Chroma
        query_embedding = self.embedder.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        context_text = "\n\n".join(results['documents'][0])
        
        # 3. Construct the prompt
        prompt = self._build_prompt(predictions, key_features, shap_importance, context_text, lead_importance)
        
        # 4. Call Ollama Phi3
        print(f"Sending prompt to Ollama Phi3... (Prompt length: {len(prompt)} chars)")
        
        try:
            response = ollama.chat(model='phi3', messages=[
                {
                    'role': 'system',
                    'content': 'You are CardioVision AI Assistant, an expert ECG interpretation and explainable AI assistant. Your task is to explain machine learning predictions generated from ECG features in simple, medically meaningful language. Use the provided context.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error connecting to Ollama Phi3. Please ensure the local Ollama server is running. Detailed error: {str(e)}"
            
    def _build_prompt(self, predictions, key_features, shap_importance, retrieved_docs, lead_importance):
        # Format Predictions
        pred_str = ""
        for cls, prob in predictions.items():
            pred_str += f"- {cls}: {prob*100:.1f}% probability\n"
        if not pred_str:
            pred_str = "- No abnormalities detected (Normal ECG)\n"
            
        # Format Features
        feat_str = ""
        for feat, val in key_features.items():
            feat_str += f"- {feat}: {val:.2f}\n"
            
        # Format SHAP
        shap_str = ""
        for cls, top_feats in shap_importance.items():
            shap_str += f"\nFor {cls}:\n"
            for feat, imp in top_feats.items():
                shap_str += f"  - {feat} (Importance: {imp:.3f})\n"
                
        # Format Lead Importance
        lead_str = ""
        for lead, imp in lead_importance.items():
            if imp > 0:
                lead_str += f"  - {lead}: {imp}%\n"
                
        # Inject the rules (from user request)
        prompt = f"""
=================================
PATIENT DATA & PREDICTIONS
=================================
Predicted Conditions:
{pred_str}

Key ECG Features Measured:
{feat_str}

Machine Learning Feature Importance (SHAP):
{shap_str}

Lead Importance Breakdown:
{lead_str}

=================================
RETRIEVED MEDICAL KNOWLEDGE (RAG)
=================================
{retrieved_docs}

=================================
RULES FOR EXPLANATION
=================================
1. Never claim a diagnosis with absolute certainty. Emphasize that this is an "AI-generated prediction" requiring physician review.
2. Use the provided prediction probabilities and explain them based on the guidelines.
3. Discuss why the model made its decision by explicitly naming the SHAP Top Features and Lead Importance.
4. If multiple conditions have probabilities > 20%, provide a "Condition Comparison" explaining why one condition won over the other based on the features.
5. Structure your response EXACTLY as follows using Markdown headers:
   - Prediction Summary
   - Vital Metrics Analysis
   - Lead & Waveform Findings
   - Why The AI Predicted This (Feature Importance)
   - Condition Comparison (if applicable)
   - Recommended Follow-Up
   - Disclaimer
6. Keep language professional yet understandable. Be highly detailed.

Provide the final clinical report now.
"""
        return prompt
