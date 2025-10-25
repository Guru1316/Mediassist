import streamlit as st
import joblib
import scipy.sparse
import numpy as np
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Constants ---
LOW_CONFIDENCE_THRESHOLD = 0.1

# -------------------
# Page Configuration
# -------------------
st.set_page_config(
    page_title="MediAssist: AI Symptom Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Load Artifacts (Cached)
# -------------------
@st.cache_resource
def load_artifacts():
    """Loads TF-IDF model and disease list."""
    try:
        vectorizer = joblib.load("vectorizer.pkl")
        diseases_list = joblib.load("diseases_list.pkl")
        X = scipy.sparse.load_npz("disease_tfidf.npz")
        if not diseases_list or vectorizer is None or X is None:
             raise FileNotFoundError("One or more core model files are empty or invalid.")
        print("TF-IDF Artifacts (vectorizer, diseases_list, matrix) loaded successfully.")
        return vectorizer, diseases_list, X
    except FileNotFoundError as e:
        st.error(f"🚨 Critical Error: TF-IDF Model files not found or failed to load! ({e})")
        st.error("Ensure 'vectorizer.pkl', 'diseases_list.pkl', 'disease_tfidf.npz' are present.")
        return None, None, None
    except Exception as e:
        st.error(f"🚨 Error loading TF-IDF artifacts: {e}")
        return None, None, None

# Load artifacts
vectorizer, diseases_list, X = load_artifacts()

# -------------------
# Load Groq API Key & Initialize Client (Cached)
# -------------------
@st.cache_resource
def initialize_groq():
    """Initializes the Groq client using secrets or environment variables."""
    groq_api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
            print("Groq API Key found in Streamlit secrets.")
        else:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if groq_api_key:
                print("Groq API Key found in environment variable.")

        if not groq_api_key:
            st.warning("⚠ Groq API Key not found. AI features will be limited.")
            st.warning("Create .streamlit/secrets.toml or set Secrets in Streamlit Cloud for full functionality.")
            return None

        client = Groq(api_key=groq_api_key)
        print("Groq client initialized successfully.")
        return client
    except st.errors.StreamlitAPIException as e:
         st.error(f"🚨 Error accessing Streamlit secrets: {e}. Ensure '.streamlit/secrets.toml' format.")
         return None
    except Exception as e:
        st.error(f"🚨 Failed to initialize Groq client: {e}")
        return None

groq_client = initialize_groq()

# -------------------
# Helper Functions
# -------------------
def clean_text(text):
    """Cleans input text."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ,]', ' ', text)
    text = re.sub(r'\s+|,+', ' ', text).strip()
    return text

def get_symptom_suggestions_groq(original_symptoms):
    """Uses Groq LLM to get synonyms or rephrased symptoms for TF-IDF enhancement."""
    if not groq_client: return None
    try:
        prompt = f"""
        List common synonyms or related medical terms for these symptoms: "{original_symptoms}".
        Provide ONLY a comma-separated list. Example: Input 'sleeplessness', Output 'insomnia, difficulty sleeping'.
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3, max_tokens=60
        )
        suggestions = chat_completion.choices[0].message.content.strip()
        if suggestions and len(suggestions.split(',')) > 0:
            print(f"Groq suggestions for '{original_symptoms}': {suggestions}")
            return suggestions
        else:
             print(f"Groq returned empty/invalid suggestions for '{original_symptoms}'.")
             return None
    except Exception as e:
        st.error(f"Error calling Groq for symptom suggestions: {e}")
        print(f"Error calling Groq for symptom suggestions: {e}")
        return None

def get_treatment_info_groq(disease_name):
    """Uses Groq LLM to get general treatment info for a disease identified by TF-IDF."""
    if not groq_client: return "AI information features disabled (Groq key missing)."
    try:
        prompt = f"""
        Provide brief, general information and common treatment approaches for "{disease_name}".
        Concise (2-4 sentences), easy for non-experts. Start directly with info.
        End with the exact sentence: "This information is AI-generated and not a substitute for professional medical advice."
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.5, max_tokens=180
        )
        info = chat_completion.choices[0].message.content.strip()
        print(f"Groq treatment info for '{disease_name}': Fetched.")
        disclaimer = "This information is AI-generated and not a substitute for professional medical advice."
        if not info.endswith(disclaimer): info += f" {disclaimer}"
        return info
    except Exception as e:
        st.error(f"Error calling Groq for treatment info: {e}")
        print(f"Error calling Groq for treatment info: {e}")
        return "Could not fetch AI-generated information due to an error."

def get_prediction_and_info_directly_from_groq(symptoms):
    """Uses Groq LLM to PREDICT disease and get info when TF-IDF fails."""
    if not groq_client: return "AI prediction features disabled (Groq key missing)."
    try:
        prompt = f"""
        Based on the symptoms "{symptoms}", what is the most likely potential medical condition?
        Then, provide brief, general information and common treatment approaches for that condition.
        Keep the response concise (3-5 sentences total) and easy for a non-expert.
        Format: "Potential Condition: [Disease Name]. [Brief info and treatments...]"
        Crucially, end the entire response with the exact sentence: "This prediction and information are purely AI-generated based on symptoms provided and ARE NOT a medical diagnosis. Consult a doctor."
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.6,
            max_tokens=250
        )
        result = chat_completion.choices[0].message.content.strip()
        print(f"Groq direct prediction for '{symptoms}': Fetched.")
        disclaimer = "This prediction and information are purely AI-generated based on symptoms provided and ARE NOT a medical diagnosis. Consult a doctor."
        if not result.endswith(disclaimer):
            result += f" {disclaimer}"
        return result
    except Exception as e:
        st.error(f"Error calling Groq for direct prediction: {e}")
        print(f"Error calling Groq for direct prediction: {e}")
        return "Could not perform AI-based prediction due to an error."

def predict_topk_tfidf(symptom_input, topk=3):
    """Performs the core TF-IDF prediction."""
    if vectorizer is None or X is None or not diseases_list:
        st.error("TF-IDF Model artifacts not loaded.")
        return [], None
    input_clean = clean_text(symptom_input)
    if not input_clean: return [], None
    try:
        q = vectorizer.transform([input_clean])
        sims = cosine_similarity(q, X).flatten()
        num_results = min(topk, X.shape[0])
        top_idxs = np.argsort(sims)[-num_results:][::-1]
        results = []
        top_score = 0.0
        if len(top_idxs) > 0: top_score = float(sims[top_idxs[0]])
        for idx in top_idxs:
            if 0 <= idx < len(diseases_list):
                 disease = diseases_list[idx]
                 score = float(sims[idx])
                 if score > 0.01: results.append((disease, score))
            else: print(f"Warning: Index {idx} out of bounds.")
        return results, top_score
    except Exception as e:
        st.error(f"Error during TF-IDF prediction: {e}")
        return [], None

def get_final_predictions_enhanced(symptom_input, topk=3):
    """Gets TF-IDF predictions, attempts Groq rephrasing if low confidence."""
    predictions, top_score = predict_topk_tfidf(symptom_input, topk)
    should_enhance = groq_client and (not predictions or (top_score is not None and top_score < LOW_CONFIDENCE_THRESHOLD))

    if should_enhance:
        st.info("Initial match confidence low. Attempting AI symptom enhancement...")
        with st.spinner("Asking AI for related terms..."):
            suggestions = get_symptom_suggestions_groq(symptom_input)
        if suggestions:
            combined_symptoms = clean_text(symptom_input + " " + suggestions)
            st.info(f"Retrying analysis with enhanced context...")
            enhanced_predictions, enhanced_top_score = predict_topk_tfidf(combined_symptoms, topk)
            if enhanced_predictions and (not predictions or (enhanced_top_score is not None and top_score is not None and enhanced_top_score > top_score)):
                 print("Using enhanced predictions.")
                 st.success("AI enhancement improved potential matches.")
                 return enhanced_predictions
            else:
                 print("Enhancement didn't improve score. Using original.")
                 if predictions: st.info("AI enhancement did not significantly change results.")
                 return predictions
        else:
            print("Groq suggestion failed. Using original TF-IDF.")
            if predictions: st.info("Could not enhance symptoms using AI.")
            return predictions
    else:
        print("Using original TF-IDF predictions.")
        return predictions

# -------------------
# Sidebar Navigation
# -------------------
with st.sidebar:
    st.title("MediAssist 🩺")
    st.image("https://placehold.co/300x150/e8f0fe/1967d2?text=MediAssist+V3&font=inter", use_container_width=True)
    
    # Navigation
    page = st.radio("Navigate to:", [
        "Symptom Analyzer", 
        "Diseases Information", 
        "Model Performance", 
        "About"
    ])
    
    st.markdown("---")
    st.warning("🚨 *Disclaimer:* Informational tool only. *Not* a diagnostic substitute. AI predictions/info require verification by a medical professional. Always consult a doctor.")

# -------------------
# Symptom Analyzer Page
# -------------------
if page == "Symptom Analyzer":
    st.title("🏥 Symptom Analyzer")
    st.markdown("Enter symptoms for potential condition insights. Uses TF-IDF similarity + AI enhancement & info via Groq.")

    col1, col2 = st.columns([3, 1])
    with col1:
        symptoms_input = st.text_area(
            "*Enter symptoms here:* (e.g., 'high fever, headache, body aches')",
            height=120, placeholder="Describe how you are feeling..."
        )
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("Analyze Symptoms", type="primary", use_container_width=True, disabled=(vectorizer is None))

    st.markdown("---")

    if predict_button:
        if vectorizer and X is not None and diseases_list:
            cleaned_input = symptoms_input.strip()
            if not cleaned_input:
                st.error("⚠ Please enter symptoms before analyzing.")
            else:
                with st.spinner("🧠 Analyzing symptoms with AI..."):
                    final_tfidf_predictions = get_final_predictions_enhanced(cleaned_input, topk=3)

                    if final_tfidf_predictions:
                        st.subheader("📊 Top Matches (from Symptom Similarity Model):")
                        st.caption("Similarity score reflects how closely input matches dataset symptoms.")

                        for i, (disease, score) in enumerate(final_tfidf_predictions, 1):
                            disease_display = ' '.join(word.capitalize() for word in disease.split())
                            with st.expander(f"{i}. {disease_display}** (Similarity Score: {score:.2f})", expanded=(i==1)):
                                st.progress(int(score * 100), text=f"Match Confidence: {score*100:.0f}%")

                                with st.spinner(f"Fetching AI info for {disease_display}..."):
                                    info_from_groq = get_treatment_info_groq(disease)

                                st.markdown(f"*General Information / Common Approach:* [AI Generated Info]")
                                if info_from_groq:
                                    st.info(f"{info_from_groq}")
                                    st.warning("⚠ Verify AI-generated information with a medical professional.")
                                else:
                                    st.error("Could not fetch AI-generated information.")

                                st.markdown("---")
                                st.caption(f"Note: Confidence reflects symptom text similarity, not diagnostic probability.")
                        st.success("✅ Analysis complete (using similarity model). Consult a doctor.")

                    else:
                        st.warning("The similarity model found no strong matches for your symptoms.")
                        if groq_client:
                             st.info("Attempting direct prediction using general AI (use results with extreme caution)...")
                             with st.spinner("Asking AI for direct prediction..."):
                                  direct_ai_result = get_prediction_and_info_directly_from_groq(cleaned_input)

                             st.subheader("🤖 AI Direct Suggestion (Fallback):")
                             if direct_ai_result:
                                  st.warning(f"{direct_ai_result}")
                             else:
                                  st.error("Could not get a direct prediction from the AI.")
                        else:
                             st.error("No matches found, and AI fallback features are disabled (Groq key missing).")
                        st.error("🛑 Please consult a doctor for any health concerns.")
        else:
            st.error("Cannot perform analysis because core model files failed to load.")

# -------------------
# Diseases Information Page
# -------------------
elif page == "Diseases Information":
    st.title("📚 Diseases Information Hub")
    st.markdown("Browse information about various medical conditions in our database.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if diseases_list:
            selected_disease = st.selectbox(
                "Select a disease to learn more:",
                options=sorted(diseases_list),
                index=0 if diseases_list else None,
                placeholder="Choose a disease..."
            )
        else:
            st.error("Disease database not loaded.")
            selected_disease = None
    
    with col2:
        st.write("")
        st.write("")
        get_info_btn = st.button("Get Disease Information", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    if get_info_btn and selected_disease:
        with st.spinner(f"Fetching information about {selected_disease}..."):
            disease_display = ' '.join(word.capitalize() for word in selected_disease.split())
            st.subheader(f"🩺 {disease_display}")
            
            # Get information from Groq
            info_from_groq = get_treatment_info_groq(selected_disease)
            
            if info_from_groq and "disabled" not in info_from_groq.lower():
                st.info(f"{info_from_groq}")
                
                # Additional structured information
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**📋 Common Symptoms:**")
                    st.caption("(AI-generated common symptom patterns)")
                    with st.spinner("Generating common symptoms..."):
                        try:
                            if groq_client:
                                prompt = f"List 5-7 common symptoms for {selected_disease}. Provide as bullet points only."
                                chat_completion = groq_client.chat.completions.create(
                                    messages=[{"role": "user", "content": prompt}],
                                    model="llama-3.1-8b-instant",
                                    temperature=0.3, max_tokens=100
                                )
                                symptoms_list = chat_completion.choices[0].message.content.strip()
                                st.write(symptoms_list)
                            else:
                                st.write("• Symptom information unavailable (AI features disabled)")
                        except:
                            st.write("• Could not fetch symptom details")
                
                with col_info2:
                    st.markdown("**💊 General Care Tips:**")
                    st.caption("(AI-generated general advice)")
                    with st.spinner("Generating care tips..."):
                        try:
                            if groq_client:
                                prompt = f"Provide 3-5 general care tips for managing {selected_disease}. Brief bullet points."
                                chat_completion = groq_client.chat.completions.create(
                                    messages=[{"role": "user", "content": prompt}],
                                    model="llama-3.1-8b-instant",
                                    temperature=0.3, max_tokens=120
                                )
                                care_tips = chat_completion.choices[0].message.content.strip()
                                st.write(care_tips)
                            else:
                                st.write("• Care tips unavailable (AI features disabled)")
                        except:
                            st.write("• Could not fetch care tips")
                
            else:
                st.error("Could not fetch disease information. AI features may be disabled.")
            
            st.warning("⚠️ This information is AI-generated and should be verified with healthcare professionals.")

# -------------------
# Model Performance Page
# -------------------
elif page == "Model Performance":
    st.title("📈 Model Performance & Analytics")
    st.markdown("Technical insights about the symptom analysis model and its performance.")
    
    tab1, tab2, tab3 = st.tabs(["Model Metrics", "Disease Coverage", "System Information"])
    
    with tab1:
        st.subheader("TF-IDF Model Performance")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(
                label="Diseases in Database",
                value=len(diseases_list) if diseases_list else "N/A",
                delta=None
            )
        
        with col_metric2:
            st.metric(
                label="Feature Dimensions",
                value=f"{X.shape[1] if X is not None else 'N/A'}",
                delta=None
            )
        
        with col_metric3:
            st.metric(
                label="AI Enhancement",
                value="Active" if groq_client else "Inactive",
                delta=None
            )
        
        # Sample performance visualization
        st.subheader("Similarity Score Distribution")
        
        # Generate sample data for demonstration
        np.random.seed(42)
        sample_scores = np.random.beta(2, 5, 1000)  # Simulated similarity scores
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sample_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(LOW_CONFIDENCE_THRESHOLD, color='red', linestyle='--', 
                  label=f'Low Confidence Threshold ({LOW_CONFIDENCE_THRESHOLD})')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predicted Similarity Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.caption("Simulated distribution of TF-IDF cosine similarity scores for demonstration purposes.")
    
    with tab2:
        st.subheader("Disease Database Overview")
        
        if diseases_list:
            # Show sample of diseases
            st.write(f"**Total Diseases:** {len(diseases_list)}")
            
            # Create a sample for display
            sample_size = min(20, len(diseases_list))
            sample_diseases = np.random.choice(diseases_list, sample_size, replace=False)
            
            col_disp1, col_disp2 = st.columns(2)
            
            with col_disp1:
                st.markdown("**Sample Diseases in Database:**")
                for i, disease in enumerate(sample_diseases[:10], 1):
                    st.write(f"{i}. {disease}")
            
            with col_disp2:
                st.markdown("**Database Statistics:**")
                disease_lengths = [len(disease.split()) for disease in diseases_list]
                avg_words = np.mean(disease_lengths)
                st.write(f"• Average disease name length: {avg_words:.1f} words")
                st.write(f"• Shortest name: {min(disease_lengths)} words")
                st.write(f"• Longest name: {max(disease_lengths)} words")
                
                # Word cloud simulation
                st.write("• Most common terms in disease names:")
                all_terms = ' '.join(diseases_list).lower().split()
                from collections import Counter
                common_terms = Counter(all_terms).most_common(5)
                for term, count in common_terms:
                    st.write(f"  - {term}: {count} occurrences")
        
        else:
            st.error("Disease database not loaded.")
    
    with tab3:
        st.subheader("System Configuration")
        
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.markdown("**AI Services Status**")
            status_color = "🟢" if groq_client else "🔴"
            st.write(f"{status_color} Groq AI API: {'Connected' if groq_client else 'Not Available'}")
            
            st.write("🔵 TF-IDF Model:", "Loaded ✅" if vectorizer else "Not Loaded ❌")
            st.write("🔵 Disease Database:", "Loaded ✅" if diseases_list else "Not Loaded ❌")
            st.write("🔵 Similarity Matrix:", "Loaded ✅" if X is not None else "Not Loaded ❌")
        
        with col_sys2:
            st.markdown("**Performance Settings**")
            st.write(f"• Low Confidence Threshold: {LOW_CONFIDENCE_THRESHOLD}")
            st.write("• Top Predictions Displayed: 3")
            st.write("• AI Model: LLaMA 3.1 8B Instant")
            st.write("• Vectorization: TF-IDF")
            
            st.markdown("**Feature Flags**")
            st.write("• AI Symptom Enhancement: Enabled")
            st.write("• Fallback AI Prediction: Enabled")
            st.write("• Real-time Information: Enabled")

# -------------------
# About Page
# -------------------
elif page == "About":
    st.title("ℹ️ About MediAssist")
    
    st.image("https://placehold.co/800x200/e8f0fe/1967d2?text=MediAssist+V3+-+AI-Powered+Symptom+Analysis&font=inter", 
             use_container_width=True)
    
    st.markdown("""
    ## 🩺 Our Mission
    
    MediAssist is an AI-powered symptom analysis tool designed to provide preliminary health insights 
    and support healthcare decision-making, especially in underserved and rural communities.
    
    ## 🚀 How It Works
    
    **Multi-Stage AI Analysis:**
    
    1.  **🔍 TF-IDF Similarity Matching** 
        - Compares input symptoms with our medical database using text similarity
        - Provides evidence-based matches with confidence scores
    
    2.  **🤖 AI Symptom Enhancement**
        - When confidence is low, uses Groq's LLaMA to suggest related medical terms
        - Enhances matching accuracy through semantic understanding
    
    3.  **💡 AI-Powered Information**
        - Provides general treatment information for matched conditions
        - Offers educational content about various diseases
    
    4.  **🆘 AI Fallback Prediction**
        - When no strong matches are found, provides direct AI analysis
        - Always accompanied by strong medical disclaimers
    
    ## 🌍 Alignment with Sustainable Development Goals
    
    **SDG 3: Good Health & Well-Being**
    - Improving health literacy and access to health information
    - Supporting early symptom recognition and healthcare seeking behavior
    - Reducing barriers to preliminary health information
    
    ## 🛠️ Technical Architecture
    
    - **Frontend:** Streamlit Web Application
    - **AI/ML:** Scikit-learn TF-IDF + Cosine Similarity
    - **LLM Integration:** Groq API with LLaMA 3.1
    - **Data:** Medical symptom-disease correlation database
    
    ## ⚠️ Important Disclaimers
    
    - 🚨 **Not a Medical Diagnostic Tool** - Always consult healthcare professionals
    - 🔒 **Privacy Focused** - Symptoms are processed anonymously
    - 📚 **Educational Purpose** - Designed for information and awareness
    - 🌐 **Accessibility** - Built for low-bandwidth environments
    
    ## 👥 Target Users
    
    - Individuals seeking preliminary health information
    - Community health workers in rural areas
    - Healthcare students and educators
    - General public for health literacy improvement
    
    ## 📞 Support & Contact
    
    For technical support or to report issues, please contact the development team.
    
    **Version:** 3.0 | **Last Updated:** 2024
    """)
    
    st.success("Thank you for using MediAssist! Together, we're making health information more accessible.")

# -------------------
# Footer (All Pages)
# -------------------
st.markdown("---")
st.caption("MediAssist V3 | Supporting Rural Health | AI Enhanced | SDG 3: Good Health & Well-Being")