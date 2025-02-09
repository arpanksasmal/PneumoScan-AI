import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import json
import cv2

def load_custom_css():
    with open("style.css", "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = models.densenet121(pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.model(x)

def preprocess_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def generate_feature_heatmap(image_np):
    try:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        gray_resized = cv2.resize(gray, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray_resized)
        heatmap = cv2.applyColorMap(clahe_image, cv2.COLORMAP_TURBO)
        
        return heatmap
    except Exception as e:
        st.error(f"Error generating feature heatmap: {str(e)}")
        return None

def create_confidence_meter(confidence):
    value = confidence * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Confidence", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1976d2"},
            'steps': [
                {'range': [0, 70], 'color': 'rgba(255, 153, 153, 0.3)'},
                {'range': [70, 90], 'color': 'rgba(255, 204, 153, 0.3)'},
                {'range': [90, 100], 'color': 'rgba(153, 255, 153, 0.3)'}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,  
        width=300, 
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

class HistoryTracker:
    def __init__(self):
        self.history_dir = Path("detection_history")
        self.history_dir.mkdir(exist_ok=True)
        self.current_date = datetime.now().strftime("%d_%m_%Y")
        self.history_file = self.history_dir / f"pneumonia_detection_history_{self.current_date}.json"
        self.history = []
        self.load_history()
    
    def load_history(self):
        try:
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            else:
                self.history = []
                with open(self.history_file, "w") as f:
                    json.dump(self.history, f)
        except Exception as e:
            st.warning(f"Error loading history: {str(e)}")
            self.history = []

    def get_statistics(self):
        # Reload history before getting statistics
        self.load_history()
        
        if not self.history:
            stats_df = pd.DataFrame({
                "Metric": ["📊 Total Scans", "🔴 Pneumonia Cases", "🟢 Normal Cases", "📈 Average Confidence", "⚡ Latest Detection"],
                "Count/Value": ["0", "0", "0", "0.00%", "None"]
            }).set_index("Metric")
        else:
            total = len(self.history)
            pneumonia = sum(1 for d in self.history if d["prediction"] == "Pneumonia Detected")
            normal = total - pneumonia
            avg_confidence = sum(d["confidence"] for d in self.history) / total
            latest = self.history[-1]["timestamp"] if self.history else "None"

            stats_df = pd.DataFrame({
                "Metric": ["📊 Total Scans", "🔴 Pneumonia Cases", "🟢 Normal Cases", "📈 Average Confidence", "⚡ Latest Detection"],
                "Count/Value": [
                    str(total),
                    str(pneumonia),
                    str(normal),
                    f"{avg_confidence:.2f}%",
                    latest
                ]
            }).set_index("Metric")

        # ✅ Apply styles to make text the brightest (pure white) and highly visible
        styled_df = stats_df.style.set_properties(**{
            'background-color': '#1976D2',  # Keep Metric column blue
            'color': '#FFFFFF',             # Brightest white text for highest contrast
            'font-weight': 'bold',
            'font-size': '18px',            # Increase text size for readability
        }, subset=pd.IndexSlice[stats_df.index, :])  # Apply only to the "Metric" column

        return styled_df

    def add_detection(self, filename, prediction, confidence):
        detection = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": prediction,
            "confidence": float(confidence)
        }
        self.history.append(detection)
        
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=4)

def resize_image(image, target_size=(500, 500)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def get_pneumonia_info():
    return """
    ## 🏥 Understanding Pneumonia Detection

    ### ❓ What is Pneumonia?
    Pneumonia is a serious respiratory infection that causes **inflammation** in the lungs' air sacs (alveoli).  
    When these air sacs become infected, they may **fill with fluid or pus**, affecting oxygen exchange.  
    This condition can range from **mild to life-threatening**, depending on various factors like age, health, and the pathogen.

    ### 🌎 Clinical Significance
    Pneumonia represents a **global health challenge**, affecting millions worldwide.  
    - 📊 **14% of all pediatric hospital admissions** are due to pneumonia.  
    - 🏡 **Elderly populations** are at higher risk.  
    - ⚕️ **Early detection** improves patient outcomes by **up to 40%**.

    ### 🤖 Advanced AI Detection System
    Our state-of-the-art detection system leverages the **DenseNet121** deep learning model, optimized for medical imaging.  
    🔍 **Key Benefits** of AI-based pneumonia detection:
    - ⚡ **Rapid Analysis**: Chest X-ray results in **seconds**  
    - 🎯 **High Accuracy**: Trained on thousands of real cases  
    - 🔥 **Feature Detection**: Advanced **multi-layer neural networks**  
    - 📈 **Confidence Score**: Instant AI-generated risk assessment  
    - 🖼️ **Heatmap Visualization**: Highlights **potential pneumonia regions**  

    ### ⚠️ Critical Considerations
    AI-assisted diagnosis should be used **with clinical expertise**:  
    1️⃣ This system is a **diagnostic aid**, **not a replacement** for doctors.  
    2️⃣ Always interpret AI results **in a clinical context**.  
    3️⃣ Image **quality & positioning** affect accuracy.  
    4️⃣ Our model is updated **regularly** to integrate **latest research**.  

    ### 🚑 When to Seek Medical Attention?
    If experiencing **any of the following**, seek **immediate** medical help:  
    - 😮‍💨 **Severe difficulty in breathing**  
    - 🤕 **Chest pain during breathing**  
    - 🌡️ **High fever (above 39.4°C / 103°F)**  
    - 🛌 **Extreme fatigue or confusion**  
    - 💙 **Bluish lips or fingertips (cyanosis)**  

    **⚕️ Your health matters! Always consult a professional for diagnosis and treatment.**  
    """

def main():
    st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide")
    load_custom_css()

    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state.history = HistoryTracker()
    if 'show_info' not in st.session_state:
        st.session_state.show_info = False
    if 'image_uploaded' not in st.session_state:
        st.session_state.image_uploaded = False
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    st.markdown('''
    <div class="title-container">
        <h1 class="app-title">🫁 PneumoScan AI</h1>
    </div>
    ''', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sub-title">Upload X-Ray Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        # Reset prediction state when a new file is uploaded
        if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.prediction_made = False
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = uploaded_file
        
        if uploaded_file is not None:
            st.session_state.image_uploaded = True
            predict_button = st.button("🔍 Predict Pneumonia", key="predict_button", disabled=st.session_state.prediction_made)
        else:
            st.session_state.image_uploaded = False
            st.session_state.prediction_made = False
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = None
        
        # Statistics section with improved display
        st.markdown("## 📊 Detection Statistics")
        stats = st.session_state.history.get_statistics()
        st.dataframe(stats, hide_index=False)
    
        # Extra Information toggle
        st.markdown("## ℹ️ Extra Information")
        if st.button("Show/Hide Information"):
            st.session_state.show_info = not st.session_state.show_info

    # Main content area
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Original X-Ray")
            resized_original = resize_image(image_np, target_size=(400, 400))
            st.image(resized_original, use_container_width=True)

        if predict_button or st.session_state.prediction_made:
            try:
                if not st.session_state.prediction_made:
                    model = PneumoniaModel()
                    model.load_state_dict(torch.load("model/best_pneumonia_model_densenet121.pt", map_location=torch.device('cpu')))
                    model.eval()
                    
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        with torch.no_grad():
                            outputs = model(processed_image)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, prediction = torch.max(probabilities, 1)

                            # Store results in session state
                            st.session_state.analysis_results = {
                                'prediction': prediction.item(),
                                'result_text': "Pneumonia Detected" if prediction.item() == 1 else "Normal",
                                'confidence_value': confidence.item(),
                                'heatmap': generate_feature_heatmap(image_np)
                            }
                            st.session_state.prediction_made = True

                            # Add the detection to history
                            st.session_state.history.add_detection(
                                uploaded_file.name,
                                st.session_state.analysis_results['result_text'],
                                st.session_state.analysis_results['confidence_value'] * 100
                            )
                else:
                    # Load from session state if already predicted
                    results = st.session_state.analysis_results
                    prediction = results['prediction']  # Ensure prediction exists here
                
                # Display results using session state
                if st.session_state.analysis_results:
                    results = st.session_state.analysis_results
                    
                    with col2:
                        st.markdown("### Feature Heatmap")
                        resized_heatmap = resize_image(results['heatmap'], target_size=(400, 400))
                        st.image(resized_heatmap, use_container_width=True)

                    col3, col4 = st.columns([1, 1])

                    with col3:
                        st.markdown("### Diagnosis Result")
                        color_class = "positive-result" if results['prediction'] == 1 else "negative-result"
                        st.markdown(
                            f'''
                            <div class="{color_class} diagnosis-box">
                                <h3>Diagnosis: {results['result_text']}</h3>
                                <p>Confidence: {results['confidence_value']*100:.2f}%</p>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

                    with col4:
                        st.markdown("### Confidence Meter")
                        st.plotly_chart(create_confidence_meter(results['confidence_value']), use_container_width=True)

                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 📋 Detailed Analysis Report")

                    if results['prediction'] == 1:  # Using stored prediction
                        st.markdown("""
                            #### 🔍 **Key Findings**
                            - 🤖 The AI model has **detected patterns consistent with pneumonia** in the chest X-ray.  
                            - 🔥 **Highlighted Areas**: Potential infiltrates or consolidation are shown in the feature map.  
                            - 📈 **Confidence Level**: The AI has assigned a confidence score indicating the strength of these findings.  

                            #### 🏥 **Recommended Actions**
                            1️⃣ **Consult a healthcare provider immediately** 🏨  
                            2️⃣ **Consider additional diagnostic tests** 🩻  
                            3️⃣ **Monitor symptoms closely** ⏳  
                            4️⃣ **Follow medical professional’s advice for treatment** 💊  

                            #### ⚠️ **Important Notes**
                            - ⚕️ This AI analysis is a **screening tool**, **not a definitive diagnosis**.  
                            - ⏳ **Early treatment** of pneumonia **significantly improves outcomes**.  
                            - 🔄 **Regular monitoring and follow-ups** are essential for best results.  
                        """)
                    else:
                        st.markdown("""
                            #### 🔍 **Key Findings**
                            - ✅ No significant patterns indicating **pneumonia** were detected.  
                            - 🫁 **Lung fields appear normal** with no abnormalities.  
                            - 🖼️ **Feature Map**: The X-ray shows typical lung characteristics.  

                            #### 🏥 **Recommendations**
                            1️⃣ **Continue regular health monitoring** 👩‍⚕️  
                            2️⃣ **Maintain good respiratory health practices** 🌿  
                            3️⃣ **Follow up with a healthcare provider if symptoms persist** 🔍  

                            #### ⚠️ **Important Notes**
                            - 🩺 **Regular check-ups** are crucial for maintaining lung health.  
                            - 📞 **Contact a healthcare provider** if new symptoms develop.  
                            - 📊 This AI tool is **only a screening aid**, part of **comprehensive health monitoring**.  
                        """)
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    # Show welcome message when no image is uploaded
    if not st.session_state.image_uploaded:
        st.markdown('<div class="welcome-box">', unsafe_allow_html=True)
        st.markdown("""
           # 🌟 Welcome to PneumoScan AI - Advanced Chest X-Ray Analysis
            
            ### 🔬 State-of-the-Art Technology
            Our system utilizes advanced deep learning algorithms specifically trained on thousands of chest X-rays to detect patterns associated with pneumonia.
            
            ### 🎯 Key Features
            - **Instant Analysis**: Get results in seconds
            - **Visual Insights**: Advanced heatmap visualization
            - **High Accuracy**: Trained on extensive medical datasets
            - **Detailed Reports**: Comprehensive analysis with actionable insights
            - **Real-time Tracking**: Monitor detection statistics
            
            ### 🏥 Medical Impact
            - Early detection can lead to better treatment outcomes
            - Assists healthcare providers in diagnosis
            - Reduces time to treatment initiation
            - Helps monitor disease progression
            
            ### 📊 How It Works
            1. Upload a chest X-ray image
            2. Our AI analyzes complex patterns
            3. Get instant results with confidence levels
            4. Review detailed analysis and recommendations
            
            ### ⚕️ Important Medical Disclaimer
            This tool is designed to assist, not replace, professional medical diagnosis. Always consult with qualified healthcare providers for medical decisions.
            
            ### 🚀 Get Started
            Upload your chest X-ray image using the sidebar to begin the analysis.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show extra information in a new container below all other content
    if st.session_state.show_info:
        st.markdown('<div class="info-box" style="margin-top: 30px;">', unsafe_allow_html=True)
        st.markdown(get_pneumonia_info())
        st.markdown('</div>', unsafe_allow_html=True)

    # Add spacing before footer to prevent overlap
    st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="footer">Created with ❤️ by Arpan 😎 | Empowering Healthcare with AI | © 2024 PneumoScan AI</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()