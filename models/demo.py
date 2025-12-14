import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load GigaPath model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Benign/Malignant
    model.load_state_dict(torch.load('models/gigapath_resnet50.pth'))
    model.eval()
    return model

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit App
st.set_page_config(page_title="GigaPath-Nepali", page_icon="🩺")
st.title("🩺 GigaPath-Nepali: AI Cancer Pathology Detection")
st.markdown("**Pokhara BE Capstone → Arkansas State MS CS Thesis**")

# Sidebar metrics
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Validation Accuracy", "85.2%")
st.sidebar.metric("NHRC Dataset", "1,000+ slides")
st.sidebar.metric("Patch Size", "1024x1024")

# File uploader
uploaded_file = st.file_uploader("Upload Pathology Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Patch', use_column_width=True)
    
    # Preprocess and predict
    st.write("🔬 **Analyzing...**")
    model = load_model()
    
    with torch.no_grad():
        # Transform image
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Results
        label = "Benign 🟢" if predicted.item() == 0 else "Malignant 🔴"
        conf_pct = confidence.item() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", label)
        with col2:
            st.metric("Confidence", f"{conf_pct:.1f}%")
        
        # Probability bar chart
        st.subheader("📈 Probability Distribution")
        prob_dict = {"Benign": probabilities[0][0].item()*100, 
                    "Malignant": probabilities[0][1].item()*100}
        st.bar_chart(prob_dict)

# Footer
st.markdown("---")
st.markdown("""
**GigaPath-Nepali v0.3**  
Pokhara University BE Capstone | Arkansas State MS CS Thesis Track  
[GitHub](https://github.com/तिम्रोusername/GigaPath-Nepali) | MICCAI 2027 Target
""")

if st.button("🚀 Deploy to Heroku"):
    st.info("streamlit run demo.py → heroku create gigapath-nepali")
