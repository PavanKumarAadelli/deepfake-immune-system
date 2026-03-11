import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from facenet_pytorch import InceptionResnetV1

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deepfake Immune System", page_icon="🛡️")

# --- LOAD THE AI MODEL ---
# @st.cache_resource ensures the model loads only ONCE and stays in memory.
# This makes the app much faster for subsequent users.
@st.cache_resource
def load_model():
    device = torch.device('cpu') # Hugging Face free CPUs are sufficient for this
    try:
        # We download the pre-trained weights automatically
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

# --- CORE LOGIC: ADVERSARIAL ATTACK ---
def generate_protection(image_bytes, epsilon=0.03):
    """
    Adds invisible noise to the image to break Deepfake AI.
    """
    # 1. Load Image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Prepare Image for AI (Resize & Tensor)
    # We resize to 160px as the AI expects this input size
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True # Tell PyTorch to track changes

    # 3. Get the "Identity" Vector
    embedding = model(img_tensor)

    # 4. Create the Target
    # We generate random noise to force the AI to change its mind
    target_label = torch.randn_like(embedding).to(device)

    # 5. Calculate Loss
    # This measures how far the AI is from our random target
    loss = nn.MSELoss()(embedding, target_label)
    loss.backward() # Calculate the gradients (the "noise directions")

    # 6. Apply the Noise (FGSM Attack)
    data_grad = img_tensor.grad.data
    perturbed_image = img_tensor + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 7. Convert back to viewable image
    final_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
    
    # Save to memory (bytes) to let user download
    buf = io.BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    return byte_im

# --- THE WEB INTERFACE ---
st.title("🛡️ Adversarial Deepfake Immune System")
st.markdown("""
**Protect your photos from AI manipulation.**

This tool adds invisible noise to your image. To humans, it looks the same, but to Deepfake AI, your face becomes unrecognizable, resulting in glitched or failed deepfakes.
""")

# Check if model loaded correctly
if model is None:
    st.error("Model failed to load. Please check the logs.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload a clear face photo", type=["jpg", "png", "jpeg"])

# Slider for Noise Strength
epsilon = st.slider("Protection Strength", 0.01, 0.10, 0.03, 
                    help="Higher values provide stronger protection but might slightly affect image quality.")

if uploaded_file is not None:
    # Show Original
    st.subheader("Your Original Photo")
    st.image(uploaded_file, width=300)

    # Button to Start
    if st.button("🛡️ Generate Protection"):
        with st.spinner("Analyzing facial geometry and generating adversarial noise..."):
            try:
                # Get file bytes
                img_bytes = uploaded_file.read()
                
                # Run the protection logic
                protected_bytes = generate_protection(img_bytes, epsilon)
                
                # Show Result
                st.subheader("Protected Photo")
                st.image(protected_bytes, width=300)
                st.success("✅ Done! Download the image below.")
                
                # Download Button
                st.download_button(
                    label="📥 Download Protected Image",
                    data=protected_bytes,
                    file_name="protected_face.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.warning("Please try a different image or ensure the face is clearly visible.")

st.markdown("---")
st.caption("Educational Project - Adversarial Machine Learning")
