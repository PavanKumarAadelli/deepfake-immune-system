# 🛡️ Adversarial Deepfake Immune System

**Protect your digital identity using Adversarial Machine Learning.**

This project is a web-based tool that applies "Proactive Noise" to facial images. This noise is imperceptible to the human eye but acts as a "vaccine" against Deepfake generators. If a malicious actor tries to use your protected photo to train a Deepfake model, the result will be a distorted, glitched failure.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

(https://deepfake-immune-system99.streamlit.app/)


<img width="1066" height="735" alt="deepfake-immune-system" src="https://github.com/user-attachments/assets/135b6bb2-78c9-405b-93bb-9c58a70db003" />


---

## 📖 The Problem: Deepfakes
Deepfake technology is advancing rapidly, allowing anyone to replace a person's face in a video or image with startling accuracy. This poses a massive threat to privacy, consent, and identity security.

## 💡 The Solution: Adversarial Noise
Instead of trying to detect deepfakes after they are made (which is hard), this tool prevents them from being made in the first place.

It uses an **FGSM (Fast Gradient Sign Method)** attack to generate a specific layer of noise tailored to your face.
- **Human Vision:** Sees the original photo.
- **AI Vision:** Sees mathematical chaos.

### How it works:
1. The tool identifies your facial embedding (mathematical identity).
2. It calculates the gradients (directions) that would confuse a Face Recognition model.
3. It overlays this pattern onto your image.
4. When a Deepfake AI tries to read your face, it fails to identify your features, resulting in a "broken" generation.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (Interactive Web App)
- **Backend:** Python
- **Deep Learning:** [PyTorch](https://pytorch.org/), [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)
- **Attack Method:** Fast Gradient Sign Method (FGSM)
- **Deployment:** Hugging Face Spaces (Docker) or Streamlit Cloud

---

## 💻 Installation & Local Usage

If you want to run this project on your local machine:

**1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-immune-system.git
cd deepfake-immune-system
```

**2. Install Dependencies**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

**3. Run the App**
```bash
streamlit run app.py
```

---

## 🧠 Code Logic (Simplified)

The core of the project relies on the `generate_protection` function:

```python
# 1. Load Pre-trained Face Recognition Model
model = InceptionResnetV1(pretrained='vggface2').eval()

# 2. Get the face identity vector (Embedding)
embedding = model(image)

# 3. Calculate how to break that embedding
loss = torch.nn.MSELoss()(embedding, target_noise)
loss.backward() # Calculate gradients

# 4. Apply the noise
protected_image = image + (epsilon * gradient.sign())
```


## ⚠️ Disclaimer
This tool is for **educational and defensive purposes only**. It demonstrates vulnerabilities in machine learning models (Adversarial Attacks) to help users protect their privacy.


## 📄 License
This project is open source and available under the [MIT License](LICENSE).
