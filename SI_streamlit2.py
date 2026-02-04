import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
from streamlit.components.v1 import html

# Set up the Streamlit app
st.title("Speech/Text to Image Converter")
st.markdown("### Using Speech Recognition and Stable Diffusion")
st.markdown("Please be patient, Image Generation takes some time.")

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        
        try:
            st.info("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        return None

# Function to generate image
@st.cache_resource
def load_pipeline():
    modelid = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16)
    pipe.to(device)
    return pipe

def generate_image(prompt):
    pipe = load_pipeline()
    with torch.autocast("cuda"):
        output = pipe(prompt, guidance_scale=8.5)
    return output.images[0]

# HTML and CSS Content for the Frontend
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Speech to Image App</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {
      background: #1a202c;
      color: white;
      font-family: 'Inter', sans-serif;
      margin: 0;
      padding: 0;
      overflow-x: hidden;
      transition: all 0.5s ease-in-out;
    }
    header {
      background-image: url('https://source.unsplash.com/1600x900/?technology,artificial-intelligence');
      background-size: cover;
      background-position: center;
      height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      color: white;
      padding: 1rem;
      box-shadow: inset 0 0 50px rgba(0,0,0,0.7);
      transition: all 0.5s ease;
    }
    header:hover {
      background-position: top;
    }
    header h1 {
      font-size: 4.5rem;
      margin-bottom: 2rem;
      text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
      animation: fadeIn 2s ease-in-out;
    }
    header p {
      font-size: 1.3rem;
      margin-bottom: 2rem;
      opacity: 0.8;
    }
    header button {
      padding: 1rem 2rem;
      font-size: 1.5rem;
      background-color: #00c9ff;
      color: #000;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    header button:hover {
      background-color: #00a0cc;
      transform: translateY(-5px);
    }
    .container {
      padding: 4rem 2rem;
      max-width: 1200px;
      margin: auto;
      text-align: center;
      position: relative;
      z-index: 10;
    }
    .speech-box {
      display: flex;
      justify-content: center;
      margin-bottom: 3rem;
    }
    .speech-box input {
      padding: 1rem;
      width: 70%;
      font-size: 1.2rem;
      border-radius: 8px;
      border: 1px solid #3c3c3c;
      background-color: #2d3748;
      color: white;
      margin-right: 1rem;
      transition: all 0.3s ease;
    }
    .speech-box input:focus {
      border-color: #00c9ff;
    }
    .speech-box button {
      padding: 1rem;
      font-size: 1.2rem;
      background-color: #ff6f61;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s ease;
    }
    .speech-box button:hover {
      background-color: #ff4a39;
    }
    .image-preview {
      margin-top: 3rem;
      transition: transform 0.3s ease;
    }
    .image-preview img {
      width: 100%;
      max-width: 800px;
      height: auto;
      border-radius: 12px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
      transition: transform 0.3s ease;
    }
    .image-preview img:hover {
      transform: scale(1.05);
    }
    footer {
      background-color: #111;
      padding: 2rem;
      text-align: center;
      color: white;
      z-index: 9999;
    }
    footer p {
      margin-bottom: 1rem;
    }
    .social-icons a {
      color: #fff;
      font-size: 2rem;
      margin: 0 15px;
      transition: transform 0.3s ease;
    }
    .social-icons a:hover {
      transform: scale(1.2);
      color: #00c9ff;
    }
    @media screen and (max-width: 768px) {
      header h1 {
        font-size: 3rem;
      }
      .speech-box input {
        width: 60%;
      }
      .speech-box button {
        padding: 1rem;
        font-size: 1rem;
      }
    }

    @keyframes fadeIn {
      0% {
        opacity: 0;
      }
      100% {
        opacity: 1;
      }
    }
  </style>
</head>
<body>

  <header>
    <h1>Transform Your Voice Into Art</h1>
    <p>Simply speak or type, and watch your imagination come to life.</p>
    <button onclick="document.getElementById('generate').scrollIntoView({ behavior: 'smooth' });">Start Now</button>
  </header>

  <div class="container" id="generate">
    <h2>Generate Stunning Images from Your Descriptions</h2>
    <p>Provide a prompt by typing or speaking to create an image using AI technology.</p>
    <div class="speech-box">
      <input type="text" id="promptInput" placeholder="Describe what you want to see..." />
      <button onclick="startSpeechRecognition()">Use Voice Input</button>
    </div>
    <div class="image-preview">
      <img id="generatedImage" src="https://source.unsplash.com/800x400/?dream,vision" alt="Generated Image" />
    </div>
  </div>

  <footer>
    <p>Made with ❤️ using Streamlit & Stable Diffusion</p>
    <div class="social-icons">
      <a href="#"><i class="fab fa-twitter"></i></a>
      <a href="#"><i class="fab fa-github"></i></a>
      <a href="#"><i class="fab fa-linkedin"></i></a>
    </div>
  </footer>

  <script>
    function startSpeechRecognition() {
      const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
      recognition.lang = 'en-US';
      recognition.start();

      recognition.onresult = function (event) {
        const speechText = event.results[0][0].transcript;
        document.getElementById('promptInput').value = speechText;
        generateImage(speechText);
      };

      recognition.onerror = function (event) {
        alert("Speech recognition error: " + event.error);
      };
    }

    async function generateImage(prompt) {
      try {
        const response = await fetch('http://localhost:8000/generate-image', {  
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt: prompt }),
        });

        if (response.ok) {
          const result = await response.json();
          document.getElementById('generatedImage').src = result.image_url;
        } else {
          console.error('Error generating image');
        }
      } catch (error) {
        console.error('Error:', error);
      }
    }
  </script>

</body>
</html>
"""

# Display HTML frontend in Streamlit
html(html_content)

# Text input for prompt (you may want to connect this with the speech input too)
prompt_text = st.text_input("Enter a prompt for the image generation:")

# Button to trigger speech recognition
if st.button("Recognize Speech"):
    recognized_text = recognize_speech()
    if recognized_text:
        prompt_text = f"{recognized_text}, 4k, High Resolution"
        if prompt_text:
            st.text_input("Recognized Prompt", value=prompt_text)
            with st.spinner("Generating image..."):
                image = generate_image(prompt_text)
                st.image(image, caption="Generated Image", use_container_width=True)
                st.success("Image generated successfully!")

# Generate button
if st.button("Generate Image"):
    if prompt_text:
        with st.spinner("Generating image..."):
            image = generate_image(prompt_text)
            st.image(image, caption="Generated Image", use_container_width=True)
            st.success("Image generated successfully!")

#python -m streamlit run SI_streamlit2.py
