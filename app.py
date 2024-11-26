import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from google.cloud import vision
import io
from transformers import pipeline

# Initialize the Google Vision client
client = vision.ImageAnnotatorClient()

# Initialize the Hugging Face GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# Function to process the drawn image using Google Vision API
def process_image(image_data):
    # Convert the image data to an Image for processing
    image = Image.fromarray(image_data.astype('uint8'))
    
    # Convert to RGB format if in another mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save image temporarily
    image_path = "temp_image.png"
    image.save(image_path)

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Google Vision API for text detection (can be extended for other uses like labels, logos, etc.)
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Get the detected text from the response
    if texts:
        detected_text = texts[0].description
        return detected_text
    else:
        return "No text detected"

# Streamlit UI
st.title("Calculator & Note Taking App")

# Set up the drawing canvas
st.subheader("Draw your diagram, equations, or notes here:")
canvas_result = st_canvas(
    fill_color="white",  # Canvas color
    stroke_color="black",  # Drawing color
    stroke_width=3,
    background_color="white",
    width=700,
    height=400,
    drawing_mode="freedraw",  # Or use 'line', 'rectangle', etc.
    key="canvas",
)

# If the user draws something, process the image
if canvas_result.image_data is not None:
    # Convert the drawn canvas to an image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    st.image(img, caption="Your Drawing", use_column_width=True)

    # Get the processed result from Google Vision API
    st.subheader("Processed Output:")
    detected_text = process_image(canvas_result.image_data)
    st.write(detected_text)

    # If the detected text is not empty, send it to GPT-2 for generating a response
    if detected_text != "No text detected":
        st.subheader("Response from GPT-2:")
        # Generate the response using GPT-2
        response = generator(detected_text, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.write(response)
