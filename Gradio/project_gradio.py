import gradio as gr
from joblib import load
import numpy as np

# Load the trained model using joblib
model = load("house_model.pkl")

def predict_price(num_rooms, area):
    # Assuming the model takes input as a NumPy array
    features = np.array([[area, num_rooms]])  # Adjusted feature order
    prediction = model.predict(features)[0][0] * 1_000_000  # Convert to full Rwandan Francs
    return f"Predicted Price: {prediction:,.0f} Rwf"

# Gradio interface
with gr.Blocks(css="body { display: flex; justify-content: center; align-items: center; height: 100vh; }") as demo:
    with gr.Column():
        gr.Markdown("# House Predictions")
        
        num_rooms = gr.Dropdown(choices=list(range(1, 21)), value=1, label="Select Number of Rooms")
        area = gr.Number(label="Enter Area (in square meters)")
        predict_button = gr.Button("Predict")
        output = gr.Textbox(label="Predicted Price")
        
        predict_button.click(predict_price, inputs=[num_rooms, area], outputs=output)

demo.launch()
