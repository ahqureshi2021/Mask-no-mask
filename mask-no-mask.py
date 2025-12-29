import gradio as gr
from ultralytics import YOLO
import numpy as np


# 1. Load the model
model = YOLO("https://github.com/ahqureshi2021/Mask-no-mask/raw/refs/heads/main/best.pt")
 

def predict_mask(input_img):
    """
    Takes an image, runs YOLOv11 inference, and returns an annotated image.
    """
    if input_img is None:
        return None
        

    results = model.predict(source=input_img, conf=0.25)
    

    annotated_img_bgr = results[0].plot()
    

    annotated_img_rgb = annotated_img_bgr[..., ::-1]
    
    return annotated_img_rgb

# 2. Define the Gradio Interface
demo = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="numpy", label="Upload Image for Mask Detection"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="Mask Detection System",
    description="Detecting mask in real-time using YOLOv11.",
    flagging_mode="never"
)

# 3. Launch the app
if __name__ == "__main__":
    
    demo.launch(inbrowser=True, share=True)
    