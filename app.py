import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gradio as gr
import numpy as np
import cv2
import keras

model = keras.models.load_model('face_alignment_model.keras')


def predict_landmarks(input_image):
    if input_image is None:
        return None

    print("1. Image received...")
    original_h, original_w = input_image.shape[:2]
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    TRAINING_IMG_SIZE = 256
    resized_img = cv2.resize(gray_img, (TRAINING_IMG_SIZE, TRAINING_IMG_SIZE))
    normalized_img = resized_img / 255.0
    input_array = np.expand_dims(normalized_img, axis=(0, -1)).astype(np.float32)

    print("2. Running forward pass...")
    preds = model(input_array, training=False).numpy()

    print("3. Drawing points...")
    pts = preds.reshape(-1, 2)
    scale_x = original_w / TRAINING_IMG_SIZE
    scale_y = original_h / TRAINING_IMG_SIZE

    output_image = input_image.copy()
    for (x, y) in pts:
        cx = int(x * scale_x)
        cy = int(y * scale_y)
        cv2.circle(output_image, (cx, cy), radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(output_image, (cx, cy), radius=12, color=(255, 255, 255), thickness=2)

    print("4. Done!")
    return output_image

demo = gr.Interface(
    fn=predict_landmarks,
    inputs=gr.Image(type="numpy", label="Upload Face Image"),
    outputs=gr.Image(type="numpy", label="Predicted Landmarks"),
    title="Facial Landmark Detection CNN",
    description="Upload an image to see the neural network predict 5 facial landmarks in real-time.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)

