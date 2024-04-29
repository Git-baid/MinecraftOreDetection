from ultralytics import YOLO
import os, random
import mss
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

confidence_threshold = 0.2 # determine minimum confidence level in order to display
pred_size = 640 # resize prediction inputs (lower values increase speed but decrease accuracy)
half = True # Speeds up inference with minimal impact on accuracy on suppored GPU
device = '0' # Use GPU for inference
agnostic_nms = True # combine same classes very close together

def draw_boxes_on_image(image, detections): # [class_id, confidence, x, y, width, height]
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Define colors for each class (modify as needed)

    for detection in detections:
        class_id, confidence, x, y, width, height = detection
        color = colors[class_id % len(colors)]
        box = [(x, y), (x + width, y + height)]
        draw.rectangle(box, outline=color)
        draw.text((x, y), f"{class_id} ({confidence:.2f})", fill=color)

    return image

model = YOLO("OreDetectionModelv2.1.pt")

# Initialize tkinter window
root = tk.Tk()
root.title("Minecraft Ore Detection")
root.resizable(False, False)

# Create a label to display the screenshot
screenshot_label = tk.Label(root)
screenshot_label.pack()

with mss.mss() as sct:
    monitor = sct.monitors[2] # which monitor to capture
    while True:
        monitor_screenshot = sct.grab(monitor)
        screenshot_img = Image.frombytes("RGB", monitor_screenshot.size, monitor_screenshot.bgra, "raw", "BGRX") # Convert to PIL

        # Use the screenshot as the input source for prediction
        results = model.predict(source=screenshot_img, 
                                stream=True, 
                                conf=confidence_threshold,
                                imgsz=pred_size,
                                half = half,
                                device = device,
                                agnostic_nms = agnostic_nms)

        for result in results:
            result.save(filename="frameresult.jpg")
            img_result = Image.open("frameresult.jpg")

            # Convert the PIL image to a Tkinter PhotoImage
            tk_img = ImageTk.PhotoImage(img_result)
            screenshot_label.config(image=tk_img)
            screenshot_label.image = tk_img
            root.update_idletasks()
            root.update()


