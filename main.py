import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import torch
import pyttsx3

engine = pyttsx3.init()

# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("best.pt")   
class_names = model.names

# ---------------------------
# Initialize Tkinter
# ---------------------------
root = tk.Tk()
root.title("Dystonia Detection System")
root.geometry("900x650")
root.resizable(False, False)

# ---------------------------
# Background Image
# ---------------------------
bg_image = Image.open("background.jpeg")
bg_image = bg_image.resize((900, 600), Image.Resampling.LANCZOS)

bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# ---------------------------
# Header
# ---------------------------
header_frame = tk.Frame(root, bg="#000000", pady=10)

header_frame.pack(fill="x")

header_label = tk.Label(
    header_frame,
    text="🧠 Dystonia Detection System 🌀",
    font=("Helvetica", 26, "bold"),
    fg="white",
    bg="#000000"
)
header_label.pack()

# ---------------------------
# Display Frame for Images / Live Feed
# ---------------------------
display_frame = tk.Label(root, bg="white", bd=3, relief="solid")
display_frame.place(x=150, y=120, width=600, height=350)

# ---------------------------
# Result Label (Class + Probability)
# ---------------------------
result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Helvetica", 18, "bold"),
    fg="white",
    bg="#000000",
    pady=10
)
result_label.place(x=150, y=490, width=600)

# ---------------------------
# Functions
# ---------------------------
camera_active = False


def predict_image(image_path):
    results = model.predict(
        source=image_path, save=False, imgsz=640, conf=0.25)
    img = cv2.imread(image_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[cls]} ({conf*100:.2f}%)"
            color = (0, 255, 0) if class_names[cls] == "non dystonia" else (
                0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            print(label)
            mytext = ""
            if class_names[cls] == "Non_Dystonia":
                print("if")
                mytext = "Dystonia is not detected"
            elif class_names[cls] == "Dystonia":
                mytext = "Dystonia is detected"
            print("my",mytext)

            engine.say(mytext)
            engine.runAndWait()

            # update result label
            result_label.config(
                text=f"Prediction: {class_names[cls]}  |  Confidence: {conf*100:.2f}%"
            )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (500, 350))
    imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    display_frame.imgtk = imgtk
    display_frame.configure(image=imgtk)


def browse_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
    )
    if file_path:
        selected_image_path = file_path
        img = Image.open(file_path)
        img = img.resize((500, 350))
        imgtk = ImageTk.PhotoImage(img)
        display_frame.imgtk = imgtk
        display_frame.configure(image=imgtk)

        # Show Predict Button after selecting image


def predict_button_action():
    if selected_image_path:
        predict_image(selected_image_path)


def live_detection():
    global camera_active
    camera_active = True

    cap = cv2.VideoCapture(0)

    def run_camera():
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{class_names[cls]} ({conf*100:.2f}%)"
                    color = (0, 255, 0) if class_names[cls] == "non dystonia" else (
                        0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    result_label.config(
                        text=f"Prediction: {class_names[cls]}  |  Confidence: {conf*100:.2f}%"
                    )
                    mytext = ""
                    if class_names[cls] == "Non_Dystonia":
                        mytext = "Non Dystonia"
                    elif class_names[cls] == "Dystonia":
                        mytext = "Dystonia"

                    engine.say(mytext)
                    engine.runAndWait()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (550, 350))
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            display_frame.imgtk = imgtk
            display_frame.configure(image=imgtk)

        cap.release()

    threading.Thread(target=run_camera, daemon=True).start()


def stop_camera():
    global camera_active
    camera_active = False
    # clear display after stopping camera
    display_frame.config(image='')
    result_label.config(text="Prediction: -")


# ---------------------------
# Buttons
# ---------------------------
live_btn = tk.Button(
    root,
    text="📷 Live Detection",
    font=("Helvetica", 14, "bold"),
    bg="#1E90FF",
    fg="white",
    activebackground="#4682B4",
    command=live_detection
)
live_btn.place(x=250, y=70, width=180, height=40)

browse_btn = tk.Button(
    root,
    text="🖼️Browse Image",
    font=("Helvetica", 14, "bold"),
    bg="#32CD32",
    fg="white",
    activebackground="#228B22",
    command=browse_image
)
browse_btn.place(x=470, y=70, width=185, height=40)

predict_btn = tk.Button(
    root,
    text="🔍 Predict",
    font=("Helvetica", 14, "bold"),
    bg="#FFA500",
    fg="white",
    activebackground="#FF8C00",
    command=predict_button_action
)

predict_btn.place(x=280, y=550, width=160, height=40)

stop_btn = tk.Button(
    root,
    text="⏹ Stop Camera",
    font=("Helvetica", 14, "bold"),
    bg="#FF4500",
    fg="white",
    activebackground="#B22222",
    command=stop_camera
)
stop_btn.place(x=470, y=550, width=160, height=40)


# ---------------------------
# On close
# ---------------------------


def on_close():
    stop_camera()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

# ---------------------------
# Start GUI
# ---------------------------
root.mainloop() 