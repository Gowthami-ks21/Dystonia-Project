---------------Dystonia Detection using Spiral Handwriting Samples------------------------


#Required Python Libraries
Install the required dependencies using the following command:
pip install opencv-python torch torchvision ultralytics numpy
pillow pyttsx3
Ensure pip is updated before installation:
python -m pip install --upgrade pip

Steps to Run the Project
Step 1: Clone or Extract the Project

Download and extract the project folder to your local system
Step 2: Install Dependencies
Navigate to the project directory and install all required packages:
pip install -r requirements.txt
Step 3: Verify YOLO Model Weights
Ensure the trained YOLO model file (best.pt) is placed inside the weights/ folder.
Step 4: Run the Application
 Open VS Code.
 Open the main file – main.py
 Click the run python file button

Using the Application
Upload Mode
1. Click Browse Image
2. Select a spiral handwriting image
3. Click Predict
4. View result and listen to voice output
Live Detection Mode
1. Click Live Detection
2. Show handwritten spiral to the webcam
3. Click Predict
4. Click Stop Camera to end live mode
Output Details
 Displays predicted label (Dystonia / Non-Dystonia)

 Shows confidence score
 Draws bounding box on detected region
 Announces result using voice feedback
