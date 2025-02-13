# # 🤖 Math Gesture Problem Solver Controlled AI Assistant ✋

## 📜 Description
The **Math Gesture Problem Solver Controlled AI Assistant** is an innovative project that allows users to interact with an AI to solve math problems using hand gestures. This project integrates **computer vision** for gesture recognition and a **generative AI model** for solving math problems.

The user can **draw math problems** on a canvas using their fingers, and with a specific hand gesture, the system will send the drawn problem to an **AI-powered math solver** and return the solution in real time.

### 🔑 Key Features:
- **✍️ Gesture-based Drawing**: Use your **index finger** to draw math problems like equations and expressions.
- **🧹 Clear Canvas**: Raise all **five fingers** to clear the canvas and start over.
- **🧠 AI Math Solver**: The system uses the **Gemini AI model** to solve the math problems that are drawn on the canvas.
- **👁️‍🗨️ Real-Time Interaction**: Uses **webcam input** to track hand gestures and provide immediate feedback.
- **🌐 Streamlit UI**: A simple **Streamlit web interface** makes the application user-friendly and interactive.


## 🛠️ How the Project Works

### 1. **Hand Gesture Detection**:
   The project uses the **cvzone** and **OpenCV** libraries to track and detect **hand gestures** in real time. A **HandDetector** from the `cvzone` library is used to detect hand landmarks and identify the gestures. These gestures include:
   - **One finger up (index finger)**: Used for **drawing**.
   - **Five fingers up**: Used to **clear** the canvas.
   - **Three fingers up**: Used to **trigger** the AI for solving the math problem.

   The **fingersUp** method checks the state of each finger and provides the necessary input for the drawing and interaction.

### 2. **Canvas Drawing**:
   - The user can draw **math problems** (like equations) on a **canvas** using the index finger. This drawing is done by tracking the position of the index finger in the video feed and drawing lines on a black canvas.
   - If the user raises **five fingers**, the canvas is cleared, allowing them to draw a new problem.

### 3. **Sending Math Problem to AI**:
   - Once the math problem is drawn on the canvas, the user can raise **three fingers**, which triggers a request to the **Gemini AI model** (a powerful generative AI).
   - The **mathematical expression** drawn on the canvas is converted into an **image** (using the **PIL** library) and sent to the AI model via an API call.
   - The AI processes the image, recognizes the math problem, and generates a solution, which is then displayed on the **Streamlit interface**.

### 4. **Real-Time Feedback**:
   - The webcam feed and canvas are merged to create an interactive experience. The **real-time webcam feed** is displayed to the user, with the canvas drawn on top. The **AI response** (the solution to the math problem) is shown beside the webcam feed.
   - This provides instant feedback to the user, making the interaction seamless and intuitive.

### 5. **Streamlit Interface**:
   The entire application runs on **Streamlit**, a powerful framework for creating interactive web apps. The **Streamlit** UI allows the user to interact with the webcam feed, see their gestures, and view the solution to their math problems without needing to refresh the page.


## 🛠️ Tools & Technologies Used
This project leverages several powerful libraries and tools for hand gesture recognition, AI interaction, and web-based user interfaces:

- **OpenCV**: Used for webcam feed capture, image processing, and gesture detection.
- **cvzone**: Provides the **HandDetector** module to track hand landmarks and detect gestures.
- **Google Gemini AI**: A generative AI model used to process math problems and provide solutions.
- **Pillow (PIL)**: Used to convert the canvas into an image format that can be sent to the AI model.
- **Streamlit**: A Python framework for creating interactive web applications, which is used to display the webcam feed, canvas, and AI results in real-time.
- **NumPy**: Used for matrix operations and handling images for drawing on the canvas.


## 📜 Installation

### Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- A working webcam
- **Gemini AI API Key** (replace the placeholder in the code with your actual API key)

### Steps to Set Up the Project

1. **Clone the Repository**:
   Clone this repository to your local machine using Git:
   ```bash
   https://github.com/Syam-1133/-Math-Gesture-Problem-Solver-Controlled-AI-Assistant-
   
GENAI_API_KEY = "enter your'e API key"

```bash

streamlit run Main.py


