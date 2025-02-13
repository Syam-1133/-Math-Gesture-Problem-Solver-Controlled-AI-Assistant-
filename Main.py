import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout='wide')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.button('Run')  # Correctly defining the button
    FRAME_WINDOW = st.image([])

with col2:
    st.title('Answer')  # Title for the output
    output_text_area = st.empty()  # Create an empty container to display AI result

# 🔹 Replace with your actual Gemini API key
GENAI_API_KEY = "AIzaSyClKmbmDmFw8jO_EKWOoZxJNh3EADj5-VE"

# 🔹 Configure the AI model
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# 🔹 Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1248)
cap.set(4, 720)

# 🔹 Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# 🔹 Initialize variables
pre_pos = None
canvas = None  # Drawing canvas
last_ai_request_time = 0  # ✅ For AI request rate limiting

def getHandInfo(img):
    """Detects hand and returns finger states + landmark positions."""
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]  # Get first detected hand
        lmList = hand["lmList"]  # Get 21 landmark points
        fingers = detector.fingersUp(hand)  # Get finger states
        print(f"🖐️ Fingers Detected: {fingers}")  # ✅ Debugging Output
        return fingers, lmList

    return None, None  # No hands detected


def draw(info, pre_pos, canvas):
    """Draws on the canvas using the index finger."""
    fingers, lmList = info

    if lmList is None:  # Ensure landmarks exist
        return pre_pos, canvas

    current_pos = tuple(map(int, lmList[8][:2]))  # Index finger tip (x, y)

    if fingers == [0, 1, 0, 0, 0]:  # 🔹 Index finger up (drawing mode)
        if pre_pos is not None:  # ✅ Only draw if pre_pos is valid
            cv2.line(canvas, pre_pos, current_pos, (255, 0, 255), 10)
        pre_pos = current_pos  # ✅ Update pre_pos for next frame

    else:
        pre_pos = None  # ✅ Reset pre_pos when not drawing

    if fingers == [0, 0, 0, 0, 1]:  # Five fingers up → Clear canvas
        canvas[:] = 0  # Reset canvas

    return pre_pos, canvas


def sendTOAI(fingers):
    """Sends a math query to Gemini AI when three fingers are up."""
    global last_ai_request_time

    if fingers == [1, 1, 1, 0, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["hey solve this math problem", pil_image])
        return response.text

output_text_area_value = ''  # Initialize as an empty string

# 🟢 Main loop
if run:  # Only run the process if the button is clicked
    while True:
        success, img = cap.read()  # Capture frame
        img = cv2.flip(img, flipCode=1)  # Flip for better usability

        if not success:
            print("🚨 Failed to capture image")
            continue

        # 🔹 Initialize canvas on first frame
        if canvas is None:
            canvas = np.zeros_like(img)  # 🔹 Black background

        fingers, lmList = getHandInfo(img)  # Get hand info

        if fingers is not None and lmList is not None:
            pre_pos, canvas = draw((fingers, lmList), pre_pos, canvas)  # ✅ Fix: Properly update pre_pos
            output_text_area_value = sendTOAI(fingers)  # ✅ AI request when triggered

        # 🔹 Merge canvas with webcam feed
        img_combined = cv2.addWeighted(img, 0.7, canvas, 0.15, 0)
        FRAME_WINDOW.image(img_combined, channels='BGR')

        # Display the AI result text in Streamlit with custom styles
        if output_text_area_value:
            # Using Markdown to style the output
            output_text_area.markdown(
                f"""
                <div style="background-color:#d3d3d3; padding: 20px; border-radius: 10px; font-size: 18px; color: #000080; font-weight: bold;">
                    {output_text_area_value}
                </div>
                """, unsafe_allow_html=True
            )

        # 🔹 Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()