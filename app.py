import math
import os
import webbrowser
from pathlib import Path
import time

import streamlit as st
import click
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

# Multilingual support dictionary
LANGUAGES = {
    "English": {
        "title": "Real/Fake Face Detection",
        "description": "The application will detect faces for 10 seconds and provide a final result.",
        "start_button": "Start Detection",
        "time_left": "Time left: {:.1f} seconds",
        "current_counts": "Current Counts - Real: {}, Fake: {}",
        "result_header": "Detection Result",
        "real_success": "Real Person Detected (Real: {}, Fake: {})",
        "fake_error": "Fake/Spoofed Person Detected (Real: {}, Fake: {})",
        "inconclusive": "Inconclusive Result (Real: {}, Fake: {})"
    },
    "Hindi": {
        "title": "वास्तविक/नकली चेहरा पहचान",
        "description": "एप्लिकेशन 10 सेकंड तक चेहरे का पता लगाएगा और अंतिम परिणाम प्रदान करेगा।",
        "start_button": "पहचान शुरू करें",
        "time_left": "बचा हुआ समय: {:.1f} सेकंड",
        "current_counts": "वर्तमान गणना - वास्तविक: {}, नकली: {}",
        "result_header": "पहचान का परिणाम",
        "real_success": "वास्तविक व्यक्ति पहचाना गया (वास्तविक: {}, नकली: {})",
        "fake_error": "नकली/स्पूफ्ड व्यक्ति पहचाना गया (वास्तविक: {}, नकली: {})",
        "inconclusive": "अनिश्चित परिणाम (वास्तविक: {}, नकली: {})"
    },
    "Kannada": {
        "title": "ನಿಜ/ಕೃತಕ ಮುಖ ಗುರುತಿಸುವಿಕೆ",
        "description": "ಅನ್ವಯವು 10 ಸೆಕೆಂಡುಗಳ ಕಾಲ ಮುಖಗಳನ್ನು ಗುರುತಿಸಿ ಅಂತಿಮ ಫಲಿತಾಂಶವನ್ನು ಒದಗಿಸುತ್ತದೆ.",
        "start_button": "ಗುರುತಿಸುವಿಕೆ ಆರಂಭಿಸಿ",
        "time_left": "ಉಳಿದ ಸಮಯ: {:.1f} ಸೆಕೆಂಡುಗಳು",
        "current_counts": "ಪ್ರಸಕ್ತ ಎಣಿಕೆ - ನಿಜ: {}, ಕೃತಕ: {}",
        "result_header": "ಗುರುತಿಸುವಿಕೆ ಫಲಿತಾಂಶ",
        "real_success": "ನಿಜ ವ್ಯಕ್ತಿಯನ್ನು ಗುರುತಿಸಲಾಗಿದೆ (ನಿಜ: {}, ಕೃತಕ: {})",
        "fake_error": "ಕೃತಕ/ಸ್ಪೂಫ್ಡ್ ವ್ಯಕ್ತಿಯನ್ನು ಗುರುತಿಸಲಾಗಿದೆ (ನಿಜ: {}, ಕೃತಕ: {})",
        "inconclusive": "ಅನಿಶ್ಚಿತ ಫಲಿತಾಂಶ (ನಿಜ: {}, ಕೃತಕ: {})"
    }
}

def detect_faces_with_timer(lang_texts):
    # Setup paths and configurations
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    path_to_weights = Path.joinpath(Path.joinpath(dir_path, "saved_weights"), "best.pt")

    # Model and detection parameters
    confidence_threshold = 0.8
    frame_width = 640
    frame_height = 480
    class_names = ["fake", "real"]

    # Initialize model
    model = YOLO(path_to_weights)

    # Streamlit video capture
    capture = cv2.VideoCapture(0)
    capture.set(3, frame_width)
    capture.set(4, frame_height)

    # Prediction tracking
    predictions = {
        "fake": 0,
        "real": 0
    }

    # Timing and results display
    start_time = time.time()
    detection_duration = 10  # seconds

    # Placeholder for Streamlit image display
    image_placeholder = st.empty()
    status_text = st.empty()

    while (time.time() - start_time) < detection_duration:
        success, image = capture.read()
        if not success:
            st.error("Failed to capture frame")
            break

        results = model(image, stream=True, verbose=False)

        # Track the most confident detection
        most_confident_detection = None
        max_confidence = 0

        for result in results:
            boxes = result.boxes

            for box in boxes:
                current_confidence = box.conf[0]
                
                # Find the most confident detection
                if current_confidence > confidence_threshold and current_confidence > max_confidence:
                    most_confident_detection = box
                    max_confidence = current_confidence

        # Process only the most confident detection
        if most_confident_detection is not None:
            x1, y1, x2, y2 = most_confident_detection.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            current_class = int(most_confident_detection.cls[0])

            # Update prediction counts
            predictions[class_names[current_class]] += 1

            # Choose color based on class
            color = (0, 255, 0) if class_names[current_class] == "real" else (0, 0, 255)

            # Draw bounding box
            cvzone.cornerRect(image, (x1, y1, w, h), colorC=color, colorR=color)
            
            # Put text with confidence
            cvzone.putTextRect(image, 
                f"{class_names[current_class].upper()} {round(max_confidence.item(), 2) * 100}%",
                (max(0, x1), max(35, y1)), 
                scale=2, 
                thickness=4,
                colorR=color, 
                colorB=color
            )

        # Convert OpenCV image to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display image and time left
        time_left = max(0, detection_duration - (time.time() - start_time))
        image_placeholder.image(image_rgb, channels="RGB", caption=lang_texts['time_left'].format(time_left))
        
        # Optional: show current counts
        status_text.text(lang_texts['current_counts'].format(predictions['real'], predictions['fake']))

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    capture.release()

    # Determine final result
    st.subheader(lang_texts['result_header'])
    if predictions['real'] > predictions['fake']:
        st.success(lang_texts['real_success'].format(predictions['real'], predictions['fake']))
        
        # IMPORTANT: Replace this URL with the website you want to redirect to
        # redirect_url = "https://face-api-tanishqthuses-projects.vercel.app/"
        redirect_url = "https://complete-active-liveness.vercel.app/"
        
        # Open the website in the default web browser
        webbrowser.open(redirect_url)
    elif predictions['fake'] > predictions['real']:
        st.error(lang_texts['fake_error'].format(predictions['real'], predictions['fake']))
    else:
        st.warning(lang_texts['inconclusive'].format(predictions['real'], predictions['fake']))

def main():
    # Add language selection to the sidebar
    # Add this line at the beginning of the main function
    # st.sidebar.image("C:\PSG\VIT\sih\SOME_CHANGES\SIH_logo_2024_horizontal.png", use_column_width=True)

    # Place the logo in the same directory as your app.py file
    st.sidebar.image("SIH_logo_2024_horizontal.png", use_column_width=True)

    st.sidebar.header("Language / भाषा / ಭಾಷೆ")
    selected_language = st.sidebar.selectbox(
        "Choose Language",
        list(LANGUAGES.keys()),
        index=0
    )

    # Get language-specific texts
    lang_texts = LANGUAGES[selected_language]

    # Set the title and description based on selected language
    st.title(lang_texts['title'])
    st.write(lang_texts['description'])
    
    # Style the start button
    start_button_style = """
    <style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """
    st.markdown(start_button_style, unsafe_allow_html=True)
    
    # Create start button with language-specific text
    if st.button(lang_texts['start_button']):
        detect_faces_with_timer(lang_texts)

if __name__ == "__main__":
    main()

# CORRECT WORKING CODE WITH SOME CHANGES BY TANISHQ
# import math
# import os
# import webbrowser
# from pathlib import Path
# import time

# import streamlit as st
# import click
# import cv2
# import cvzone
# import numpy as np
# from ultralytics import YOLO

# def detect_faces_with_timer():
#     # Setup paths and configurations
#     dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
#     path_to_weights = Path.joinpath(Path.joinpath(dir_path, "saved_weights"), "best.pt")

#     # Model and detection parameters
#     confidence_threshold = 0.8
#     frame_width = 640
#     frame_height = 480
#     class_names = ["fake", "real"]

#     # Initialize model
#     model = YOLO(path_to_weights)

#     # Streamlit video capture
#     capture = cv2.VideoCapture(0)
#     capture.set(3, frame_width)
#     capture.set(4, frame_height)

#     # Prediction tracking
#     predictions = {
#         "fake": 0,
#         "real": 0
#     }

#     # Timing and results display
#     start_time = time.time()
#     detection_duration = 10  # seconds

#     # Placeholder for Streamlit image display
#     image_placeholder = st.empty()
#     status_text = st.empty()

#     while (time.time() - start_time) < detection_duration:
#         success, image = capture.read()
#         if not success:
#             st.error("Failed to capture frame")
#             break

#         results = model(image, stream=True, verbose=False)

#         # Track the most confident detection
#         most_confident_detection = None
#         max_confidence = 0

#         for result in results:
#             boxes = result.boxes

#             for box in boxes:
#                 current_confidence = box.conf[0]
                
#                 # Find the most confident detection
#                 if current_confidence > confidence_threshold and current_confidence > max_confidence:
#                     most_confident_detection = box
#                     max_confidence = current_confidence

#         # Process only the most confident detection
#         if most_confident_detection is not None:
#             x1, y1, x2, y2 = most_confident_detection.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#             w, h = x2 - x1, y2 - y1

#             current_class = int(most_confident_detection.cls[0])

#             # Update prediction counts
#             predictions[class_names[current_class]] += 1

#             # Choose color based on class
#             color = (0, 255, 0) if class_names[current_class] == "real" else (0, 0, 255)

#             # Draw bounding box
#             cvzone.cornerRect(image, (x1, y1, w, h), colorC=color, colorR=color)
            
#             # Put text with confidence
#             cvzone.putTextRect(image, 
#                 f"{class_names[current_class].upper()} {round(max_confidence.item(), 2) * 100}%",
#                 (max(0, x1), max(35, y1)), 
#                 scale=2, 
#                 thickness=4,
#                 colorR=color, 
#                 colorB=color
#             )

#         # Convert OpenCV image to RGB for Streamlit display
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Display image and time left
#         time_left = max(0, detection_duration - (time.time() - start_time))
#         image_placeholder.image(image_rgb, channels="RGB", caption=f"Time left: {time_left:.1f} seconds")
        
#         # Optional: show current counts
#         status_text.text(f"Current Counts - Real: {predictions['real']}, Fake: {predictions['fake']}")

#         # Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture
#     capture.release()

#     # Determine final result
#     st.subheader("Detection Result")
#     if predictions['real'] > predictions['fake']:
#         st.success(f"Real Person Detected (Real: {predictions['real']}, Fake: {predictions['fake']})")
        
#         # IMPORTANT: Replace this URL with the website you want to redirect to
#         redirect_url = "https://face-api-tanishqthuses-projects.vercel.app/"
        
#         # Open the website in the default web browser
#         # st.write(f"Redirecting to: {redirect_url}")
#         webbrowser.open(redirect_url)
#     elif predictions['fake'] > predictions['real']:
#         st.error(f"Fake/Spoofed Person Detected (Real: {predictions['real']}, Fake: {predictions['fake']})")
#     else:
#         st.warning(f"Inconclusive Result (Real: {predictions['real']}, Fake: {predictions['fake']})")

# def main():
#     st.title("Real/Fake Face Detection")
#     st.write("The application will detect faces for 10 seconds and provide a final result.")
    
#     if st.button("Start Detection"):
#         detect_faces_with_timer()

# if __name__ == "__main__":
#     main()



# CORRECT OLD CODE BY PSG
# import math
# import os
# import webbrowser
# from pathlib import Path
# import time

# import streamlit as st
# import click
# import cv2
# import cvzone
# import numpy as np
# from ultralytics import YOLO

# def detect_faces_with_timer():
#     # Setup paths and configurations
#     dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
#     path_to_weights = Path.joinpath(Path.joinpath(dir_path, "saved_weights"), "best.pt")

#     # Model and detection parameters
#     confidence_threshold = 0.8
#     frame_width = 640
#     frame_height = 480
#     class_names = ["fake", "real"]

#     # Initialize model
#     model = YOLO(path_to_weights)

#     # Streamlit video capture
#     capture = cv2.VideoCapture(0)
#     capture.set(3, frame_width)
#     capture.set(4, frame_height)

#     # Prediction tracking
#     predictions = {
#         "fake": 0,
#         "real": 0
#     }

#     # Timing and results display
#     start_time = time.time()
#     detection_duration = 10  # seconds

#     # Placeholder for Streamlit image display
#     image_placeholder = st.empty()
#     status_text = st.empty()

#     while (time.time() - start_time) < detection_duration:
#         success, image = capture.read()
#         if not success:
#             st.error("Failed to capture frame")
#             break

#         results = model(image, stream=True, verbose=False)

#         for result in results:
#             boxes = result.boxes

#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 w, h = x2 - x1, y2 - y1

#                 current_confidence = box.conf[0]
#                 current_class = int(box.cls[0])

#                 if current_confidence > confidence_threshold:
#                     # Update prediction counts
#                     predictions[class_names[current_class]] += 1

#                     # Choose color based on class
#                     color = (0, 255, 0) if class_names[current_class] == "real" else (0, 0, 255)

#                     # Draw bounding box
#                     cvzone.cornerRect(image, (x1, y1, w, h), colorC=color, colorR=color)
                    
#                     # Put text with confidence
#                     cvzone.putTextRect(image, 
#                         f"{class_names[current_class].upper()} {round(current_confidence.item(), 2) * 100}%",
#                         (max(0, x1), max(35, y1)), 
#                         scale=2, 
#                         thickness=4,
#                         colorR=color, 
#                         colorB=color
#                     )

#         # Convert OpenCV image to RGB for Streamlit display
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Display image and time left
#         time_left = max(0, detection_duration - (time.time() - start_time))
#         image_placeholder.image(image_rgb, channels="RGB", caption=f"Time left: {time_left:.1f} seconds")
        
#         # Optional: show current counts
#         status_text.text(f"Current Counts - Real: {predictions['real']}, Fake: {predictions['fake']}")

#         # Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture
#     capture.release()

#     # Determine final result
#     st.subheader("Detection Result")
#     if predictions['real'] > predictions['fake']:
#         st.success(f"Real Person Detected (Real: {predictions['real']}, Fake: {predictions['fake']})")
        
#         # IMPORTANT: Replace this URL with the website you want to redirect to
#         redirect_url = "https://face-api-tanishqthuses-projects.vercel.app/"
        
#         # Open the website in the default web browser
#         # st.write(f"Redirecting to: {redirect_url}")
#         webbrowser.open(redirect_url)
#     elif predictions['fake'] > predictions['real']:
#         st.error(f"Fake/Spoofed Person Detected (Real: {predictions['real']}, Fake: {predictions['fake']})")
#     else:
#         st.warning(f"Inconclusive Result (Real: {predictions['real']}, Fake: {predictions['fake']})")

# def main():
#     st.title("Real/Fake Face Detection")
#     st.write("The application will detect faces for 10 seconds and provide a final result.")
    
#     if st.button("Start Detection"):
#         detect_faces_with_timer()

# if __name__ == "__main__":
#     main()