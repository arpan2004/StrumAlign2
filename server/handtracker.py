import cv2
import mediapipe as mp
from flask import Flask, Response, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB (MediaPipe uses RGB images)
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Process the frame to detect hands
        result = mp_hands.process(RGB_frame)

        # If hand landmarks are detected, draw them
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                if handedness.classification[0].label == 'Right':
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    finger_curvatures = analyze_curves(hand_landmarks)

                    for finger, angles in finger_curvatures.items():
                        if angles['Wrist-MCP'] < 170.75 and angles['MCP-PIP'] < 152.50 and angles['PIP-DIP'] < 162.25:
                            curvature_feedback = f"{finger.capitalize()} Curved Enough!"
                            color = (0, 255, 0)
                            bg_color = (0, 128, 0)
                            
                            # cv2.putText(frame, f"{finger.capitalize()} Curved Enough!", 
                            #             (50, 50 + 30 * list(finger_curvatures.keys()).index(finger)), 
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            curvature_feedback = f"{finger.capitalize()} Not Curved Enough!"
                            color = (0, 0, 255)
                            bg_color = (128, 0, 0)

                            # cv2.putText(frame, f"{finger.capitalize()} Not Curved Enough!", 
                            #             (50, 50 + 30 * list(finger_curvatures.keys()).index(finger)), 
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                        text_size, _ = cv2.getTextSize(curvature_feedback, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_width, text_height = text_size
                        text_x = 50
                        text_y = 50 + 30 * list(finger_curvatures.keys()).index(finger)

                        cv2.rectangle(frame, (text_x - 3, text_y - text_height - 3), 
                                    (text_x + text_width + 3, text_y + 3), bg_color, -1)
                        
                        cv2.putText(frame, curvature_feedback,
                                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def angle_between_points(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    return np.degrees(angle)

def analyze_finger_curvature(landmarks, finger_indices):
    wrist = landmarks[0] #wrist
    mcp = landmarks[finger_indices[0]]  # MCP joint
    pip = landmarks[finger_indices[1]]  # PIP joint
    dip = landmarks[finger_indices[2]]  # DIP joint
    tip = landmarks[finger_indices[3]]  # Fingertip

    # Calculate angles between joints
    angle_wrist_mcp = angle_between_points(wrist, mcp, pip)
    angle_mcp_pip = angle_between_points(mcp, pip, dip)
    angle_pip_dip = angle_between_points(pip, dip, tip)

    return angle_wrist_mcp, angle_mcp_pip, angle_pip_dip

def analyze_curves(hand_landmarks):
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    fingers = {
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }

    finger_curvatures = {}

    for finger, indices in fingers.items():
        angle_wrist_mcp, angle_mcp_pip, angle_pip_dip = analyze_finger_curvature(landmarks, indices)
        finger_curvatures[finger] = {
            "Wrist-MCP": angle_wrist_mcp,
            "MCP-PIP": angle_mcp_pip,
            "PIP-DIP": angle_pip_dip,
        }

    print(finger_curvatures)
    return finger_curvatures


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)