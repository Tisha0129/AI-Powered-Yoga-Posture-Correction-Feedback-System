import cv2
import mediapipe as mp
import numpy as np
import joblib

# ✅ Load compressed model
model = joblib.load("yoga_model_2.pkl")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose()


# 🔥 Angle calculation
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# 🔥 UPDATED FEATURE EXTRACTION (VERY IMPORTANT)
def extract_features(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image)

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        coords = []
        for point in lm:
            coords.extend([point.x, point.y, point.z])

        coords = np.array(coords).reshape(33, 3)

        # ✅ NORMALIZATION
        center = coords[23]  # LEFT_HIP
        coords = coords - center

        max_value = np.max(np.abs(coords))
        if max_value != 0:
            coords = coords / max_value

        # ✅ ANGLES
        shoulder = coords[11][:2]
        elbow = coords[13][:2]
        wrist = coords[15][:2]

        hip = coords[23][:2]
        knee = coords[25][:2]
        ankle = coords[27][:2]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)

        # 🔥 NEW FEATURES

        # 1️⃣ Ankle distance
        left_ankle = coords[27][:2]
        right_ankle = coords[28][:2]
        ankle_distance = np.linalg.norm(
            np.array(left_ankle) - np.array(right_ankle)
        )

        # 2️⃣ Knee height difference
        left_knee_y = coords[25][1]
        right_knee_y = coords[26][1]
        knee_height_diff = abs(left_knee_y - right_knee_y)

        # 3️⃣ Hip alignment
        left_hip = coords[23][:2]
        right_hip = coords[24][:2]
        hip_distance = np.linalg.norm(
            np.array(left_hip) - np.array(right_hip)
        )

        # ✅ FINAL FEATURE VECTOR
        features = coords.flatten().tolist() + [
            elbow_angle,
            knee_angle,
            ankle_distance,
            knee_height_diff,
            hip_distance
        ]

        return features, results.pose_landmarks, elbow_angle

    return None, None, None


# Feedback logic
def get_feedback(prediction):

    prediction = prediction.lower()

    pose = prediction.split("_")[0]   # e.g. balasana
    quality = prediction.split("_")[-1]  # good / avg / poor

    feedback_map = {

        "balasana": {
            "poor": "Relax your hips back toward heels and lower your chest fully.",
            "avg": "Good, but try to extend your arms forward and relax deeper.",
            "good": "Perfect Balasana. Calm and well-aligned."
        },

        "bhujangasana": {
            "poor": "Lift your chest higher and keep elbows slightly bent.",
            "avg": "Good, open your chest more and pull shoulders back.",
            "good": "Excellent Bhujangasana. Strong spinal extension."
        },

        "padmasana": {
            "poor": "Adjust your legs and keep your spine straight.",
            "avg": "Good, but sit more upright and relax your shoulders.",
            "good": "Perfect Padmasana. Stable and relaxed posture."
        },

        "parvatasana": {
            "poor": "Lift your hips higher and straighten your back.",
            "avg": "Good, press your heels slightly towards the ground.",
            "good": "Great Parvatasana. Strong and stable alignment."
        },

        "tadasana": {
            "poor": "Stand straight and distribute weight evenly.",
            "avg": "Good, engage your core and relax shoulders.",
            "good": "Perfect Tadasana. Balanced and aligned."
        },

        "trikonasana": {
            "poor": "Extend your arms fully and avoid bending forward.",
            "avg": "Good, open your chest more and align your torso.",
            "good": "Excellent Trikonasana. Great body alignment."
        },

        "vrikshasan": {
            "poor": "Focus on balance and place your foot firmly.",
            "avg": "Good, stabilize your standing leg and align hips.",
            "good": "Perfect Vrikshasana. Strong balance and posture."
        }
    }

    # Default fallback (production safety)
    if pose in feedback_map and quality in feedback_map[pose]:
        return feedback_map[pose][quality]
    else:
        return "Adjust your posture and try again."


# Start webcam
cap = cv2.VideoCapture(0)

# Fullscreen window
cv2.namedWindow("Yoga Posture Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Yoga Posture Detection",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    features, pose_landmarks, elbow_angle = extract_features(frame)

    if features:

        # 🔥 Prediction using NEW features
        prediction = model.predict([features])[0]

        feedback = get_feedback(prediction)

        # Display prediction
        cv2.putText(frame, prediction, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Display feedback
        cv2.putText(frame, feedback, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display angle
        cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Yoga Posture Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()