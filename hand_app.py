import torch
import torch.nn as nn
import mediapipe as mp
import cv2
import numpy as np

# ==========================================
# 1. تعريف نموذج PyTorch (PoseLifter3D)
# ==========================================
class LinearBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        return self.block(x)

class PoseLifter3D(nn.Module):
    def __init__(self, input_size=42, output_size=63, linear_size=1024, num_stages=2, p_dropout=0.5):
        super(PoseLifter3D, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout)
        )
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                LinearBlock(linear_size, p_dropout),
                LinearBlock(linear_size, p_dropout)
            ) for _ in range(num_stages)
        ])
        
        self.output_layer = nn.Linear(linear_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for stage in self.stages:
            x = x + stage(x)
        out = self.output_layer(x)
        return out.view(-1, 21, 3)

# ==========================================
# 2. تصميم الواجهة الاحترافية (HUD)
# ==========================================
def create_hud_overlay(image, text_dict, alpha=0.6):
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 20), (350, 180), (15, 15, 15), -1)
    cv2.rectangle(overlay, (20, 20), (350, 180), (255, 255, 0), 2)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    y_offset = 60
    cv2.putText(image, "3D Gesture UI System", (35, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(image, (35, y_offset + 10), (330, y_offset + 10), (255, 255, 0), 1)
    
    y_offset += 35
    for key, value in text_dict.items():
        text = f"{key}: {value}"
        cv2.putText(image, text, (35, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += 30

# ==========================================
# 3. التشغيل واستخراج النقاط
# ==========================================
def run_gesture_ui_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PoseLifter3D().to(device)
    model.eval()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    custom_hand_style = mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=4)
    custom_connection_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            hud_data = {
                "Status": "Searching for Hand...",
                "Index Z-Depth": "N/A",
                "MPJPE Target": "< 15 mm"
            }

            if results.multi_hand_landmarks:
                hud_data["Status"] = "Hand Detected (Tracking)"
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints_2d = []
                    for lm in hand_landmarks.landmark:
                        keypoints_2d.extend([lm.x, lm.y])
                    
                    input_2d = torch.tensor(keypoints_2d, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_3d = model(input_2d) 
                    
                    index_fingertip_3d = pred_3d[0][8].cpu().numpy()
                    hud_data["Index Z-Depth"] = f"{index_fingertip_3d[2]:.4f} mm"
                    
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        custom_hand_style,
                        custom_connection_style)
                    
                    h, w, _ = image.shape
                    cx, cy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                    cv2.circle(image, (cx, cy), 12, (0, 255, 0), -1)
                    cv2.circle(image, (cx, cy), 18, (0, 255, 0), 2)

            create_hud_overlay(image, hud_data)
            cv2.imshow('Gesture UI - 3D Hand Pose Estimation', image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'): 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("جاري تشغيل النظام... اضغط على زر 'q' أو 'ESC' للإغلاق.")
    run_gesture_ui_inference()