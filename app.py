"""
🎭 Gradio Live Face Detection & Age-Gender Analysis App

Pipeline: Camera/Video → YOLO (Face Detection) → ConvNeXt + CORAL (Age-Gender Prediction)

Model: NestedAgeModelCORAL (CORAL Ordinal Regression)
"""

import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import timm
import torch.nn as nn
from torchvision import transforms

# ============================================================
# 1. CONSTANTS
# ============================================================

AGE_MIN, AGE_MAX = 1, 90

# CORAL için yaş sınırları (thresholds)
# Her threshold: "Yaş >= threshold mı?"
# 2'şerli artışlarla: 2, 4, 6, 8, ..., 88, 90
AGE_THRESHOLDS = list(range(2, 91, 2))  # [2, 4, 6, 8, ..., 88, 90] = 45 threshold
NUM_THRESHOLDS = len(AGE_THRESHOLDS)

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

class NestedAgeModelCORAL(nn.Module):
    """
    Nested Multi-task Model with CORAL Ordinal Regression
    
    CORAL Mantığı:
    - K threshold için K binary classifier
    - Hepsi aynı feature'dan tek bir logit üretir
    - Her threshold için farklı bias: logit - bias_k
    - Consistent rank: P(age > t_k) > P(age > t_{k+1})
    
    Nested: Gender → Race → Age
    """
    
    def __init__(self, backbone='convnext_tiny', num_race=5, num_thresholds=45, pretrained=False):
        super().__init__()
        self.num_thresholds = num_thresholds
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        
        # Gender Head
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Race Head (gender conditioned)
        self.race_head = nn.Sequential(
            nn.Linear(feat_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_race)
        )
        
        # CORAL Age Head
        # Tek bir lineer katman (bias yok) → tüm thresholds için ortak logit
        self.coral_fc = nn.Linear(feat_dim + 2 + num_race, 1, bias=False)
        
        # Her threshold için ayrı bias (öğrenilebilir)
        # Sıralı olması için başlangıçta artan değerler
        initial_bias = torch.linspace(-3, 3, num_thresholds)
        self.coral_bias = nn.Parameter(initial_bias)
        
    def forward(self, x):
        # Backbone features
        feat = self.backbone(x)
        
        # Gender prediction
        gender_logits = self.gender_head(feat)
        gender_probs = torch.softmax(gender_logits, dim=1)
        
        # Race prediction (conditioned on gender)
        race_in = torch.cat([feat, gender_probs], dim=1)
        race_logits = self.race_head(race_in)
        race_probs = torch.softmax(race_logits, dim=1)
        
        # CORAL Age prediction
        age_in = torch.cat([feat, gender_probs, race_probs], dim=1)
        
        # Tek logit → tüm threshold'lar için
        base_logit = self.coral_fc(age_in)  # [B, 1]
        
        # Her threshold için: logit_k = base_logit - bias_k
        # coral_logits[k] = base_logit - bias[k]
        coral_logits = base_logit - self.coral_bias  # [B, num_thresholds]
        
        return gender_logits, race_logits, coral_logits
    
    def predict_age(self, coral_logits):
        """
        CORAL logits → Yaş tahmini
        
        Her threshold için P(age > t_k) = sigmoid(logit_k)
        Beklenen rank = sum(P(age > t_k))
        Yaş = interpolate(rank, thresholds)
        """
        probs = torch.sigmoid(coral_logits)  # [B, K]
        
        # Beklenen rank (kaç threshold geçildi)
        expected_rank = probs.sum(dim=1)  # [B]
        
        # Rank → Yaş dönüşümü (lineer interpolasyon)
        # rank=0 → AGE_MIN, rank=K → AGE_MAX
        age_pred = AGE_MIN + (expected_rank / self.num_thresholds) * (AGE_MAX - AGE_MIN)
        
        return age_pred


# ============================================================
# 3. GLOBAL MODEL LOADING
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {DEVICE}")

# Load YOLO Face Detector
try:
    face_detector = YOLO("face_yolo_best.pt")
    print("✅ YOLO Face Detector loaded")
except Exception as e:
    print(f"⚠️ YOLO model not found, using default: {e}")
    face_detector = YOLO("yolov8n.pt")

# Load Age-Gender Model (CORAL Model)
age_gender_model = NestedAgeModelCORAL(
    backbone="convnext_tiny",
    num_race=5,
    num_thresholds=NUM_THRESHOLDS,
    pretrained=False
).to(DEVICE)

try:
    # CORAL model dosyasını yükle (önce yeni notebook'tan gelen)
    state_dict = torch.load("best_age_model_coral.pth", map_location=DEVICE)
    age_gender_model.load_state_dict(state_dict, strict=True)
    print("✅ Age-Gender CORAL Model loaded (best_age_model_coral.pth)")
except FileNotFoundError:
    try:
        # Alternatif dosya adları
        state_dict = torch.load("best_coral_model.pth", map_location=DEVICE)
        age_gender_model.load_state_dict(state_dict, strict=False)
        print("✅ Age-Gender CORAL Model loaded (best_coral_model.pth)")
    except Exception as e:
        print(f"⚠️ Age-Gender model not found: {e}")
        print("   Model will use random weights")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    print("   Model will use random weights")

age_gender_model.eval()


# ============================================================
# 4. PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@torch.no_grad()
def predict_face_attributes(face_bgr):
    """Predict age and gender from face crop using CORAL model."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    x = preprocess(face_rgb).unsqueeze(0).to(DEVICE)
    
    # CORAL model çıktısı: gender_logits, race_logits, coral_logits
    gender_logits, race_logits, coral_logits = age_gender_model(x)
    
    # Gender
    gender_id = gender_logits.argmax(1).item()
    gender = "Male" if gender_id == 0 else "Female"
    gender_conf = torch.softmax(gender_logits, dim=1).max().item()
    
    # Age (CORAL prediction)
    age_years = age_gender_model.predict_age(coral_logits).item()
    
    # Race (opsiyonel, gösterim için)
    race_id = race_logits.argmax(1).item()
    race_names = ['White', 'Black', 'Asian', 'Indian', 'Other']
    race = race_names[race_id] if race_id < len(race_names) else 'Unknown'
    
    return {
        "gender": gender,
        "gender_conf": gender_conf,
        "age": age_years,
        "race": race
    }


def draw_results(img_rgb, boxes, predictions):
    """Draw bounding boxes and labels on image."""
    img_draw = img_rgb.copy()
    
    for box, pred in zip(boxes, predictions):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Color based on gender (Blue=Male, Red=Female)
        color = (66, 133, 244) if pred["gender"] == "Male" else (219, 68, 55)
        
        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = f"{pred['gender']} | {pred['age']:.0f}y"
        
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_draw, (x1, y1 - h - 12), (x1 + w + 8, y1), color, -1)
        
        # Label text
        cv2.putText(img_draw, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_draw


# ============================================================
# 4. MAIN PROCESSING FUNCTION
# ============================================================

def process_frame(frame, conf_threshold=0.4):
    """
    Process a single frame: detect faces and predict attributes.
    
    Args:
        frame: RGB image from Gradio (numpy array)
        conf_threshold: YOLO confidence threshold
    
    Returns:
        Annotated image, detection info string
    """
    if frame is None:
        return None, "❌ No input received"
    
    # Convert RGB to BGR for OpenCV/YOLO
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Face detection with YOLO
    results = face_detector(img_bgr, conf=conf_threshold, verbose=False)[0]
    
    if results.boxes is None or len(results.boxes) == 0:
        return frame, "🔍 No faces detected"
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    boxes_with_conf = np.column_stack([boxes, confs])
    
    # Process each face
    predictions = []
    info_lines = []
    valid_boxes = []
    
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure valid crop
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
        
        # Crop face
        face_crop = img_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue
        
        # Predict attributes
        try:
            pred = predict_face_attributes(face_crop)
            predictions.append(pred)
            valid_boxes.append([x1, y1, x2, y2, conf])
            
            # Info string
            info_lines.append(
                f"👤 Face {i+1}: {pred['gender']} ({pred['gender_conf']:.0%}) | "
                f"Age: {pred['age']:.1f}"
            )
        except Exception:
            info_lines.append(f"⚠️ Face {i+1}: Error processing")
    
    # Draw results
    if predictions:
        img_result = draw_results(frame, np.array(valid_boxes), predictions)
    else:
        img_result = frame
    
    # Summary
    info_text = f"🎯 Detected {len(predictions)} face(s)\n\n" + "\n".join(info_lines)
    
    return img_result, info_text


def process_video(video_path, conf_threshold=0.4):
    """Process uploaded video file."""
    if video_path is None:
        return None, "❌ No video uploaded"
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_faces = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_frame, info = process_frame(frame_rgb, conf_threshold)
        
        if result_frame is not None:
            result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            out.write(result_bgr)
        
        frame_count += 1
        if "Detected" in info:
            try:
                n = int(info.split("Detected")[1].split("face")[0].strip())
                total_faces += n
            except:
                pass
    
    cap.release()
    out.release()
    
    info = f"✅ Processed {frame_count} frames\n📊 Average faces/frame: {total_faces/max(1,frame_count):.1f}"
    
    return output_path, info


# ============================================================
# 5. GRADIO INTERFACE
# ============================================================

def create_app():
    """Create Gradio application."""
    
    with gr.Blocks(
        title="🎭 Face Detection & Age-Gender Analysis"
    ) as app:
        
        gr.Markdown("""
        # 🎭 Face Detection & Age-Gender Analysis
        
        **Pipeline:** Input → YOLO (Face Detection) → ConvNeXt (Age-Gender Prediction)
        
        | Input Type | Description |
        |------------|-------------|
        | 📷 Webcam | Real-time live detection |
        | 🖼️ Image | Upload and analyze single image |
        | 🎬 Video | Process uploaded video file |
        """)
        
        with gr.Tabs():
            # ============== TAB 1: WEBCAM ==============
            with gr.TabItem("📷 Live Webcam"):
                with gr.Row():
                    with gr.Column():
                        webcam_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                            label="🎚️ Detection Confidence"
                        )
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Webcam Feed"
                        )
                    
                    with gr.Column():
                        webcam_output = gr.Image(label="🖼️ Result")
                        webcam_info = gr.Textbox(label="📊 Info", lines=5)
                
                webcam_input.stream(
                    fn=process_frame,
                    inputs=[webcam_input, webcam_conf],
                    outputs=[webcam_output, webcam_info]
                )
            
            # ============== TAB 2: IMAGE ==============
            with gr.TabItem("🖼️ Image Upload"):
                with gr.Row():
                    with gr.Column():
                        image_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                            label="🎚️ Detection Confidence"
                        )
                        image_input = gr.Image(
                            sources=["upload"],
                            label="Upload Image"
                        )
                        image_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(label="🖼️ Result")
                        image_info = gr.Textbox(label="📊 Info", lines=5)
                
                image_btn.click(
                    fn=process_frame,
                    inputs=[image_input, image_conf],
                    outputs=[image_output, image_info]
                )
                
                image_input.change(
                    fn=process_frame,
                    inputs=[image_input, image_conf],
                    outputs=[image_output, image_info]
                )
            
            # ============== TAB 3: VIDEO ==============
            with gr.TabItem("🎬 Video Upload"):
                with gr.Row():
                    with gr.Column():
                        video_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                            label="🎚️ Detection Confidence"
                        )
                        video_input = gr.Video(label="Upload Video")
                        video_btn = gr.Button("🎬 Process Video", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.Video(label="🖼️ Result")
                        video_info = gr.Textbox(label="📊 Info", lines=5)
                
                video_btn.click(
                    fn=process_video,
                    inputs=[video_input, video_conf],
                    outputs=[video_output, video_info]
                )
        
        gr.Markdown("""
        ---
        ### 🛠️ Models
        - **Face Detection:** Custom YOLO (`face_yolo_best.pt`)
        - **Age-Gender:** ConvNeXt-Tiny + CORAL Ordinal Regression (`best_coral_model.pth`)
        - **CORAL Thresholds:** 45 (2, 4, 6, ..., 90)
        
        ### 🎨 Color Coding
        - 🔵 **Blue:** Male
        - 🔴 **Red:** Female
        """)
    
    return app


# ============================================================
# 6. LAUNCH
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Face Detection & Age-Gender App")
    print("="*50 + "\n")
    
    app = create_app()
    app.launch(
        share=False,              # Set True for public URL
        server_name="0.0.0.0",    # Allow external connections
        server_port=7861,         # Changed port
        show_error=True
    )
