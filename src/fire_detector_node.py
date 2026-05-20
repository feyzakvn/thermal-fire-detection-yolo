import cv2
import numpy as np
import time
from ultralytics import YOLO

# ==========================================
# 1. GÖRÜNTÜ ÖN İŞLEME FONKSİYONLARI
# ==========================================

def apply_clahe(frame):
    """Görüntüyü LAB renk uzayına çevirip CLAHE ile duman/alev kontrastını artırır."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_horizon_mask(frame):
    """Dinamik ufuk çizgisi tespiti yapar ve gökyüzünü karartır."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=width//4, maxLineGap=20)
    
    horizon_y = int(height * 0.4) # Fallback (Güvenli Mod) %40
    
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1): 
                horizontal_lines.append((y1 + y2) // 2)
        
        valid_horizons = [y for y in horizontal_lines if int(height * 0.1) < y < int(height * 0.6)]
        if valid_horizons:
            horizon_y = max(valid_horizons)
            
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[0:horizon_y, 0:width] = 0
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Ekranda ufuk çizgisinin nerede olduğunu görmek için ince yeşil bir referans çizgisi çizelim
    cv2.line(masked_frame, (0, horizon_y), (width, horizon_y), (0, 255, 0), 1)
    
    return masked_frame

# ==========================================
# 2. ANA ÇALIŞTIRMA DÖNGÜSÜ (PIPELINE)
# ==========================================

def run_fire_detection_node(model_path, source):
    print(f"[BİLGİ] Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    
    print(f"[BİLGİ] Video kaynağı başlatılıyor: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("[HATA] Video veya Kamera açılamadı! Yolu kontrol edin.")
        return

    # İşlem hızını (FPS) hesaplamak için değişkenler
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[BİLGİ] Video bitti veya akış koptu.")
            break
            
        # İstersen görüntüyü ekrana sığdırmak için yeniden boyutlandırabilirsin
        frame = cv2.resize(frame, (1024, 640))

        # --- AŞAMA 1: ÖN İŞLEME (PREPROCESSING) ---
        # 1. CLAHE ile kontrast artır
        enhanced_frame = apply_clahe(frame)
        
        # 2. Gökyüzünü maskele
        processed_frame = apply_horizon_mask(enhanced_frame)

        # --- AŞAMA 2: YOLOv8 ÇIKARIMI (INFERENCE) ---
        # Model, maskelenmiş ve netleştirilmiş görüntüye bakıyor!
        results = model(processed_frame, conf=0.25, verbose=False) 
        
        # YOLO'nun çizdiği kutuları (bounding boxes) al
        annotated_frame = results[0].plot()

        # --- AŞAMA 3: FPS HESAPLAMA VE EKRANA YAZDIRMA ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Modelin baktığı son hali ekranda göster
        cv2.imshow("AI Fire & Smoke Detection Node (CLAHE + Mask)", annotated_frame)

        # Çıkış yapmak için 'q' tuşuna basılması beklenir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[BİLGİ] Kullanıcı çıkış yaptı.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# TEST BAŞLATICI
# ==========================================
if __name__ == "__main__":
    # 1. Eğittiğin modelin (best.pt) tam yolunu buraya yaz:
    MODEL_YOLU = r"C:\Users\dfeyz\runs\detect\b4_fire_experiments\yolov8n_dfire_baseline\weights\best.pt"
    
    # 2. Test etmek istediğin videonun yolunu yaz (veya bilgisayar kamerası için 0 yaz):
    # KAYNAK = 0  # Web kamerası için
    KAYNAK = 0 # Kendi test videonun yolunu gir
    
    run_fire_detection_node(MODEL_YOLU, KAYNAK)