from ultralytics import YOLO

def convert_to_tensorrt():
    # 1. Eğitilmiş PyTorch (best.pt) modelimizin tam yolu
    model_path = r"C:\Users\dfeyz\runs\detect\b4_fire_experiments\yolov8n_dfire_baseline\weights\best.pt"
    
    print(f"[BİLGİ] PyTorch modeli yükleniyor: {model_path}")
    model = YOLO(model_path)
    
    print("[BİLGİ] Model ONNX ve ardından TensorRT formatına derleniyor...")
    print("[BİLGİ] Lütfen bekleyin, ekran kartınızın hızına bağlı olarak bu işlem 3-10 dakika sürebilir.")
    
    # 2. Dışa Aktarma (Export) İşlemi
    exported_file = model.export(
        format="engine",   # Hedef format: TensorRT (.engine)
        half=True,         # FP16 Optimizasyonu: Hassasiyeti 16-bit'e düşürerek model hızını 2 katına çıkarır
        dynamic=False,     # Dinamik boyut yerine sabit (640x640) giriş boyutu kullan (Edge cihazlar için daha performanslıdır)
        simplify=True,     # ONNX grafiğini sadeleştirir 
        workspace=2        # Dönüşüm işlemi için ayrılan maksimum RAM (Gigabayt)
    )
    
    print(f"\n[BAŞARILI] Optimizasyon tamamlandı!")
    print(f"[BİLGİ] Üretilen TensorRT dosyasının konumu: {exported_file}")

if __name__ == "__main__":
    convert_to_tensorrt()