from ultralytics import YOLO

def main():
    # 1. Eğittiğin modelin yolu
    model_path = r"C:\Users\dfeyz\runs\detect\b4_fire_experiments\yolov8n_dfire_baseline\weights\best.pt"
    model = YOLO(model_path)

    print("[BİLGİ] Model yükleme başarılı. Gerçek TEST seti üzerinde metrikler hesaplanıyor...")

    # 2. Test splitini çalıştırıyoruz (Windows kilitlenmesin diye workers=0 yapıyoruz)
    metrics = model.val(
        data=r"C:\Users\dfeyz\Desktop\D_Fire_YOLO_Dataset\data.yaml", 
        split='test', 
        save_json=True, 
        plots=True,
        workers=0, # Çoklu işlemci hatasını Windows üzerinde kökten çözer
        name="yolov8n_dfire_real_test"
    )

    print("\n[BAŞARILI] Test metrikleri hesaplandı!")
    print(f"Genel mAP50: {metrics.box.map50}")

# Windows güvenliği için kodun giriş kapısı burası olmak zorunda
if __name__ == '__main__':
    main()