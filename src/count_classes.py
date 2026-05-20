import os

def count_yolo_classes(dataset_path):
    """
    Verilen ana klasördeki (train, test, valid alt klasörleri dahil) 
    tüm txt dosyalarını okuyarak sınıfların sayısını çıkarır.
    """
    fire_count = 0
    smoke_count = 0
    normal_count = 0
    both_count = 0 # Hem alev hem duman içeren resimler

    print(f"\nTarama Başladı: {dataset_path}")
    print("-" * 40)

    # os.walk ile klasörün içindeki tüm alt klasörleri (train, test vb.) gez
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt') and file != "classes.txt":
                filepath = os.path.join(root, file)
                
                # Dosya boyutuna bak, eğer 0 ise veya içi tamamen boşsa Normal'dir
                if os.path.getsize(filepath) == 0:
                    normal_count += 1
                    continue
                
                has_fire = False
                has_smoke = False
                
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    
                    if not lines: # Dosya açıldı ama içi boşsa
                        normal_count += 1
                        continue
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            class_id = parts[0]
                            if class_id == '0':
                                has_fire = True
                            elif class_id == '1':
                                has_smoke = True
                                
                if has_fire and has_smoke:
                    both_count += 1
                    fire_count += 1
                    smoke_count += 1
                elif has_fire:
                    fire_count += 1
                elif has_smoke:
                    smoke_count += 1

    print(f"🔥 Sadece Alev (Fire) İçeren Fotoğraflar: {fire_count - both_count}")
    print(f"💨 Sadece Duman (Smoke) İçeren Fotoğraflar: {smoke_count - both_count}")
    print(f"🌪️ Hem Alev Hem Duman İçerenler: {both_count}")
    print(f"🌲 Normal (Yangın Yok / Boş TXT): {normal_count}")
    print("-" * 40)
    print(f"Toplam Alev Görüntüsü: {fire_count}")
    print(f"Toplam Duman Görüntüsü: {smoke_count}")

# KULLANIM:
# Bilgisayarındaki veri seti klasörünün yolunu buraya yaz.
# Önemli: Bu yolun içinde "labels" klasörünün (veya train/val/test'in) olması yeterlidir.

dfire_yolu = "C:/Users/dfeyz/Desktop/dataset/D-Fire.v2i.yolov7pytorch"
roboflow_03wi7_yolu = "C:/Users/dfeyz/Desktop/dataset/Forest Fire.v5i.yolov7pytorch"

count_yolo_classes(dfire_yolu)
count_yolo_classes(roboflow_03wi7_yolu)