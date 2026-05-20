import os
import random
import shutil

# Sabit Seed (Veri sızıntısını kesin olarak engeller)
random.seed(42)

def create_dfire_yolo_dataset(master_dir, output_dir):
    img_dir = os.path.join(master_dir, "images")
    lbl_dir = os.path.join(master_dir, "labels")
    
    print(f"🔍 D-Fire Havuzu Taranıyor: {master_dir}")
    
    # Resim dosyalarını uzantılarıyla birlikte bul
    image_files = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            name = os.path.splitext(f)[0]
            image_files[name] = f
            
    # Havuzları ayırıyoruz 
    fire_only = []
    smoke_only = []
    both_fire_and_smoke = []
    background = [] # "Normal" sınıfı yerine Negatif/Arka Plan örnekleri
    
    # Etiketleri oku ve sınıflandır
    for name, img_filename in image_files.items():
        txt_filename = f"{name}.txt"
        txt_path = os.path.join(lbl_dir, txt_filename)
        
        if not os.path.exists(txt_path):
            continue # Etiketi hiç olmayan resmi atla
            
        # Dosya boşsa arka plan (negatif) örnektir
        if os.path.getsize(txt_path) == 0:
            background.append(name)
            continue
            
        has_fire = False
        has_smoke = False
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if not lines: # İçi boş satırlıysa yine arka plandır
                background.append(name)
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    if parts[0] == '0': # Roboflow yaml'a göre 0: Fire
                        has_fire = True
                    elif parts[0] == '1': # Roboflow yaml'a göre 1: Smoke
                        has_smoke = True
        
        # 2. "Hem alev hem duman varsa silme" hatasının çözümü
        if has_fire and has_smoke:
            both_fire_and_smoke.append(name)
        elif has_fire:
            fire_only.append(name)
        elif has_smoke:
            smoke_only.append(name)
        else:
            background.append(name)

    print("-" * 50)
    print("📊 D-FIRE VERİ DAĞILIMI (TÜM VERİ KULLANILIYOR)")
    print(f"🔥 Sadece Alev İçerenler: {len(fire_only)}")
    print(f"💨 Sadece Duman İçerenler: {len(smoke_only)}")
    print(f"🌋 Hem Alev Hem Duman İçerenler: {len(both_fire_and_smoke)}")
    print(f"☁️ Arka Plan (Negatif Örnekler): {len(background)}")
    print(f"Toplam Geçerli Veri: {len(fire_only) + len(smoke_only) + len(both_fire_and_smoke) + len(background)}")
    print("-" * 50)
    
    # Stratified Split: Her kategoriyi kendi içinde homojen bölüyoruz
    def split_list(data_list):
        random.shuffle(data_list) # Seed sayesinde hep aynı sırayla karışır
        total = len(data_list)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        return data_list[:train_end], data_list[train_end:val_end], data_list[val_end:]

    # 3. Yapay "target_per_class" sınırı kaldırıldı!
    train_f, val_f, test_f = split_list(fire_only)
    train_s, val_s, test_s = split_list(smoke_only)
    train_b, val_b, test_b = split_list(both_fire_and_smoke)
    train_bg, val_bg, test_bg = split_list(background)

    # Eğitim, doğrulama ve test setlerini birleştir
    train_files = train_f + train_s + train_b + train_bg
    val_files = val_f + val_s + val_b + val_bg
    test_files = test_f + test_s + test_b + test_bg

    # Klasör içi homojenlik için son kez karıştır
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    print(f"📂 Kopyalama Başlıyor...")
    print(f"Eğitim (Train): {len(train_files)} adet")
    print(f"Doğrulama (Val): {len(val_files)} adet")
    print(f"Test (Test): {len(test_files)} adet")
    print("-" * 50)
    
    # Yeni klasör yapılarını oluştur
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
        
    # Kopyalama Fonksiyonu
    def copy_files(file_list, split_name):
        for name in file_list:
            img_src = os.path.join(img_dir, image_files[name])
            txt_src = os.path.join(lbl_dir, f"{name}.txt")
            
            img_dst = os.path.join(output_dir, split_name, 'images', image_files[name])
            txt_dst = os.path.join(output_dir, split_name, 'labels', f"{name}.txt")
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(txt_src, txt_dst)

    # Dosyaları fiziksel olarak taşı
    copy_files(train_files, 'train')
    copy_files(val_files, 'valid')
    copy_files(test_files, 'test')
    
    print(f"✅ İŞLEM TAMAMLANDI! Hocanın standartlarına uygun D-Fire seti hazır:")
    print(f"👉 {output_dir}")

# --- ÇALIŞTIRMA BÖLÜMÜ ---
# Kaynak: D-Fire klasörünün yolu
master_klasor = r"C:\Users\dfeyz\Desktop\D-Fire" # 

# Hedef: Sadece D-Fire'ın olacağı yeni ve temiz YOLO klasörü
yeni_yolo_klasoru = r"C:\Users\dfeyz\Desktop\D_Fire_YOLO_Dataset"

create_dfire_yolo_dataset(master_klasor, yeni_yolo_klasoru)