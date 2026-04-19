import os
import random
import shutil

def create_balanced_yolo_dataset(master_dir, output_dir, target_per_class=3900):
    img_dir = os.path.join(master_dir, "images")
    lbl_dir = os.path.join(master_dir, "labels")
    
    print(f"🔍 Havuz Taranıyor: {master_dir}")
    
    # Resim dosyalarını uzantılarıyla birlikte bul (jpg, png vs. eşleşmesi için)
    image_files = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
            name = os.path.splitext(f)[0]
            image_files[name] = f
            
    fire_list = []
    smoke_list = []
    normal_list = []
    
    # Etiketleri oku ve sınıflandır
    for name, img_filename in image_files.items():
        txt_filename = f"{name}.txt"
        txt_path = os.path.join(lbl_dir, txt_filename)
        
        if not os.path.exists(txt_path):
            continue # Etiketi olmayan resmi atla
            
        # Dosya boşsa Normal sınıftır
        if os.path.getsize(txt_path) == 0:
            normal_list.append(name)
            continue
            
        has_fire = False
        has_smoke = False
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                normal_list.append(name)
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    if parts[0] == '0':
                        has_fire = True
                    elif parts[0] == '1':
                        has_smoke = True
        
        # Dengeli olması için sadece net olanları ayrı havuzlara alıyoruz
        if has_fire and not has_smoke:
            fire_list.append(name)
        elif has_smoke and not has_fire:
            smoke_list.append(name)
        elif not has_fire and not has_smoke:
            normal_list.append(name)

    print("-" * 40)
    print(f"Havuzdaki Sadece Alev: {len(fire_list)}")
    print(f"Havuzdaki Sadece Duman: {len(smoke_list)}")
    print(f"Havuzdaki Normal: {len(normal_list)}")
    
    # Hedef sayıya (3900) ulaşamıyorsa olanı al, ulaşıyorsa rastgele seç
    selected_fire = random.sample(fire_list, min(target_per_class, len(fire_list)))
    selected_smoke = random.sample(smoke_list, min(target_per_class, len(smoke_list)))
    selected_normal = random.sample(normal_list, min(target_per_class, len(normal_list)))
    
    print("-" * 40)
    print(f"🎯 Seçilen Alev: {len(selected_fire)}")
    print(f"🎯 Seçilen Duman: {len(selected_smoke)}")
    print(f"🎯 Seçilen Normal: {len(selected_normal)}")
    
    # Tüm seçilenleri birleştir ve karıştır
    all_selected = []
    for name in selected_fire: all_selected.append((name, 'fire'))
    for name in selected_smoke: all_selected.append((name, 'smoke'))
    for name in selected_normal: all_selected.append((name, 'normal'))
    
    random.shuffle(all_selected)
    total_selected = len(all_selected)
    
    # %80 Train, %10 Val, %10 Test Dağılımı
    train_end = int(total_selected * 0.8)
    val_end = train_end + int(total_selected * 0.1)
    
    train_files = all_selected[:train_end]
    val_files = all_selected[train_end:val_end]
    test_files = all_selected[val_end:]
    
    print("-" * 40)
    print(f"📂 Kopyalama Başlıyor...")
    print(f"Eğitim (Train): {len(train_files)} adet")
    print(f"Doğrulama (Val): {len(val_files)} adet")
    print(f"Test (Test): {len(test_files)} adet")
    
    # Yeni klasör yapılarını oluştur
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
        
    # Kopyalama Fonksiyonu
    def copy_files(file_list, split_name):
        for name, _ in file_list:
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
    
    print("-" * 40)
    print(f"✅ İŞLEM TAMAMLANDI! Yeni ve temiz veri setiniz hazır:")
    print(f"👉 {output_dir}")

# --- ÇALIŞTIRMA BÖLÜMÜ ---
# Kaynak: 24.260 verinin olduğu ana havuz
master_klasor = r"C:\Users\dfeyz\Desktop\Master_Dataset"

# Hedef: YOLO'nun eğitime başlayacağı tertemiz ve dengeli yeni klasör
yeni_yolo_klasoru = r"C:\Users\dfeyz\Desktop\B4_YOLO_Dataset"

create_balanced_yolo_dataset(master_klasor, yeni_yolo_klasoru)