import os
from PIL import Image
import imagehash

def remove_duplicates(master_dir, threshold=2):
    """
    threshold=2 : İki fotoğraf arasında %95-99 benzerlik varsa kopya sayar.
    """
    img_dir = os.path.join(master_dir, "images")
    lbl_dir = os.path.join(master_dir, "labels")
    
    print(f"🕵️ Kopya Taraması Başlıyor: {img_dir}")
    print("Bu işlem 24 bin fotoğraf için birkaç dakika sürebilir, lütfen bekleyin...\n")
    
    hashes = {}
    duplicates_found = 0
    
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        img_path = os.path.join(img_dir, filename)
        
        try:
            with Image.open(img_path) as img:
                # Fotoğrafın dijital parmak izini çıkar
                img_hash = imagehash.phash(img)
            
            is_duplicate = False
            for existing_hash in hashes:
                # İki parmak izi arasındaki fark threshold'dan küçükse bu bir kopyadır
                if img_hash - existing_hash <= threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                # 1. KOPYA RESMİ SİL
                os.remove(img_path)
                
                # 2. AİT OLDUĞU ETİKETİ (.txt) SİL
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(lbl_dir, txt_filename)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    
                duplicates_found += 1
                
            else:
                # Kopya değilse listeye ekle
                hashes[img_hash] = filename
                
        except Exception as e:
            print(f"Hata: {filename} okunamadı - {e}")

    print("-" * 40)
    print(f"✅ TEMİZLİK TAMAMLANDI!")
    print(f"🗑️ Silinen Kopya Görüntü Sayısı: {duplicates_found}")
    print(f"📂 Kalan Özgün Görüntü Sayısı: {len(hashes)}")

# KULLANIM
master_klasor = r"C:\Users\dfeyz\Desktop\Master_Dataset"
remove_duplicates(master_klasor)