import cv2
import numpy as np

def apply_clahe_rgb(image_path, output_path=None):
    """
    Renkli (RGB/BGR) görüntülere renkleri bozmadan CLAHE uygular.
    """
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı ({image_path})")
        return None

    # 1. Görüntüyü BGR'den LAB renk uzayına çevir (L: Parlaklık, A ve B: Renk)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 2. Kanalları ayır
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. CLAHE'yi SADECE Parlaklık (L) kanalına uygula (Renkler bozulmasın)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # 4. İyileştirilmiş parlaklık kanalıyla orijinal renk kanallarını geri birleştir
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    
    # 5. Tekrar BGR (standart görüntü) formatına dönüştür
    enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # Eğer kayıt yolu verildiyse kaydet
    if output_path:
        cv2.imwrite(output_path, enhanced_img)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_path}")
        
    return enhanced_img

# --- TEST KISMI ---
if __name__ == "__main__":
    # Test etmek için karanlık veya düşük kaliteli bir yangın fotoğrafının yolunu ver
    test_foto_yolu = "test_medyasi/karanlik_yangin.jpg" # Burayı kendi dosyana göre düzenle
    cikti_foto_yolu = "test_medyasi/clahe_sonuc.jpg"
    
    # Sadece dosya varsa çalıştır
    import os
    if os.path.exists(test_foto_yolu):
        print("CLAHE İşlemi Başlıyor...")
        sonuc_gorsel = apply_clahe_rgb(test_foto_yolu, cikti_foto_yolu)
        
        # Yan yana görmek için (Herhangi bir tuşa basınca kapanır)
        orijinal = cv2.imread(test_foto_yolu)
        yan_yana = np.hstack((orijinal, sonuc_gorsel)) # İki resmi birleştir
        
        # Ekrana sığması için yeniden boyutlandır (Opsiyonel)
        h, w = yan_yana.shape[:2]
        yan_yana_kucuk = cv2.resize(yan_yana, (w//2, h//2))
        
        cv2.imshow("Sol: Orijinal | Sag: CLAHE (Renkli)", yan_yana_kucuk)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Lütfen 'test_foto_yolu' değişkenine geçerli bir resim yolu verin.")