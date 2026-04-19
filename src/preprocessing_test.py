import cv2
import numpy as np

def apply_clahe(image_path):
    # Görüntüyü oku
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # CLAHE objesi oluştur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Algoritmayı uygula
    cl1 = clahe.apply(img)
    
    # Sonuçları yan yana göster
    res = np.hstack((img, cl1)) 
    cv2.imshow('Original vs CLAHE', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test etmek için bir termal görüntü yolu ver
# apply_clahe('data/test_thermal.jpg')