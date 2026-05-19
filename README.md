# 🔥 B4: Termal/RGB Yangın Tespiti: YOLOv7-Tiny + Video İyileştirme Pipeline

**Geliştirici:** Dudu Feyza Kavun  
**Danışman:** Dr. Yunus Emre ÇOĞURCU  
**Kurum:** Çukurova Üniversitesi, Bilgisayar Mühendisliği Bölümü

Bu depo, uç cihazlarda (edge devices - örn. Jetson Nano) çalışmak üzere optimize edilmiş, görüntü işleme ve derin öğrenme tabanlı bir erken uyarı orman yangını tespit sisteminin geliştirilme sürecini, deneylerini ve kaynak kodlarını barındırmaktadır. 

## 🎯 Proje Amacı
Orman yangınlarına en kısa sürede müdahale edilebilmesi için, termal ve RGB kamera akışlarını gerçek zamanlı analiz ederek alev ve duman tespiti yapan yüksek doğruluklu bir sistem geliştirmek. Proje, hesaplama gücü kısıtlı cihazlarda maksimum doğruluk (mAP) ve hız (FPS) dengesini sağlamak amacıyla YOLOv7-Tiny ve YOLOv8-Nano gibi hafif mimarilerin karşılaştırmalı analizini içerir.

---

## 🗺️ Proje Yol Haritası (Roadmap)

- [x] **Adım 1:** Veri seti araştırması ve D-Fire setinin projeye izole edilmesi.
- [x] **Adım 2:** Veri setinin YOLO formatında etiketlenmesi (`0=fire, 1=smoke`).
- [x] **Adım 3:** Pipeline hatalarının (veri sızıntısı, yanlış sınıf silinmesi) giderilip sızıntısız (seed) dağılım yapılması.
- [x] **Adım 4:** YOLOv7-Tiny ve YOLOv8-Nano ilk pilot eğitimleri (Eski/Kusurlu Altyapı).
- [x] **Adım 5:** Geçerli ve Saf D-Fire Baseline modelinin oluşturulması (Güncel %76.3 mAP başarısı).
- [ ] **Adım 6:** Video iyileştirme modülü: RGB LAB-Space CLAHE entegrasyonu.
- [ ] **Adım 7:** Ufuk çizgisi tespiti: Gökyüzü maskeleme (Canny + Hough).
- [ ] **Adım 8:** ONNX dönüşümü + TensorRT optimizasyonu (Jetson donanımı için).
- [ ] **Adım 9:** `fire_detector_node` entegrasyonu (İyileştirme → Maskeleme → Çıkarım).
- [ ] **Adım 10:** Farklı senaryolarda saha testleri ve final model kartı oluşturulması.

---

## 📊 Veri Seti Yapılandırması (Geçerli Baseline)
Önceki aşamalarda tespit edilen çift etiketli (Fire+Smoke) verilerin yanlışlıkla silinmesi ve veri sızıntısı (data leakage) sorunları `master_splitter.py` üzerinde giderilmiş ve `random.seed(42)` ile deterministik bir dağılım yapılmıştır. "Normal" sınıfı tamamen kaldırılmış, bunun yerine boş etiketli negatif örnekler (background) kullanılmıştır.

**Nihai D-Fire Veri Dağılımı (Toplam 20.325 Görsel):**
* 🔥 **Sadece Alev (Fire):** `1.168`
* 💨 **Sadece Duman (Smoke):** `4.666`
* 🌋 **Hem Alev Hem Duman:** `4.653`
* ☁️ **Arka Plan (Negatif Örnek):** `9.838` (Bulut ve gökyüzündeki yanlış alarmları önlemek için)

*(Dağılım: %80 Train, %10 Valid, %10 Test)*

---

## 🧪 Deneyler ve Performans Analizi (Experiments)
Donanım kısıtlamaları (NVIDIA GTX 1050 - 3GB VRAM) göz önüne alınarak gerçekleştirilen eğitimlerde (50 Epoch, 640x640 Çözünürlük, Batch: 16), veri seti mantık hatasının giderilmesiyle muazzam bir performans sıçraması elde edilmiştir.

| Değerlendirme Metriği | 📉 Eski/Kusurlu Pipeline (v8) | 🚀 Geçerli D-Fire Baseline (v8) | Gelişim & Notlar |
| :--- | :---: | :---: | :--- |
| **Genel mAP@.5** | %68.5 | **%76.3** | **+%7.8 Net Artış** |
| **Alev (Fire) mAP@.5** | %42.0 | **%69.4** | Kayıp verilerin kurtarılmasıyla **+%27.4** sıçrama. |
| **Duman (Smoke) mAP@.5** | %75.0 | **%83.2** | Güçlü özellik çıkarımı ile stabil başarı. |
| **Çıkarım Hızı (Inference)** | - | **9.3 ms / Görüntü** | Jetson/Edge cihazlar için Gerçek Zamanlı (Real-Time) uyumlu. |

### 🔍 Analiz ve Gözlemler
Veri setindeki "hem alev hem duman" içeren 4.653 kritik görselin sisteme geri kazandırılmasıyla modelin alev karakteristiğini öğrenme yeteneği büyük ölçüde artmıştır. Alev başarı oranının %42'den %69.4'e fırlaması, önceki darboğazın veri azlığından değil, hatalı filtreleme mantığından kaynaklandığını kanıtlamıştır.

---

## 🚀 Gelecek Adımlar (Next Steps)
Geçerli ve sızıntısız baseline referans noktamız oluşturulduğuna göre sistem doğrudan dış ortam zorluklarını aşmaya odaklanacaktır:
1. **CLAHE Algoritması Entegrasyonu:** Karanlık, sisli veya düşük kontrastlı videoların, alev renkleri bozulmadan (LAB renk uzayında) model çıkarımından önce netleştirilmesi.
2. **Gökyüzü Maskeleme:** Yoğun beyaz bulutların yanlış alarm (False Positive) üretmesini engellemek için ufuk çizgisinin üstünün maskelenmesi.
3. **TensorRT Optimizasyonu:** Uç cihazlarda (Jetson) çalışabilmesi için PyTorch ağırlıklarının ONNX ve TensorRT motoruna (.engine) dönüştürülmesi.

---
*Bu depo, Çukurova Üniversitesi akademik lisans bitirme projesi günlüğü olarak sürdürülmektedir.*