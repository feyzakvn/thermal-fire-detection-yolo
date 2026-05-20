# 🔥 B4: Termal/RGB Yangın Tespiti: YOLOv7-Tiny + Video İyileştirme Pipeline

**Geliştirici:** Dudu Feyza Kavun  
**Danışman:** Dr. Yunus Emre ÇOĞURCU  
**Kurum:** Çukurova Üniversitesi, Bilgisayar Mühendisliği Bölümü

Model çıkarım süresi (Inference Speed), NVIDIA GTX 1050 GPU üzerinde kare başına 9.3 ms olarak ölçülmüştür. Sistemin uç donanımlar (NVIDIA Jetson Nano) üzerindeki gerçek zamanlı çıkarım profili ve TensorRT optimizasyon süreçleri, proje arkadaşlarının çalışmalarıyla koordine edilerek raporda tamamen ayrı bir başlık ve tablo altında sunulacaktır.

## 🎯 Proje Amacı
Orman yangınlarına en kısa sürede müdahale edilebilmesi için, termal ve RGB kamera akışlarını gerçek zamanlı analiz ederek alev ve duman tespiti yapan yüksek doğruluklu bir sistem geliştirmek. Proje, hesaplama gücü kısıtlı cihazlarda maksimum doğruluk (mAP) ve hız (FPS) dengesini sağlamak amacıyla YOLOv7-Tiny ve YOLOv8-Nano gibi hafif mimarilerin karşılaştırmalı analizini içerir.

---

## 🗺️ Proje Yol Haritası (Roadmap)

- [x] **Adım 1:** Veri seti araştırması ve D-Fire setinin projeye izole edilmesi.
- [x] **Adım 2:** Veri setinin YOLO formatında etiketlenmesi (`0=fire, 1=smoke`).
- [x] **Adım 3:** Pipeline hatalarının giderilip sızıntısız (seed) dağılım yapılması.
- [x] **Adım 4:** YOLOv7-Tiny ve YOLOv8-Nano ilk pilot eğitimleri (Eski/Kusurlu Altyapı).
- [x] **Adım 5:** Geçerli ve Saf D-Fire Baseline modelinin oluşturulması (Gerçek Test Setinde **%76.8 mAP** başarısı).
- [x] **Adım 6:** Video iyileştirme modülü: RGB LAB-Space CLAHE entegrasyonu.
- [x] **Adım 7:** Ufuk çizgisi tespiti: Gökyüzü maskeleme (Canny + Hough).
- [x] **Adım 8:** Çıkarım motorları için ONNX dönüşümü (`best.onnx` entegrasyonu).
- [x] **Adım 9:** `fire_detector_node` entegrasyonu (Hibrit PyTorch & ONNX Pipeline).
- [ ] **Adım 10:** Jetson üzerinde TensorRT derlemesi ve saha testleri.

---

## 📊 Veri Seti Yapılandırması (Roboflow D-Fire v2)
Önceki aşamalarda tespit edilen çift etiketli (Fire+Smoke) verilerin yanlışlıkla silinmesi ve veri sızıntısı (data leakage) sorunları `master_splitter.py` üzerinde giderilmiş ve deterministik bir dağılım yapılmıştır. Veri seti doğrudan orijinal 2022 makalesinden değil, ardışık kare (overfitting) temizliği yapılmış olan **Roboflow D-Fire - v2** sürümünden elde edilmiştir.

**Nihai D-Fire Veri Dağılımı (Toplam 20.325 Görsel):**
* 🔥 **Sadece Alev (Fire):** `1.168`
* 💨 **Sadece Duman (Smoke):** `4.666`
* 🌋 **Hem Alev Hem Duman:** `4.653`
* ☁️ **Arka Plan (Negatif Örnek):** `9.838` (Yanlış alarmları önlemek için)

*(Dağılım: %80 Train, %10 Valid, %10 Test)*

---

## 🧪 Deneyler ve Performans Analizi (Gerçek Test Seti Metrikleri)
Donanım kısıtlamaları (NVIDIA GTX 1050 - 3GB VRAM) göz önüne alınarak gerçekleştirilen eğitimlerde (50 Epoch, 640x640), modelin daha önce hiç görmediği **2037 adet bağımsız test görseli** üzerinde ulaşılan nihai baseline sonuçları aşağıdadır:

| Değerlendirme Metriği | 📉 Eski/Kusurlu Pipeline | 🚀 Yeni Baseline (Gerçek Test Seti) | Gelişim & Notlar |
| :--- | :---: | :---: | :--- |
| **Genel mAP@.5** | %68.5 | **%76.8** | **+%8.3 Net Artış** (Kusursuz Dağılım) |
| **Alev (Fire) mAP@.5** | %42.0 | **%70.4** | Veri sızıntısının önlenmesiyle **+%28.4** sıçrama. |
| **Duman (Smoke) mAP@.5** | %75.0 | **%83.1** | Güçlü özellik çıkarımı ile stabil başarı. |
| **Çıkarım Modu** | PyTorch (.pt) | **ONNX Runtime (GPU)** | Hibrit node üzerinden evrensel çıkarım desteği. |

### 🔍 Analiz ve Gözlemler
Validation (Doğrulama) metrikleri yerine doğrudan **Test Seti** kullanılarak elde edilen bu sonuçlar, modelin dış ortam koşullarındaki gerçek genelleme yeteneğini yansıtmaktadır. Veri setindeki "hem alev hem duman" içeren 4.653 kritik görselin sisteme geri kazandırılmasıyla alev tespiti %42'den %70.4'e yükselmiştir.

---

## 🚀 Gelecek Adımlar (Next Steps)
Projenin baseline referans noktası ve ön işleme (CLAHE + Maskeleme) pipeline'ı tamamlanmıştır:
1. **YOLOv7-Tiny Eğitimi:** Arka planda devam eden YOLOv7 eğitiminin tamamlanıp YOLOv8 baseline metrikleriyle kafa kafaya karşılaştırılması.
2. **Uç Cihaz (Edge AI) Kurulumu:** Elde edilen ONNX modelinin doğrudan NVIDIA Jetson Nano cihazı üzerinde TensorRT (`.engine`) formatına derlenmesi.
3. **Akademik Raporlama:** Tüm bu süreçlerin bitirme tezi standartlarında dokümante edilmesi.

---
*Bu depo, Çukurova Üniversitesi akademik lisans bitirme projesi günlüğü olarak sürdürülmektedir.*