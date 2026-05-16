# 🔥 B4: Termal/RGB Yangın Tespiti: YOLOv7-Tiny + Video İyileştirme Pipeline

Bu depo, uç cihazlarda (edge devices - örn. Jetson Nano) çalışmak üzere optimize edilmiş, görüntü işleme ve derin öğrenme tabanlı bir erken uyarı orman yangını tespit sisteminin geliştirilme sürecini, deneylerini ve kaynak kodlarını barındırmaktadır. 

**Geliştirici:** Dudu Feyza Kavun  
**Danışman:** Dr. Yunus Emre ÇOĞURCU  
**Kurum:** Çukurova Üniversitesi, Bilgisayar Mühendisliği Bölümü

## 🎯 Proje Amacı
Orman yangınlarına en kısa sürede müdahale edilebilmesi için, termal ve RGB kamera akışlarını gerçek zamanlı analiz ederek alev ve duman tespiti yapan yüksek doğruluklu bir sistem geliştirmek. Proje, hesaplama gücü kısıtlı cihazlarda maksimum doğruluk (mAP) ve hız (FPS) dengesini sağlamak amacıyla YOLOv7-Tiny ve YOLOv8-Nano gibi hafif mimarilerin karşılaştırmalı analizini içerir.

---

## 🗺️ Proje Yol Haritası (Roadmap)

- [x] **Adım 1:** Termal/RGB yangın veri setleri araştırması (FLAME, D-Fire vb.) ve birleştirilmesi.
- [x] **Adım 2:** Veri setinin temizlenmesi (P-Hash ile kopya silme) ve YOLO formatında etiketlenmesi.
- [ ] **Adım 3:** Video iyileştirme modülü: CLAHE, histogram eşitleme.
- [ ] **Adım 4:** Ufuk çizgisi tespiti: Gökyüzü maskeleme (Canny + Hough).
- [x] **Adım 5:** YOLOv7-Tiny Pilot Eğitimi (Baseline oluşturma).
- [x] **Adım 6:** YOLOv8-Nano ile karşılaştırmalı eğitim (Hız vs. Doğruluk).
- [ ] **Adım 7:** Veri Çoğaltma (Data Augmentation) ile Sınıf Dengesizliğinin (Class Imbalance) çözülmesi.
- [ ] **Adım 8:** ONNX dönüşümü + TensorRT optimizasyonu.
- [ ] **Adım 9:** `fire_detector_node` entegrasyonu (İyileştirme → Maskeleme → Çıkarım).
- [ ] **Adım 10:** Farklı senaryolarda saha testleri ve final model kartı oluşturulması.

---

## 📊 Veri Seti Optimizasyonu
Aşırı uyumu (overfitting) engellemek için ~24.000 ham görsel üzerinde Algısal Parmak İzi (Perceptual Hashing) uygulanmış ve birbirine %95'ten fazla benzeyen **8.531 kopya görsel** silinmiştir. 

**Nihai Veri Seti (8.758 Görsel):**
* ☁️ **Normal (Bulut/Arka Plan):** `3.900` (Yanlış alarmları önlemek için)
* 💨 **Smoke (Duman):** `3.638` (Erken uyarı tespiti için)
* 🔥 **Fire (Alev):** `1.220` *(Sentetik çoğaltma aşaması bekleniyor)*

---

## 🧪 Deneyler ve Performans Analizi (Experiments)
Donanım kısıtlamaları (NVIDIA GTX 1050 - 3GB VRAM) göz önüne alınarak, aynı veri seti ve hiperparametrelerle (50 Epoch, 640x640 Çözünürlük, Batch: 16) iki farklı pilot eğitim gerçekleştirilmiştir. 

| Değerlendirme Metriği | 📉 Exp 01: YOLOv7-Tiny | 🚀 Exp 02: YOLOv8-Nano | Gelişim & Notlar |
| :--- | :---: | :---: | :--- |
| **Model Parametresi** | ~6.0 Milyon | **3.01 Milyon** | YOLOv8 %50 daha hafif. |
| **VRAM Tüketimi** | ~2.58 GB | **2.15 GB** | YOLOv8 daha verimli. |
| **Genel mAP@.5** | %45.8 | **~%68.5** | **+%22.7 Doğruluk Artışı** |
| **Duman mAP@.5** | %60.4 | **~%75.0** | C2f blokları ile daha iyi özellik çıkarımı. |
| **Alev mAP@.5** | %31.2 | **~%42.0** | Veri azlığı nedeniyle darboğaz yaşanıyor. |

### 🔍 Canlı Test Çıkarımları (Inference Analysis)
1. **Bulut/Yanlış Alarm Testleri:** YOLOv7-Tiny bulutları kusursuz şekilde arka plan olarak ayırırken, daha agresif çalışan YOLOv8-Nano düşük güven eşiklerinde yoğun bulutlarda yanlış alarm (False Positive) üretme eğilimindedir. (Eşik optimizasyonu gerektirir).
2. **Düşük Kaliteli Medya:** Bulanık veya kontrastı düşük karanlık görsellerde her iki model de dumanı/alevi kaçırmaktadır. (Görüntü ön işleme gerektirir).
3. **Sınıf Dengesizliği (Class Imbalance):** `Fire` sınıfının yalnızca 1.220 görsel barındırması sebebiyle, Karmaşıklık Matrisine (Confusion Matrix) göre birçok gerçek alev görseli arka plan sanılarak kaçırılmaktadır. 

---

## 🚀 Gelecek Adımlar (Next Steps)
Mevcut deneylerin analizlerine dayanarak sistem şu adımlarla iyileştirilecektir:
1. **CLAHE Algoritması Entegrasyonu:** Karanlık ve düşük kontrastlı videoların model çıkarımından (inference) önce netleştirilmesi.
2. **Data Augmentation:** "Fire" sınıfındaki görsellerin döndürme, bulanıklaştırma ve parlaklık oyunlarıyla sentetik olarak 3.000 seviyelerine çıkarılarak veri setinin dengelenmesi.
3. **Hyperparameter Tuning:** YOLOv8 için optimum güven eşiği (Confidence Threshold) ayarlarının yapılandırılması.

---
*Bu depo, akademik amaçlı bir lisans bitirme projesi günlüğü olarak sürdürülmektedir.*