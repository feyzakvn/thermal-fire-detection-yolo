# 🔥 B4: Termal/RGB Yangın Tespiti: YOLOv7-Tiny + Video İyileştirme Pipeline

**Geliştirici:** Dudu Feyza Kavun  
**Danışman:** Dr. Yunus Emre ÇOĞURCU  
**Kurum:** Çukurova Üniversitesi, Bilgisayar Mühendisliği Bölümü

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
- [x] **Adım 10:** Edge AI Uç Cihaz (Jetson Orin Nano) TensorRT derlemesi, Çoklu Sensör Füzyonu ve Canlı Saha Testleri.
- [ ] **Adım 11:** Nihai proje raporunun ve bitirme tezinin yazılması.

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

## 🧪 1. Deneyler ve Performans Analizi (Baseline Model)
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

## ⚡ 2. Uç Cihaz (Edge AI) Dağıtımı ve Sensör Füzyonu
*Projenin canlı donanım testleri, TensorRT motorları ve log dosyaları `jetson_edge_deployment/` dizininde sunulmaktadır.*

Geliştirilen modelin gerçek dünya koşullarında çalışabilirliğini kanıtlamak amacıyla, mimari **NVIDIA Jetson Orin Nano** uç cihazına entegre edilmiştir. Model, Jetson'ın Tensor Çekirdeklerini (Tensor Cores) kullanabilmesi için FP16 hassasiyetinde `.engine` formatına derlenmiştir.

### 🚀 Donanım Hızlandırma Karşılaştırması (GTX 1050 vs. Jetson Orin Nano)
| Metrik | Geliştirme Ortamı (NVIDIA GTX 1050) | Üretim/Uç Cihaz Ortamı (Jetson Orin Nano) |
| :--- | :--- | :--- |
| **Model Formatı** | ONNX (`best.onnx`) | TensorRT Engine (`best.engine`) |
| **Hassasiyet** | FP32 | **FP16 (Optimizasyonlu)** |
| **Dosya Boyutu** | ~12.2 MB | **~8.9 MB (%27 Hafıza Tasarrufu)** |
| **Çıkarım (Inference) Süresi** | 9.30 ms | **3.72 ms (Medyan GPU Compute)** |
| **İşlem Hacmi (Throughput)**| ~107.5 FPS | **264.31 FPS** |

### 🔥 Çoklu Sensör Füzyonu ve Canlı Saha Testi
Sistem, yanlış alarmları (false-positive) tamamen önlemek için YOLOv8 görsel verisini, **Flir Lepton (Termal)** ve **Intel RealSense (Derinlik)** kameralarından gelen ham matrislerle ROS2 düğümleri üzerinden çapraz doğrulamaya tabi tutar.

Sistemin donanım üzerinde doğrulanması için laboratuvar ortamında kontrollü bir **Çakmak Alevi Testi** gerçekleştirilmiştir. `decision_node.py` içerisine eklenen asenkron kesme (interrupt) mantığı sayesinde sistem;
1. Sıcaklığın **60°C'yi** (`THERMAL_DANGEROUS`) aştığını,
2. Nesnenin **50mm-300mm** geçerli derinlik aralığında olduğunu,
3. Gürültü olmadığını (>5 pixel kümelenme) 

matematiksel olarak doğruladığı anlık saniyede aşağıdaki **CRITICAL (Kritik)** ısı haritası kanıtını kaydetmiştir:

![Canlı Saha Testi - Termal Isı Haritası](jetson_edge_deployment/bitmap/bitmap_162105_CRITICAL.png)

*(Orijinal ROS2 sensör füzyon altyapısı ve Edge AI topolojisi takım arkadaşım **Gizem** tarafından geliştirilmiş olup, bu repodaki `decision_node.py` dosyası anlık kanıt kaydetme mekanizması için modifiye edilmiştir.)*

---

## 🚀 Gelecek Adımlar (Next Steps)
Projenin baseline referans noktası, uç cihaz donanım dağıtımı ve sensör füzyonu testleri başarıyla tamamlanmıştır:
1. **Akademik Raporlama (Bitirme Tezi):** Jüri sunumu için "Abstract - Introduction - Methodology - Results - Discussions - Conclusions" formatında nihai tezin yazımına başlanması.

---
*Bu depo, Çukurova Üniversitesi akademik lisans bitirme projesi günlüğü olarak sürdürülmektedir.*