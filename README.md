Bu projede, diş hekimliği veriseti kullanılarak YOLOv8 ile nesne tespiti (object detection) ve model performans değerlendirmesi gerçekleştirilmektedir. Aşağıda projenin detaylı bir `README.md` dosyası bulunmaktadır:

---

# Dental X-Ray Veriseti ile YOLOv8 Nesne Tespiti ve Performans Analizi

Bu proje, Google Colab ortamında YOLOv8 kullanılarak diş hekimliği röntgen/görsel veriseti üzerinde nesne tespiti modeli eğitilmesini, test edilmesini ve elde edilen tahmin sonuçlarının başarı metriklerinin (Precision, Recall, F1-Score, IoU) hesaplanmasını içeren bir uçtan uca makine öğrenmesi çalışmasıdır.

## Proje Adımları

1. **Google Drive Bağlantısı:** Colab ortamı ile Google Drive entegrasyonu sağlanır.


2. **Veri Filtreleme:** Belirli kriterlere göre (örneğin; belirli sınıfları içeren ve içermeyen etiketler) veri seti filtreleme işlemleri gerçekleştirilir.


3. **YOLO Formatına Dönüştürme:** Ground truth etiket dosyaları YOLO formatına ($x\_center, y\_center, width, height$) uyarlanır.


4. **Model Eğitimi (Training):** `ultralytics` kütüphanesi kullanılarak `yolov8s.pt` modeli ile 100 epoch süreyle eğitim başlatılır.


5. **Ağırlık Dosyasının Kaydedilmesi:** Eğitilen modelin en iyi ağırlıkları (`best.pt`) Google Drive üzerindeki hedef klasöre yedeklenir.


6. **Test ve Tahmin (Inference):** Eğitilen model ile test görüntüleri üzerinde tahminler yapılıp sonuçlar ve etiketler kaydedilir.


7. **Başarı Metrikleri (Evaluation):** Gerçek etiketler (ground truth) ile tahmin sonuçları karşılaştırılarak IoU, Precision, Recall ve F1-Score metrikleri sınıf bazında hesaplanır.



---

##  Kurulum ve Gereksinimler

Projeyi çalıştırmak için gerekli olan temel Python kütüphanesi:

```bash
pip install ultralytics

```

---

##  Kullanım Rehberi

### 1. Veri Hazırlığı ve Format Dönüşümü

Verilerinizi uygun klasör yollarına yerleştirdikten sonra, etiketleri YOLO formatına dönüştürmek için ilgili hücreyi çalıştırabilirsiniz:

```python
python convert_gt_to_yolo(gt_file_path, yolo_file_path)

```

### 2. Model Eğitimi

Modeli başlatmak ve eğitmek için aşağıdaki Python kod bloğunu kullanabilirsiniz:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="/content/drive/MyDrive/DemoColab/donem2/data.yaml",
    epochs=100,
    batch=32,
    lr0=0.01,
    imgsz=416,
    optimizer="AdamW",
)

```

### 3. Test Aşaması ve Tahmin

Eğitilen model ağırlıklarını yükleyerek test verileri üzerinde tahminler gerçekleştirebilirsiniz:

```python
model = YOLO('/content/drive/MyDrive/DemoColab/donem2/weights/bestW2.pt')
results = model.predict(source=test_images_path, save=True, save_txt=True)

```

### 4. Başarı Metriklerinin Hesaplanması

Gerçek etiketler ile model tahminlerini karşılaştırarak sınıf bazlı performans raporu almak için değerlendirme scriptini çalıştırabilirsiniz:

```python
results, mean_iou = evaluate_metrics(gt_folder, pred_folder)
display_results_as_table(results, mean_iou)

```

---

##  Değerlendirme Metrikleri

Kod çıktısı olarak aşağıdaki metrikler elde edilmektedir:

* **Precision (Kesinlik):** Modelin yaptığı tahminlerin ne kadarının doğru olduğunu gösterir.
* **Recall (Duyarlılık):** Gerçek nesnelerin ne kadarının model tarafından başarılı bir şekilde tespit edildiğini gösterir.
* **F1-Score:** Precision ve Recall metriklerinin harmonik ortalamasıdır.
* **IoU (Intersection over Union):** Tahmin edilen kutu ile gerçek kutunun örtüşme oranını ifade eder.
