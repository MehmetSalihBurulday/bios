# BIOS Veri Analizi - Makine Öğrenmesi Modelleri

Bu proje BIOS verileri üzerinde üç farklı makine öğrenmesi modeli uygulamaktadır.

## 📊 Modeller

### 1. **XGBoost** (`1_xgboost_model.py`)
- **Açıklama**: Gradient Boosting tabanlı güçlü bir ağaç ensemble yöntemi
- **Avantajları**: 
  - Yüksek doğruluk ve performans
  - Özellik önemini kolayca elde edebilir
  - Kategorisel veriler doğrudan işlenebilir
- **Parametreler**:
  - `learning_rate`: Öğrenme hızı (default: 0.1)
  - `max_depth`: Ağaç derinliği (default: 5)
  - `n_estimators`: Ağaç sayısı (default: 100)

**Kullanım**:
```bash
python 1_xgboost_model.py
```

**Çıktılar**:
- `xgboost_model.pkl`: Kaydedilen model
- `xgboost_feature_importance.png`: Özellik önem grafiği

---

### 2. **Random Forest** (`2_random_forest_model.py`)
- **Açıklama**: Paralel ağaçlardan oluşan ensemble yöntemi
- **Avantajları**:
  - Parallel işleme ile hızlı eğitim
  - Overfitting'e karşı dayanıklı
  - Okunması ve anlaşılması kolay
- **Parametreler**:
  - `n_estimators`: Ağaç sayısı (default: 100)
  - `max_depth`: Ağaç derinliği (default: 15)
  - `min_samples_split`: Split için minimum örnek (default: 5)

**Kullanım**:
```bash
python 2_random_forest_model.py
```

**Çıktılar**:
- `random_forest_model.pkl`: Kaydedilen model
- `random_forest_feature_importance.png`: Özellik önem grafiği
- `random_forest_depth_analysis.png`: Derinlik analiz grafiği

---

### 3. **ANN (Yapay Sinir Ağı)** (`3_ann_model.py`)
- **Açıklama**: scikit-learn `MLPClassifier` ile çok katmanlı yapay sinir ağı
- **Mimarisi**:
  - Gizli katman kombinasyonları optimize edilir
  - `relu` aktivasyonu ve `adam` optimizer kullanılır
  - `early_stopping=True` ile validation performansına göre durur
- **Avantajları**:
  - Karmaşık doğrusal olmayan ilişkileri öğrenebilir
  - Python 3.13 üzerinde ek derleme problemi olmadan çalışır
  - Potansiyel olarak yüksek performans

**Kullanım**:
```bash
python 3_ann_model.py
```

**Çıktılar**:
- `ann_model.pkl`: Kaydedilen model
- `ann_model.meta.pkl`: Scaler, threshold ve metadata
- `ann_training_history.png`: Loss ve validation score grafikleri
- `ann_confusion_matrix.png`: Karışıklık matrisi

---

### 4. **Model Karşılaştırması** (`4_model_comparison.py`)
Üç modeli aynı veri üzerinde eğitir ve performanslarını karşılaştırır.

**Kullanım**:
```bash
python 4_model_comparison.py
```

**Çıktılar**:
- `model_comparison_results.csv`: Sonuçlar tablosu
- `model_comparison.png`: Bar grafiklerle karşılaştırma
- `model_radar_comparison.png`: Radar grafik ile karşılaştırma

---

## 🚀 Kurulum

1. **Bağımlılıkları Yükle**:
```bash
pip install -r requirements.txt
```

2. **Modelleri Çalıştır**:

**Tüm modelleri ayrı ayrı eğitmek için:**
```bash
python 1_xgboost_model.py
python 2_random_forest_model.py
python 3_ann_model.py
```

**Hepsini bir arada karşılaştırmak için:**
```bash
python 4_model_comparison.py
```

---

## 📈 Performans Metrikleri

Tüm modeller aşağıdaki metriklerle değerlendirilir:

- **Accuracy (Doğruluk)**: Doğru tahminlerin yüzdesi
- **Balanced Accuracy**: Sınıf dengesizliğinde daha güvenilir doğruluk
- **Precision (Hassasiyet)**: Pozitiflerin doğru tahmin oranı
- **Recall (Geri Çağırma)**: Bulunan pozitifler oranı
- **F1 Score**: Precision ve Recall'un harmonik ortalaması
- **ROC AUC**: Çeşitli eşik değerler için performans
- **Average Precision**: Özellikle dengesiz sınıflarda daha anlamlı özet skor

---

## 📊 Veri Seti

Şu anda örnek veriler üretiliyor. Kendi BIOS verinizi kullanmak için:

```python
# model.py dosyalarında load_data yerine:
import pandas as pd

df = pd.read_csv('your_bios_data.csv')
X = df.drop('target_column', axis=1).values
y = df['target_column'].values

model.load_data(X, y)
```

---

## 🔧 Gelişmiş Kullanım

### Model Parametrelerini Ayarlama

**XGBoost Hyperparameters**:
```python
xgb_model.train(
    learning_rate=0.05,  # Daha düşük = daha yavaş ama daha iyi
    max_depth=7,         # Daha derin = daha karmaşık
    n_estimators=200     # Daha fazla ağaç = daha iyi (genelde)
)
```

**Random Forest Hyperparameters**:
```python
rf_model.train(
    n_estimators=200,
    max_depth=20,
    min_samples_split=3
)
```

**ANN Architecture**:
```python
ann_model.build_model(
  input_dim=15,
  layer_sizes=(256, 128, 64),
  learning_rate=0.0008,
  alpha=0.0003
)
```

---

## 📝 Not

- Modeller çeşitli veri türlerine uyarlanabilir
- Hyperparameter tuning ve threshold optimizasyonu eklendi
- Cross-validation sonuçları için `cross_val_score` kullanılabilir

---

## 👨‍💻 Geliştirici

Türkçe BIOS Veri Analizi Projesi
