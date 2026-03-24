# 🧬 ClinVar Missense Varyant Sınıflandırma
**TeknoFest 2026 — Sağlıkta Yapay Zeka**

> ClinVar veritabanındaki missense varyantları **Pathogenic / Benign** olarak sınıflandıran çok-modelli bir makine öğrenmesi sistemi.

---

## 📁 Proje Yapısı

```
.
├── clinvar_eda_pipeline.ipynb        # EDA & ham veri → ML-hazır CSV
├── clinvar_eda_pipeline_v3.ipynb     # EDA pipeline geliştirilmiş v3
├── clinvar_model_training_v2.ipynb   # Model eğitimi v2
└── clinvar_model_training_v3.ipynb   # ✅ Ana model pipeline (son sürüm)
```

**Çalıştırma sırası:** `EDA pipeline → Model training v3`

---

## 🎯 Problem Tanımı

NCBI ClinVar veritabanı, klinik anlam durumu belirlenmiş (veya belirsiz — VUS) genomik varyantlar içermektedir. Bu proje:

- ~8.9 milyon satırlık `variant_summary.txt` dosyasından **missense** varyantları filtreler
- `Pathogenic/Likely Pathogenic` → **1** , `Benign/Likely Benign` → **0** olarak etiketler
- Birden fazla hastalık paeline (PAH, CFTR vb.) genelleme yapabilen classifier üretir

---

## 🛠️ Kurulum

```bash
pip install lightgbm xgboost shap optuna scikit-learn pandas numpy matplotlib seaborn joblib
```

> Not: Notebook'lar Google Colab ortamı için yazılmıştır.

---

## 📊 İş Akışı

### 1️⃣ EDA & Veri Pipeline (`clinvar_eda_pipeline_v3.ipynb`)

| Adım | Açıklama |
|------|----------|
| **Veri İndirme** | NCBI FTP'den `variant_summary.txt.gz` (~700 MB) akış okuma |
| **Bellek Yönetimi** | `usecols` ile yalnızca gerekli sütunlar; 400 MB altı RAM kullanımı |
| **Filtreleme** | Missense + Pathogenic/Benign gold-standard kriterleri |
| **Panel Etiketleme** | Gen bazlı hastalık paneli ataması (PAH→BRCA, CFTR vb.) |
| **Çıktı** | `clinvar_missense_final.csv` — model eğitime hazır |

---

### 2️⃣ Feature Engineering (`clinvar_model_training_v3.ipynb` — Bölüm 2)

Üretilen özellikler:

| Özellik Grubu | Özellikler | Kaynak |
|---|---|---|
| **Aminoasit Biyokimyası** | `charge_change`, `polarity_change`, `size_change`, `hydro_change`, `is_nonsense` | `Name` sütunundan regex |
| **Protein Pozisyonu** | `aa_pos` | `Name` sütunundaki konum numarası |
| **Genetik Değişim Tipi** | `mutation_type` (transition/transversion) | `ReferenceAlleleVCF` + `AlternateAlleleVCF` |
| **Klinik Etki** | `phenotype_count` | `PhenotypeList` pipe-sayısı |
| **Değerlendirme Olgunluğu** | `eval_year` | `LastEvaluated` yılı |
| **Genomik Konum** | `cyto_arm`, `cyto_chr` | `Cytogenetic` |
| **Gen & Köken** | `GeneSymbol`, `OriginSimple` | doğrudan sütun |

**Toplam: 13 özellik** — LabelEncoder ile kategorikler sayısala dönüştürülür.

---

### 3️⃣ Model Mimarisi (`clinvar_model_training_v3.ipynb` — Bölümler 4–8)

```
                   ┌─────────────┐
                   │ Ham Özellik │
                   └──────┬──────┘
             ┌────────────┼────────────┬──────────────┐
             ▼            ▼            ▼               ▼
       LightGBM       XGBoost   RandomForest  LogisticRegression
      (Optuna 100)  (scale_pos)  (balanced)   (StandardScaler)
             │            │            │               │
             └────── Isotonic Kalibrasyon ─────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      Stacking Ensemble         Optuna Ağırlıklı
      (LGB+XGB+RF → LR)          Ensemble (100t)
              └───────────┬───────────┘
                          ▼
                   🏆 Final Ensemble
```

**Hiperparametre optimizasyonu:** Optuna (LightGBM için 100 deneme, `roc_auc` hedef)
**Sınıf dengesizliği çözümü:** `scale_pos_weight` (XGBoost), `class_weight='balanced'` (RF, LR, LGB)

---

### 4️⃣ Eşik Optimizasyonu (Klinik Kısıt)

Varsayılan 0.5 eşiği klinik açıdan yetersizdir çünkü **kaçırılan Pathogenic (FN) kritik risk** taşır.

```
Kısıt: Duyarlılık (Sensitivity/Recall) ≥ 0.90
Hedef: Bu kısıtı sağlayan en yüksek F1
```

Isotonic kalibrasyon ile olasılık tahmini daha güvenilir hale getirilmiştir.

---

### 5️⃣ Değerlendirme Metrikleri

| Metrik | Neden Seçildi |
|--------|--------------|
| **ROC-AUC** | Eşik bağımsız genel ayırt edicilik (ana metrik) |
| **PR-AUC** | Dengesiz sınıflarda hassas ölçüm |
| **F1** | Kesinlik / duyarlılık dengesi |
| **Balanced Accuracy** | Sınıf dengesizliğinde yanıltıcı olmayan doğruluk |
| **Sensitivity** | Klinik kritik — kaçırılan Pathogenic'i minimize eder |
| **Specificity** | FP yükü / yanlış alarm kontrolü |

---

## 📈 Teknik Evrim & Sonuçlar

| Sürüm | Sorun | Müdahale | AUC |
|-------|-------|----------|-----|
| v1 | Colab session çöküyor (RAM ~8 GB) | Chunk-by-chunk okuma, usecols | ~0.78 |
| v2 | Train ~0.98 / Val ~0.81 — overfitting | num_leaves↓, early stopping (50), min_child_samples↑ | ~0.87 |
| v3 | Panel AUC genel modelden ~0.07 düşük | RepeatedStratifiedKFold (5×10 = 50 fold) | ~0.89 |
| v4 | Sensitivity ~0.82 — klinik yetersiz | Sensitivity≥0.90 F1-eşik + isotonic kalibrasyon (Brier -15%) | ~0.90 |
| **v5** | Tek model tavan ~0.90'da sabit | Stacking + Optuna ağırlıklı ensemble | **~0.93+** |

---

## 🔍 SHAP Açıklanabilirlik

`shap.TreeExplainer` ile LightGBM üzerinde hesaplanan özellik gruplarının modele katkısı:

| Özellik Grubu | Önem |
|---|---|
| Gen Kimliği (`GeneSymbol`) | ⭐⭐⭐⭐⭐ En yüksek |
| Biyokimyasal Değişim (charge, hydro, vb.) | ⭐⭐⭐⭐ |
| Klinik Etki Genişliği (`phenotype_count`) | ⭐⭐⭐ |
| Değerlendirme Olgunluğu (`eval_year`) | ⭐⭐ |
| Genomik Konum (`cyto_arm`, `cyto_chr`) | ⭐⭐ |
| Varyant Kökeni (`OriginSimple`) | ⭐ |

---

## ⚠️ Bilinen Sınırlamalar & Dikkat Edilmesi Gerekenler

1. **Veri sızıntısı riski:** `GeneSymbol` özelliği güçlü bir öngörücü olsa da train/test aynı genlerden katkı alabilir. Panel-bazlı CV bunu kısmen kontrol eder.
2. **Sınıf dengesizliği:** Benign:Pathogenic oranı tipik olarak ~70:30; `scale_pos_weight` ve `class_weight='balanced'` kullanılmaktadır.
3. **VUS alanı:** 0.4–0.6 olasılık bölgesi modelin kararsızlık zonudur — klinik karar için ek inceleme önerilir.
4. **`__pycache__` commit:** Derlenmiş `.pyc` dosyaları repo'ya dahil edilmiş; `.gitignore` ile çıkarılması önerilir.
5. **Colab bağımlılığı:** `from google.colab import files` kullanımı lokal ortamda hata verir; yerel çalıştırma için bu hücreleri atlayın.

---

## 💾 Üretilen Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `clinvar_missense_final.csv` | Temizlenmiş ML veri seti |
| `model_lightgbm.pkl` | En iyi LightGBM modeli |
| `model_stacking.pkl` | Stacking ensemble |
| `models_calibrated.pkl` | Kalibrasyonlu 4 model sözlüğü |
| `ensemble_weights.pkl` | Optuna ağırlıkları |
| `label_encoders.pkl` | Kategorik encode sözlüğü |
| `feature_cols.pkl` | Özellik listesi |
| `model_curves.png` | ROC + PR + Kalibrasyon eğrileri |
| `confusion_matrices.png` | Tüm modeller için karışıklık matrisleri |
| `panel_auc.png` | Panel bazlı AUC karşılaştırması |
| `shap_beeswarm.png` | SHAP küresel özellik önemi |
| `shap_force.png` | Örnek tahmin açıklaması |
| `error_analysis.png` | FN/FP dağılım analizi |
| `evolution_chart.png` | Model sürüm evrimi grafiği |

---

## 🚀 Hızlı Başlangıç

1. **EDA notebook'u çalıştırın** → `clinvar_missense_final.csv` elde edin
2. **Model training v3'ü çalıştırın** → tüm modeller eğitilir, `model_lightgbm.pkl` vb. kaydedilir

```python
import joblib, pandas as pd

feature_cols = joblib.load('feature_cols.pkl')
calibrated   = joblib.load('models_calibrated.pkl')
ens_weights  = joblib.load('ensemble_weights.pkl')

# Yeni veri üzerinde tahmin
X_new = pd.read_csv('yeni_varyantlar.csv')[feature_cols].fillna(0)

probs_dict = {n: m.predict_proba(X_new)[:,1] for n, m in calibrated.items()}
total_w    = sum(ens_weights.values())
y_prob_ens = sum(ens_weights[f'w_{k}'] * probs_dict[k] for k in probs_dict) / total_w

print("Pathogenic olasılıkları:", y_prob_ens)
```
