"""Optimize edilmis modellerin karsilastirilmasi."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_utils import generate_bios_like_data


def load_class(file_path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


BASE_DIR = Path(__file__).resolve().parent
XGBoostBiosModel = load_class(BASE_DIR / '1_xgboost_model.py', 'XGBoostBiosModel', 'bios_xgb')
RandomForestBiosModel = load_class(BASE_DIR / '2_random_forest_model.py', 'RandomForestBiosModel', 'bios_rf')
ANNBiosModel = load_class(BASE_DIR / '3_ann_model.py', 'ANNBiosModel', 'bios_ann')

class ModelComparison:
    def __init__(self):
        """Model karsilastirmasini baslatir."""
        self.results = {}
        self.X = None
        self.y = None
        
    def generate_sample_data(self, n_samples=1000, n_features=15):
        """Ayni veri seti uzerinde tum modelleri karsilastirir."""
        return generate_bios_like_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=42,
        )
    
    def run_xgboost(self):
        """XGBoost modelini calistirir."""
        print("\n" + "="*50)
        print("1. XGBoost Modeli Baslatiliyor...")
        print("="*50)
        
        xgb_model = XGBoostBiosModel()
        xgb_model.load_data(self.X.copy(), self.y.copy())
        xgb_model.train(optimize=True)
        results = xgb_model.evaluate()
        
        self.results['XGBoost'] = results
        xgb_model.save_model('xgboost_model.pkl')
    
    def run_random_forest(self):
        """Random Forest modelini calistirir."""
        print("\n" + "="*50)
        print("2. Random Forest Modeli Baslatiliyor...")
        print("="*50)
        
        rf_model = RandomForestBiosModel()
        rf_model.load_data(self.X.copy(), self.y.copy())
        rf_model.train(optimize=True)
        results = rf_model.evaluate()
        
        self.results['Random Forest'] = results
        rf_model.save_model('random_forest_model.pkl')
    
    def run_ann(self):
        """ANN modelini calistirir."""
        print("\n" + "="*50)
        print("3. ANN (Sinir Agi) Modeli Baslatiliyor...")
        print("="*50)
        
        ann_model = ANNBiosModel()
        ann_model.load_data(self.X.copy(), self.y.copy())
        ann_model.train(optimize=True)
        results = ann_model.evaluate()
        
        self.results['ANN'] = results
        ann_model.save_model('ann_model')
    
    def compare_models(self):
        """Tum modelleri calistirir ve karsilastirir."""
        self.X, self.y = self.generate_sample_data()
        self.run_xgboost()
        self.run_random_forest()
        self.run_ann()
        self.display_results()
        self.plot_comparison()
    
    def display_results(self):
        """Sonuclari tablo halinde gosterir."""
        print("\n" + "="*60)
        print("MODEL PERFORMANS KARŞILAŞTIRMASI")
        print("="*60)

        df = pd.DataFrame(self.results).T
        numeric_columns = [
            'accuracy',
            'balanced_accuracy',
            'precision',
            'recall',
            'f1_score',
            'roc_auc',
            'average_precision',
            'threshold',
        ]
        summary_df = df[numeric_columns].copy()

        print("\n", df.to_string())
        summary_df.to_csv('model_comparison_results.csv')
        print("\n✓ Sonuçlar kaydedildi: model_comparison_results.csv")

        best_model = summary_df['average_precision'].idxmax()
        print(f"\nEn iyi model (Average Precision): {best_model} ({summary_df.loc[best_model, 'average_precision']:.4f})")
    
    def plot_comparison(self):
        """Modelleri karsilastirmali grafiklerde gosterir."""
        df = pd.DataFrame(self.results).T
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performans Karşılaştırması', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        ax = axes[0, 0]
        df['accuracy'].plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Doğruluk (Accuracy)', fontweight='bold')
        ax.set_ylabel('Değer')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax = axes[0, 1]
        df['precision'].plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Hassasiyet (Precision)', fontweight='bold')
        ax.set_ylabel('Değer')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax = axes[1, 0]
        df['recall'].plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Geri Çağırma (Recall)', fontweight='bold')
        ax.set_ylabel('Değer')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax = axes[1, 1]
        df['f1_score'].plot(kind='bar', ax=ax, color=colors)
        ax.set_title('F1 Skoru', fontweight='bold')
        ax.set_ylabel('Değer')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        print("\n✓ Grafik kaydedildi: model_comparison.png")
        
        self.plot_radar_chart(df)
    
    def plot_radar_chart(self, df):
        """Radar grafik ile modelleri karsilastirir."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (model_name, row) in enumerate(df.iterrows()):
            values = row[metrics].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Model Performans Radar Grafiği', size=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('model_radar_comparison.png', dpi=100, bbox_inches='tight')
        print("✓ Grafik kaydedildi: model_radar_comparison.png")


if __name__ == "__main__":
    comparator = ModelComparison()
    comparator.compare_models()
    
    print("\n" + "="*60)
    print("✓ TÜM MODELLERİN KURULUMU TAMAMLANDI!")
    print("="*60)
    print("\nÜretilen Dosyalar:")
    print("  - xgboost_model.pkl")
    print("  - random_forest_model.pkl")
    print("  - ann_model/")
    print("  - model_comparison_results.csv")
    print("  - model_comparison.png")
    print("  - model_radar_comparison.png")
