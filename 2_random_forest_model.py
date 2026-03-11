"""Optimize edilmis Random Forest modeli."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from model_utils import (
    evaluate_binary_classifier,
    find_best_threshold,
    generate_bios_like_data,
    print_split_summary,
    split_dataset,
)

class RandomForestBiosModel:
    def __init__(self, random_state=42):
        """Random Forest modelini baslatir."""
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.history = {}
        self.threshold = 0.5
        self.best_params = {}
        self.cv_score = None
        
    def generate_sample_data(self, n_samples=1000, n_features=10):
        """Modelin ogrenebilecegi ornek veri uretir."""
        return generate_bios_like_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=self.random_state,
        )
    
    def load_data(self, X, y, test_size=0.2, val_size=0.2):
        """Veriyi train/validation/test olarak ayirir."""
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_dataset(
            X,
            y,
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state,
        )

        print_split_summary("Egitim seti", self.X_train, self.y_train)
        print_split_summary("Validation seti", self.X_val, self.y_val)
        print_split_summary("Test seti", self.X_test, self.y_test)

    def optimize_hyperparameters(self, n_iter=16):
        """Agac toplulugunu asiri karmasiklastirmadan optimize eder."""
        search_space = {
            "n_estimators": [200, 300, 500, 700],
            "max_depth": [None, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.7, 0.9],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
        estimator = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_score = float(search.best_score_)
        
    def train(self, optimize=True, n_estimators=300, max_depth=None, min_samples_split=2):
        """Modeli egitir ve validation esigini ayarlar."""
        print("\n=== Random Forest Modeli Egitiliyor ===")

        if optimize:
            self.optimize_hyperparameters()
            print(f"Secilen parametreler: {self.best_params}")
            print(f"CV average precision: {self.cv_score:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )

        self.model.fit(self.X_train, self.y_train)
        validation_scores = self.model.predict_proba(self.X_val)[:, 1]
        self.threshold, validation_f1 = find_best_threshold(self.y_val, validation_scores)
        self.history["validation_f1"] = validation_f1
        self.history["threshold"] = self.threshold
        print(f"Validation icin secilen esik: {self.threshold:.2f}")
        print("✓ Model egitimi tamamlandi")
        
    def evaluate(self):
        """Modelin performansini test setinde degerlendirir."""
        print("\n=== Random Forest Performans Degerlendirmesi ===")

        test_scores = self.model.predict_proba(self.X_test)[:, 1]
        self.history.update(evaluate_binary_classifier(self.y_test, test_scores, self.threshold))
        if self.best_params:
            self.history["cv_average_precision"] = self.cv_score

        print(f"Accuracy: {self.history['accuracy']:.4f}")
        print(f"Balanced Accuracy: {self.history['balanced_accuracy']:.4f}")
        print(f"Precision: {self.history['precision']:.4f}")
        print(f"Recall: {self.history['recall']:.4f}")
        print(f"F1: {self.history['f1_score']:.4f}")
        print(f"ROC AUC: {self.history['roc_auc']:.4f}")
        print(f"Average Precision: {self.history['average_precision']:.4f}")
        print(f"\nKarisiklik Matrisi:\n{np.array(self.history['confusion_matrix'])}")

        return self.history
    
    def plot_feature_importance(self, top_n=10):
        """En onemli ozellikleri gosterir."""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices], color="#2ca02c", alpha=0.8)
        plt.yticks(range(len(indices)), [f"Feature {index}" for index in indices])
        plt.xlabel('Onem Derecesi')
        plt.title(f'Random Forest - En Onemli {top_n} Ozellik')
        plt.tight_layout()
        plt.savefig('random_forest_feature_importance.png', dpi=100)
        print("\n✓ Grafik kaydedildi: random_forest_feature_importance.png")
    
    def plot_tree_depth_analysis(self):
        """Agac derinliginin validation performansina etkisini gosterir."""
        depth_range = [2, 4, 6, 8, 12, 16, 20]
        train_scores = []
        val_scores = []
        base_estimators = self.best_params.get('n_estimators', 300) if self.best_params else 300
        base_min_samples_split = self.best_params.get('min_samples_split', 2) if self.best_params else 2
        
        for depth in depth_range:
            rf = RandomForestClassifier(
                n_estimators=base_estimators,
                max_depth=depth,
                min_samples_split=base_min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(self.X_train, self.y_train)
            train_scores.append(rf.score(self.X_train, self.y_train))
            val_scores.append(rf.score(self.X_val, self.y_val))
        
        plt.figure(figsize=(10, 6))
        plt.plot(depth_range, train_scores, label='Egitim Seti', marker='o')
        plt.plot(depth_range, val_scores, label='Validation Seti', marker='s')
        plt.xlabel('Agac Derinligi')
        plt.ylabel('Dogruluk')
        plt.title('Random Forest - Agac Derinliginin Etkisi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('random_forest_depth_analysis.png', dpi=100)
        print("\n✓ Grafik kaydedildi: random_forest_depth_analysis.png")
        
    def save_model(self, path='random_forest_model.pkl'):
        """Modeli esik ve metadata ile kaydeder."""
        joblib.dump(
            {
                'model': self.model,
                'threshold': self.threshold,
                'best_params': self.best_params,
                'history': self.history,
            },
            path,
        )
        print(f"\n✓ Model kaydedildi: {path}")
    
    def predict(self, X):
        """Yeni veriler uzerinde tahmin yapar."""
        scores = self.model.predict_proba(X)[:, 1]
        return (scores >= self.threshold).astype(int)


if __name__ == "__main__":
    rf_model = RandomForestBiosModel()
    X, y = rf_model.generate_sample_data(n_samples=1000, n_features=15)
    rf_model.load_data(X, y)
    rf_model.train(optimize=True)
    rf_model.evaluate()
    rf_model.plot_feature_importance()
    rf_model.plot_tree_depth_analysis()
    rf_model.save_model('random_forest_model.pkl')

    print("\n✓ Random Forest modeli optimizasyonu tamamlandi!")
