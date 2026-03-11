"""Optimize edilmis XGBoost modeli."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from model_utils import (
    evaluate_binary_classifier,
    find_best_threshold,
    generate_bios_like_data,
    print_split_summary,
    split_dataset,
)

class XGBoostBiosModel:
    def __init__(self, random_state=42):
        """XGBoost modelini baslatir."""
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

    def optimize_hyperparameters(self, n_iter=12):
        """Sinirli ama etkili hiperparametre taramasi yapar."""
        negative_count = (self.y_train == 0).sum()
        positive_count = max(1, (self.y_train == 1).sum())
        scale_pos_weight = negative_count / positive_count

        search_space = {
            "n_estimators": [120, 180, 260, 360],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "gamma": [0.0, 0.1, 0.3],
            "reg_lambda": [1.0, 2.0, 5.0],
            "scale_pos_weight": [max(1.0, scale_pos_weight * factor) for factor in (0.8, 1.0, 1.2)],
        }

        estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
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
            n_jobs=1,
            verbose=0,
        )
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_score = float(search.best_score_)
        
    def train(self, optimize=True, learning_rate=0.05, max_depth=4, n_estimators=180):
        """Modeli egitir ve validation esigini ayarlar."""
        print("\n=== XGBoost Modeli Egitiliyor ===")

        if optimize:
            self.optimize_hyperparameters()
            print(f"Secilen parametreler: {self.best_params}")
            print(f"CV average precision: {self.cv_score:.4f}")
        else:
            self.model = xgb.XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
            )

        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        validation_scores = self.model.predict_proba(self.X_val)[:, 1]
        self.threshold, validation_f1 = find_best_threshold(self.y_val, validation_scores)
        self.history["validation_f1"] = validation_f1
        self.history["threshold"] = self.threshold
        print(f"Validation icin secilen esik: {self.threshold:.2f}")
        print("✓ Model egitimi tamamlandi")
        
    def evaluate(self):
        """Modelin performansini test setinde degerlendirir."""
        print("\n=== XGBoost Performans Degerlendirmesi ===")

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
        plt.barh(range(len(indices)), importance[indices], color="#1f77b4")
        plt.yticks(range(len(indices)), [f"Feature {index}" for index in indices])
        plt.xlabel("Onem Derecesi")
        plt.title(f"XGBoost - En Onemli {top_n} Ozellik")
        plt.tight_layout()
        plt.savefig("xgboost_feature_importance.png", dpi=100)
        print("\n✓ Grafik kaydedildi: xgboost_feature_importance.png")
        
    def save_model(self, path='xgboost_model.pkl'):
        """Modeli esik ve metadata ile kaydeder."""
        joblib.dump(
            {
                "model": self.model,
                "threshold": self.threshold,
                "best_params": self.best_params,
                "history": self.history,
            },
            path,
        )
        print(f"\n✓ Model kaydedildi: {path}")
    
    def predict(self, X):
        """Yeni veriler uzerinde tahmin yapar."""
        scores = self.model.predict_proba(X)[:, 1]
        return (scores >= self.threshold).astype(int)


if __name__ == "__main__":
    xgb_model = XGBoostBiosModel()
    X, y = xgb_model.generate_sample_data(n_samples=1000, n_features=15)
    xgb_model.load_data(X, y)
    xgb_model.train(optimize=True)
    xgb_model.evaluate()
    xgb_model.plot_feature_importance()
    xgb_model.save_model('xgboost_model.pkl')

    print("\n✓ XGBoost modeli optimizasyonu tamamlandi!")
