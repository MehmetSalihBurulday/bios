"""Optimize edilmis ANN modeli.

Bu surum TensorFlow yerine scikit-learn MLPClassifier kullanir.
Boylece Python 3.13 uzerinde daha kolay calisir.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from model_utils import (
    evaluate_binary_classifier,
    find_best_threshold,
    generate_bios_like_data,
    print_split_summary,
    split_dataset,
)

class ANNBiosModel:
    def __init__(self, random_state=42):
        """ANN modelini baslatir."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.history = None
        self.metrics = {}
        self.threshold = 0.5
        self.best_config = None
        self.class_weight = None
        
    def generate_sample_data(self, n_samples=1000, n_features=10):
        """Modelin ogrenebilecegi ornek veri uretir."""
        return generate_bios_like_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=self.random_state,
        )
    
    def load_data(self, X, y, test_size=0.2, val_size=0.2):
        """Veriyi train/validation/test olarak ayirir ve normalize eder."""
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_dataset(
            X,
            y,
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state,
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print_split_summary("Egitim seti", self.X_train, self.y_train)
        print_split_summary("Validation seti", self.X_val, self.y_val)
        print_split_summary("Test seti", self.X_test, self.y_test)

        class_counts = np.bincount(self.y_train)
        total = class_counts.sum()
        self.class_weight = {
            0: total / (2 * max(1, class_counts[0])),
            1: total / (2 * max(1, class_counts[1])),
        }
        
    def build_model(self, input_dim, layer_sizes=(128, 64, 32), learning_rate=0.001, alpha=0.0001):
        """MLP tabanli ANN modelini kurar."""
        del input_dim
        self.model = MLPClassifier(
            hidden_layer_sizes=layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=learning_rate,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            max_iter=300,
            random_state=self.random_state,
        )
        return self.model

    def _candidate_configs(self):
        return [
            {
                'layer_sizes': (128, 64),
                'learning_rate': 0.001,
                'alpha': 0.0001,
            },
            {
                'layer_sizes': (256, 128, 64),
                'learning_rate': 0.0008,
                'alpha': 0.0003,
            },
            {
                'layer_sizes': (128, 64, 32),
                'learning_rate': 0.0005,
                'alpha': 0.001,
            },
        ]

    def _balanced_training_data(self):
        """Pozitif sinifi kontrollu bicimde cogaltarak egitim verisini dengeler."""
        class_counts = np.bincount(self.y_train)
        if len(class_counts) < 2 or class_counts[1] == 0:
            return self.X_train, self.y_train

        majority_count = int(class_counts.max())
        X_parts = []
        y_parts = []

        for class_label in (0, 1):
            class_indices = np.where(self.y_train == class_label)[0]
            if len(class_indices) == 0:
                continue
            sampled_indices = np.random.choice(class_indices, size=majority_count, replace=True)
            X_parts.append(self.X_train[sampled_indices])
            y_parts.append(self.y_train[sampled_indices])

        X_balanced = np.vstack(X_parts)
        y_balanced = np.concatenate(y_parts)
        shuffle_indices = np.random.permutation(len(y_balanced))
        return X_balanced[shuffle_indices], y_balanced[shuffle_indices]

    def _fit_candidate(self, config):
        model = self.build_model(
            input_dim=self.X_train.shape[1],
            layer_sizes=config['layer_sizes'],
            learning_rate=config['learning_rate'],
            alpha=config['alpha'],
        )
        X_train_balanced, y_train_balanced = self._balanced_training_data()
        model.fit(X_train_balanced, y_train_balanced)
        validation_scores = model.predict_proba(self.X_val)[:, 1]
        threshold, validation_f1 = find_best_threshold(self.y_val, validation_scores)
        metrics = evaluate_binary_classifier(self.y_val, validation_scores, threshold)
        metrics['validation_f1'] = validation_f1
        return model, threshold, metrics
        
    def train(self, optimize=True, epochs=80, batch_size=32):
        """Mimari taramasi yapar ve en iyi modeli saklar."""
        del epochs
        del batch_size
        print("\n=== ANN Modeli Egitiliyor ===")

        best_score = -1.0
        selected = None
        candidate_configs = self._candidate_configs() if optimize else [
            {
                'layer_sizes': (128, 64, 32),
                'learning_rate': 0.001,
                'alpha': 0.0001,
            }
        ]

        for index, config in enumerate(candidate_configs, start=1):
            print(f"Aday mimari {index}/{len(candidate_configs)} test ediliyor: {config}")
            model, threshold, metrics = self._fit_candidate(config)
            score = metrics['average_precision']
            print(
                f"  validation average precision={metrics['average_precision']:.4f}, "
                f"f1={metrics['validation_f1']:.4f}, threshold={threshold:.2f}"
            )
            if score > best_score:
                best_score = score
                selected = (model, threshold, config, metrics)

        self.model, self.threshold, self.best_config, best_metrics = selected
        self.history = {
            'loss_curve': list(self.model.loss_curve_),
            'validation_scores': list(getattr(self.model, 'validation_scores_', [])),
        }
        self.metrics['validation_f1'] = best_metrics['validation_f1']
        self.metrics['average_precision_validation'] = best_metrics['average_precision']
        print(f"Secilen ANN konfigurasyonu: {self.best_config}")
        print(f"Validation icin secilen esik: {self.threshold:.2f}")
        print("✓ Model egitimi tamamlandi")
        
    def evaluate(self):
        """Modelin performansini test setinde degerlendirir."""
        print("\n=== ANN Performans Degerlendirmesi ===")

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        self.metrics.update(evaluate_binary_classifier(self.y_test, y_pred_proba, self.threshold))
        self.metrics['best_config'] = self.best_config

        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1: {self.metrics['f1_score']:.4f}")
        print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        print(f"\nKarisiklik Matrisi:\n{np.array(self.metrics['confusion_matrix'])}")

        return self.metrics
    
    def plot_training_history(self):
        """Egitim sureci grafiklerini gosterir."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history['loss_curve'], label='Egitim Loss')
        axes[0].set_title('ANN Loss Curve')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Iterasyon')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        validation_scores = self.history.get('validation_scores', [])
        if validation_scores:
            axes[1].plot(validation_scores, label='Validation Score', color='#ff7f0e')
            axes[1].set_title('ANN Validation Score')
            axes[1].set_ylabel('Score')
            axes[1].set_xlabel('Iterasyon')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Validation score kaydi yok', ha='center', va='center')
            axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig('ann_training_history.png', dpi=100)
        print("\n✓ Grafik kaydedildi: ann_training_history.png")
    
    def plot_confusion_matrix(self):
        """Karisiklik matrisini gosterir."""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        cm = np.array(self.metrics.get('confusion_matrix') or [[0, 0], [0, 0]])
        if not self.metrics:
            cm = np.array([[((self.y_test == 0) & (y_pred == 0)).sum(), ((self.y_test == 0) & (y_pred == 1)).sum()], [((self.y_test == 1) & (y_pred == 0)).sum(), ((self.y_test == 1) & (y_pred == 1)).sum()]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.title('ANN - Karışıklık Matrisi')
        plt.tight_layout()
        plt.savefig('ann_confusion_matrix.png', dpi=100)
        print("✓ Grafik kaydedildi: ann_confusion_matrix.png")
    
    def save_model(self, path='ann_model'):
        """Modeli, scaler'i ve metadata'yi kaydeder."""
        model_path = f"{path}.pkl" if not path.endswith('.pkl') else path
        metadata_path = f"{path}.meta.pkl" if not path.endswith('.pkl') else f"{path}.meta.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(
            {
                'scaler': self.scaler,
                'threshold': self.threshold,
                'best_config': self.best_config,
                'metrics': self.metrics,
            },
            metadata_path,
        )
        print(f"\n✓ Model kaydedildi: {model_path}")
        print(f"✓ Metadata kaydedildi: {metadata_path}")
    
    def predict(self, X):
        """Yeni veriler uzerinde tahmin yapar."""
        X_scaled = self.scaler.transform(X)
        scores = self.model.predict_proba(X_scaled)[:, 1]
        return (scores >= self.threshold).astype(int)


if __name__ == "__main__":
    ann_model = ANNBiosModel()
    X, y = ann_model.generate_sample_data(n_samples=1000, n_features=15)
    ann_model.load_data(X, y)
    ann_model.train(optimize=True)
    ann_model.evaluate()
    ann_model.plot_training_history()
    ann_model.plot_confusion_matrix()
    ann_model.save_model('ann_model')

    print("\n✓ ANN modeli optimizasyonu tamamlandi!")
