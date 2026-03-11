"""Hizli baslangic araci."""

import os
import subprocess
import sys


def run_command(args):
    """Komutu calistirir ve basarisiz olursa False doner."""
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        return False
    return True

def print_header(text):
    """Başlık yazdırır."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def install_dependencies():
    """Bagimliliklari yukler."""
    print_header("ADIM 1: Bagimliliklar Yukleniyor")

    if run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip']):
        print("✓ pip guncellendi")
    else:
        print("✗ pip guncellenemedi")

    if not run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']):
        print("✗ Bagimlilik yuklenirken hata olustu.")
        return False

    print("✓ Tum bagimliliklar yüklendi!")
    return True

def run_individual_models():
    """Her modeli ayri ayri calistirir."""
    print_header("ADIM 2: Bireysel Modelleri Calistirma")
    
    models = [
        ("1_xgboost_model.py", "XGBoost"),
        ("2_random_forest_model.py", "Random Forest"),
        ("3_ann_model.py", "ANN (MLP)")
    ]
    
    for script, name in models:
        print(f"\n▶ {name} Modeli Baslatiliyor...")
        if os.path.exists(script):
            if run_command([sys.executable, script]):
                print(f"✓ {name} tamamlandı!")
            else:
                print(f"✗ {name} çalıştırılırken hata oluştu.")
        else:
            print(f"✗ {script} bulunamadı.")

def run_comparison():
    """Model karsilastirmasini calistirir."""
    print_header("ADIM 3: Model Karsilastirmasi")
    
    if os.path.exists("4_model_comparison.py"):
        if run_command([sys.executable, "4_model_comparison.py"]):
            print("✓ Model karşılaştırması tamamlandı!")
        else:
            print("✗ Karşılaştırma çalıştırılırken hata oluştu.")
    else:
        print("✗ 4_model_comparison.py bulunamadı.")

def show_results_summary():
    """Sonuclarin ozetini gosterir."""
    print_header("ADIM 4: SONUCLAR")
    
    print("\nÜretilen Dosyalar:")
    print("  ✓ xgboost_model.pkl")
    print("  ✓ xgboost_feature_importance.png")
    print("  ✓ random_forest_model.pkl")
    print("  ✓ random_forest_feature_importance.png")
    print("  ✓ random_forest_depth_analysis.png")
    print("  ✓ ann_model.pkl")
    print("  ✓ ann_model.meta.pkl")
    print("  ✓ ann_training_history.png")
    print("  ✓ ann_confusion_matrix.png")
    print("  ✓ model_comparison_results.csv")
    print("  ✓ model_comparison.png")
    print("  ✓ model_radar_comparison.png")
    
    print("\n" + "="*60)
    print("  ✓ TEK MODEL CALISTIRIR VEYA KARSILASTIRMA YAPARSANIZ")
    print("="*60)

def main():
    """Ana fonksiyon."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BIOS VERI ANALIZI - MAKINE OGRENMESI MODELLERI        ║")
    print("║  XGBoost | Random Forest | ANN (MLP)                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    print("\nYapilandirma secenekleri:")
    print("  1) Bagimliliklari yukle")
    print("  2) Tum modelleri calistir (bireysel)")
    print("  3) Model karsilastirmasini calistir")
    print("  4) Hepsini yap (1+2+3)")
    print("  0) Cik")
    
    choice = input("\nSecenegi girin (0-4): ").strip()
    
    if choice == "1":
        install_dependencies()
    
    elif choice == "2":
        run_individual_models()
    
    elif choice == "3":
        run_comparison()
    
    elif choice == "4":
        if install_dependencies():
            run_individual_models()
            run_comparison()
        show_results_summary()
    
    elif choice == "0":
        print("Cikiliyor...")
        sys.exit(0)
    
    else:
        print("Gecersiz secenek!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram kullanici tarafindan durduruldu.")
        sys.exit(0)
