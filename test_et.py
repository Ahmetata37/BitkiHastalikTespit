import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

# 1. Kütüphane ve Sürüm Kontrolü
print(f"✅ TensorFlow Versiyonu: {tf.__version__}")
print(f"✅ NumPy Versiyonu: {np.__version__}")
print(f"✅ OpenCV Versiyonu: {cv2.__version__}")

# 2. Resim Bulma ve Gösterme
dataset_yolu = "dataset"
bulunan_resimler = []

print("\n 'dataset' klasörü taranıyor...")

# Dataset klasörünün içindeki TÜM alt klasörleri gez
for root, dirs, files in os.walk(dataset_yolu):
    for file in files:
        # Sadece resim dosyalarını al
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            tam_yol = os.path.join(root, file)
            bulunan_resimler.append(tam_yol)
            
            # Çok fazla tarayıp vakit kaybetmemek için 100 tane bulunca dur
            if len(bulunan_resimler) > 100: 
                break
    if len(bulunan_resimler) > 100:
        break

# Sonuçları Değerlendir
if len(bulunan_resimler) > 0:
    print(f" Harika! Toplamda {len(bulunan_resimler)}+ adet resim erişilebilir durumda.")
    
    # Rastgele bir tanesini seç ve göster
    secilen_resim_yolu = random.choice(bulunan_resimler)
    print(f" Örnek olarak şu resim açılıyor: {secilen_resim_yolu}")
    
    try:
        # Resmi oku
        img = cv2.imread(secilen_resim_yolu)
        # Renkleri düzelt (OpenCV BGR okur, biz RGB yapacağız)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ekrana çiz
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.title(f"Örnek Veri: {os.path.basename(os.path.dirname(secilen_resim_yolu))}") # Klasör adını başlık yap
        plt.axis("off")
        plt.show()
        print(" Test Başarılı! Veri seti sağlam.")
        
    except Exception as e:
        print(f" Resim açılırken hata oluştu: {e}")
else:
    print(" HATA: 'dataset' klasöründe hiç resim bulunamadı.")
    print("Lütfen zip dosyasının doğru klasöre ayıklandığından emin ol.")