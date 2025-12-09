import os

# Veri setinin ana klasörü
ana_klasor = "dataset"
train_klasoru = None

print(" Veri seti taranıyor... (Bu işlem sadece klasör isimlerine bakar, hızlıdır)")

# 1. Önce 'train' (Eğitim) klasörünü bulalım
for root, dirs, files in os.walk(ana_klasor):
    if "train" in dirs:
        train_klasoru = os.path.join(root, "train")
        break
    # Bazen klasör adı büyük harfle başlar 'Train'
    elif "Train" in dirs:
        train_klasoru = os.path.join(root, "Train")
        break

# 2. İçindeki bitki türlerini sayalım
if train_klasoru:
    print(f"\n Hedef Klasör Bulundu: {train_klasoru}")
    print("-" * 50)
    print(f"{'SINIF (BİTKİ TÜRÜ)':<40} | {'RESİM SAYISI'}")
    print("-" * 50)
    
    toplam_resim = 0
    siniflar = sorted(os.listdir(train_klasoru)) # Alfabetik sırala
    
    for sinif_adi in siniflar:
        sinif_yolu = os.path.join(train_klasoru, sinif_adi)
        
        # Sadece klasörleri dikkate al
        if os.path.isdir(sinif_yolu):
            dosya_sayisi = len(os.listdir(sinif_yolu))
            print(f"{sinif_adi:<40} | {dosya_sayisi}")
            toplam_resim += dosya_sayisi
            
    print("-" * 50)
    print(f" TOPLAM: {len(siniflar)} farklı hastalık/bitki türü bulundu.")
    print(f" TOPLAM RESİM: {toplam_resim} adet.")
    print("-" * 50)

else:
    print(" 'train' klasörü bulunamadı! Zip dosyası iç içe klasörler oluşturmuş olabilir.")