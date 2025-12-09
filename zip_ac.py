import zipfile
import os


zip_dosya_adi ="dataset/archive.zip" 
hedef_klasor = "dataset"

print(" Hızlı ayıklama başlıyor... Lütfen bekleyin.")

try:
    with zipfile.ZipFile(zip_dosya_adi, 'r') as zip_ref:
        zip_ref.extractall(hedef_klasor)
    print(" Bitti! Dosyalar hazır.")
except FileNotFoundError:
    print(" HATA: Zip dosyası bulunamadı. İsmi doğru yazdın mı?")