import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# --- AYARLAR (Hız ve Performans Dengesi) ---
IMG_SIZE = (128, 128)  # Resim boyutu (128x128 hızlıdır)
BATCH_SIZE = 32        # Her seferde işlenecek resim sayısı
EPOCHS = 3             # Şimdilik 3 tur dönecek (Kısa partlar halinde)
KAYIT_YOLU = "models/en_iyi_model.keras" # Modelin kaydedileceği dosya

# --- KLASÖR YOLUNU OTOMATİK BULMA ---
# (Zip'ten çıkan iç içe klasör karmaşasını çözer)
ana_klasor = "dataset"
train_dir = None
valid_dir = None

print(" Eğitim klasörü aranıyor...")
for root, dirs, files in os.walk(ana_klasor):
    # Eğer klasörün içinde hem 'train' hem 'valid' varsa aradığımız yerdir
    if "train" in dirs and "valid" in dirs:
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        break
    # Alternatif isimler (Bazen 'Train', 'Validation' yazar)
    elif "Train" in dirs:
        train_dir = os.path.join(root, "Train")
        if "Valid" in dirs: valid_dir = os.path.join(root, "Valid")
        elif "Validation" in dirs: valid_dir = os.path.join(root, "Validation")
        break

if train_dir:
    print(f" Eğitim Klasörü Bulundu: {train_dir}")
else:
    print(" HATA: 'train' klasörü bulunamadı! Lütfen dataset klasörünü kontrol et.")
    exit()

# --- 1. VERİLERİ YÜKLEME ---
print("\n Resimler hazırlanıyor (Bu işlem bir kez yapılır)...")
# Veri çoğaltma ve normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Renkleri 0-1 arasına çek
    rotation_range=20,      # Resimleri rastgele döndür (Veriyi zenginleştirir)
    horizontal_flip=True,   # Aynalama yap
    validation_split=0.2    # Eğer valid klasörü yoksa %20 ayırır
)

# Eğitim Seti
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' if not valid_dir else None 
)

# Doğrulama (Test) Seti
val_generator = train_datagen.flow_from_directory(
    valid_dir if valid_dir else train_dir, # Valid klasörü yoksa train içinden kullanır
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' if not valid_dir else None
)

# --- 2. MODELİ HAZIRLAMA ---
# Eğer daha önce kaydedilmiş bir model varsa onu yükle, yoksa sıfırdan kur
if os.path.exists(KAYIT_YOLU):
    print(f"\n Önceden eğitilmiş model bulundu: {KAYIT_YOLU}")
    print(" Eğitime KALDIĞI YERDEN devam ediliyor...")
    model = load_model(KAYIT_YOLU)
else:
    print("\n Sıfırdan model oluşturuluyor...")
    # MobileNetV2: Google'ın hafif ve hızlı modeli
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False # Temel bilgileri dondur (Hız kazandırır)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# --- 3. EĞİTİMİ BAŞLAT ---
# Her epoch sonunda en iyi sonucu kaydet
checkpoint = ModelCheckpoint(KAYIT_YOLU, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

print(f"\n EĞİTİM BAŞLIYOR! Hedef: {EPOCHS} Tur")
print("Her tur bittiğinde model otomatik kaydedilecek. İstediğin zaman kapatabilirsin.")

try:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )
    print("\n Tebrikler! Bu part tamamlandı.")
    print(f"Model şuraya kaydedildi: {KAYIT_YOLU}")
    
except KeyboardInterrupt:
    print("\n Eğitim kullanıcı tarafından durduruldu. (Modelin son hali kaydedilmişti zaten).")