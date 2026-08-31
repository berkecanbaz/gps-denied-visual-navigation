import cv2

# Boyutlandırma fonksiyonu
def boyutlandir(img, max_uzunluk=1000):
    h, w = img.shape[:2]
    if max(h, w) > max_uzunluk:
        oran = max_uzunluk / max(h, w)
        img = cv2.resize(img, None, fx=oran, fy=oran)
    return img

# Fotoğrafları yükle ve boyutlandır
foto1 = cv2.imread(r"C:\Users\Acer\Desktop\vision-uav\venv\foto1.jpeg")
foto2 = cv2.imread(r"C:\Users\Acer\Desktop\vision-uav\venv\foto2.jpeg")
foto1 = boyutlandir(foto1, 1000)
foto2 = boyutlandir(foto2, 1000)

# Gri tona çevir
gri1 = cv2.cvtColor(foto1, cv2.COLOR_BGR2GRAY)
gri2 = cv2.cvtColor(foto2, cv2.COLOR_BGR2GRAY)

# ORB ile keypoint ve descriptor çıkar
orb = cv2.ORB_create(nfeatures=1000)
keypoints1, descriptors1 = orb.detectAndCompute(gri1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gri2, None)

print(f"Foto1'de {len(keypoints1)} nokta, Foto2'de {len(keypoints2)} nokta")

# === YENİ KISIM: EŞLEŞTİRME ===

# Brute Force matcher oluştur
# NORM_HAMMING → ORB için doğru mesafe ölçüsü (ORB binary descriptor üretir)
# crossCheck=True → "A'nın en iyi eşleşmesi B ise, B'nin en iyi eşleşmesi de A olmalı" kuralı
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Eşleştir
matches = bf.match(descriptors1, descriptors2)

# Eşleşmeleri mesafeye göre sırala (küçük mesafe = iyi eşleşme)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Toplam {len(matches)} eşleşme bulundu")

# En iyi 50 eşleşmeyi çiz
eslesme_sayisi = 50
eslesme_resmi = cv2.drawMatches(
    foto1, keypoints1,
    foto2, keypoints2,
    matches[:eslesme_sayisi],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Göster
cv2.imshow("Eşleşmeler", eslesme_resmi)
cv2.waitKey(0)
cv2.destroyAllWindows()