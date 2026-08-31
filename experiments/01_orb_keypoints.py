import cv2

# Yardımcı fonksiyon: görüntüyü orantılı küçültür
def boyutlandir(img, max_uzunluk=1000):
    h, w = img.shape[:2]
    if max(h, w) > max_uzunluk:
        oran = max_uzunluk / max(h, w)
        img = cv2.resize(img, None, fx=oran, fy=oran)
    return img

# Fotoğrafları yükle
foto1 = cv2.imread(r"C:\Users\Acer\Desktop\vision-uav\venv\foto1.jpeg")
foto2 = cv2.imread(r"C:\Users\Acer\Desktop\vision-uav\venv\foto2.jpeg")

# Boyutlandır (uzun kenarı 1000 piksel olacak şekilde)
foto1 = boyutlandir(foto1, 1000)
foto2 = boyutlandir(foto2, 1000)

# Gri tona çevir
# (Feature algoritmaları renge ihtiyaç duymaz, gri ton yeterli ve hızlı)
gri1 = cv2.cvtColor(foto1, cv2.COLOR_BGR2GRAY)
gri2 = cv2.cvtColor(foto2, cv2.COLOR_BGR2GRAY)

# ORB detektörü oluştur (en fazla 1000 nokta bulsun)
orb = cv2.ORB_create(nfeatures=1000)

# Her iki fotoğrafta da keypoint'leri ve descriptor'ları bul
keypoints1, descriptors1 = orb.detectAndCompute(gri1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gri2, None)

print(f"Foto1'de {len(keypoints1)} nokta bulundu")
print(f"Foto2'de {len(keypoints2)} nokta bulundu")

# Keypoint'leri fotoğrafların üzerine çiz
foto1_noktali = cv2.drawKeypoints(foto1, keypoints1, None, color=(0, 255, 0))
foto2_noktali = cv2.drawKeypoints(foto2, keypoints2, None, color=(0, 255, 0))

# Göster
cv2.imshow("Foto 1 - Keypoints", foto1_noktali)
cv2.imshow("Foto 2 - Keypoints", foto2_noktali)
cv2.waitKey(0)
cv2.destroyAllWindows()