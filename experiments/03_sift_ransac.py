import cv2
import numpy as np

def boyutlandir(img, max_uzunluk=1000):
    h, w = img.shape[:2]
    if max(h, w) > max_uzunluk:
        oran = max_uzunluk / max(h, w)
        img = cv2.resize(img, None, fx=oran, fy=oran)
    return img


foto1 = cv2.imread(r"C:\Users\Acer\Desktop\vision-uav\venv\foto1.jpeg")
foto1 = boyutlandir(foto1, 1000)

# foto2'yi foto1'in döndürülmüş hali olarak üret
merkez = (foto1.shape[1]//2, foto1.shape[0]//2)
M = cv2.getRotationMatrix2D(merkez, 30, 1.0)  # 30 derece döndür
foto2 = cv2.warpAffine(foto1, M, (foto1.shape[1], foto1.shape[0]))

gri1 = cv2.cvtColor(foto1, cv2.COLOR_BGR2GRAY)
gri2 = cv2.cvtColor(foto2, cv2.COLOR_BGR2GRAY)

# Bu sefer SIFT kullanalım — daha güçlü, açı farklarında daha iyi
sift = cv2.SIFT_create(nfeatures=2000)
keypoints1, descriptors1 = sift.detectAndCompute(gri1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gri2, None)

print(f"Foto1: {len(keypoints1)} nokta, Foto2: {len(keypoints2)} nokta")

# === LOWE'IN ORAN TESTİ ===
# Bu yöntem, "iyi eşleşmeler"i seçmenin standart yoludur.
# Her noktanın EN YAKIN ve İKİNCİ EN YAKIN eşleşmesine bakar.
# En yakın eşleşme, ikinci en yakından çok daha iyiyse → güvenilir
# İkisi de yakınsa → karışıklık var, atla

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

iyi_eslesmeler = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        iyi_eslesmeler.append(m)

print(f"Ham eslesme: {len(matches)}, Lowe testi sonrasi: {len(iyi_eslesmeler)}")

# === RANSAC ile HOMOGRAFİ ===
# En az 10 iyi eslesme gerekli (4 yeterli ama 10 daha güvenli)
if len(iyi_eslesmeler) >= 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
    
    # Homografi matrisi + her eşleşmenin "inlier mi outlier mi" maskesi
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    inlier_sayisi = mask.sum()
    print(f"RANSAC sonrasi gecerli eslesme (inlier): {inlier_sayisi}")
    print(f"Outlier (yanlis) elenen sayisi: {len(iyi_eslesmeler) - inlier_sayisi}")
    
    # Sadece inlier'ları çiz (yanlışları gösterme)
    matchesMask = mask.ravel().tolist()
    
    draw_params = dict(
        matchColor=(0, 255, 0),       # yeşil çizgi
        singlePointColor=None,
        matchesMask=matchesMask,       # sadece inlier'ları göster
        flags=2
    )
    
    sonuc = cv2.drawMatches(foto1, keypoints1, foto2, keypoints2, iyi_eslesmeler, None, **draw_params)
    sonuc = boyutlandir(sonuc, 1600)
    
    cv2.imshow("RANSAC sonrasi - sadece dogru eslesmeler", sonuc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Yeterli eslesme yok, foto cekimi yeniden dene")