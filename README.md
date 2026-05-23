# Mini Project: Object Counting Mobil Parkir

Program ini menghitung jumlah mobil pada foto aerial parkiran dengan pengolahan citra klasik menggunakan Python, NumPy, OpenCV, dan Matplotlib.
Deteksi tidak menggunakan daftar ROI manual per posisi mobil; kandidat mobil dibuat otomatis dari score map pada evidence mask.

## Cara Menjalankan

```bash
python counting.py --image input/parking_ori.jpg --output output
```

Output utama:

- `output/01_color_space_exploration.png`
- `output/02_parking_line_mask.jpg`
- `output/03_car_evidence_mask.jpg`
- `output/03_red_car_mask.jpg`
- `output/04_sliding_window_score_map.jpg`
- `output/05_detected_cars.jpg`

## Ringkasan Pipeline

1. Eksplorasi visual dilakukan pada RGB, grayscale, HSV, dan LAB. HSV-S membantu melihat mobil berwarna/kaca, sedangkan LAB-L dan grayscale membantu memisahkan mobil terang/gelap dari aspal.
2. Garis parkir putih dideteksi dari threshold HSV warna putih, lalu diproses dengan Canny dan HoughLinesP. Mask garis ini dikurangi dari evidence agar garis parkir tidak ikut dihitung sebagai mobil.
3. Evidence mobil dibuat dari kombinasi:
   - perbedaan grayscale terhadap background lokal hasil Gaussian blur besar,
   - saturasi HSV untuk mobil berwarna dan kaca,
   - area gelap untuk kaca/mobil gelap.
4. Morphology opening dan dilation dipakai untuk membersihkan noise dan menyambungkan bagian mobil yang terfragmentasi.
5. Mobil horizontal dicari otomatis dengan sliding window/score map. Setiap lokasi diberi skor berdasarkan kepadatan evidence mask, lalu local maxima dipilih menggunakan non-maximum suppression agar satu mobil tidak dihitung berkali-kali. Untuk mobil yang terpotong di tepi atas, program memakai window yang lebih pendek khusus area border.
6. Mobil merah dideteksi terpisah dengan threshold HSV merah, morphology, `cv2.findContours`, dan filter area/aspect ratio.

## Hasil

Pada gambar `input/parking_ori.jpg`, program menghasilkan:

```text
Total cars: 31
```

Nilai ini menghitung mobil yang terlihat jelas pada gambar, termasuk mobil yang hanya tampak sebagian di tepi atas gambar.
