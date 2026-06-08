# Particle Simulation dengan OpenMP dan OpenCL (UAS Arsitektur dan Sistem Komputer)

## Anggota Tim
* Nama Anggota Tim dan NIM:
  - Shelma Nasywa Ramadhani Munir (25032014056)
  - Khaira Humaira (25032015047)
  - Khashia Arifa Zahra (25032014019)
    
Proyek ini menampilkan simulasi ribuan partikel yang bergerak di bidang 2D secara dinamis. Versi *sequential* digunakan sebagai *baseline*, versi *OpenMP* mempercepat komputasi di CPU, dan versi *OpenCL* menyediakan jalur eksekusi paralel untuk GPU/accelerator.

## 1. Deskripsi Proyek & Fitur Utama
* Heterogeneous Computing: Menunjukkan konsep komputasi heterogen secara sederhana dan visual dengan membandingkan tiga mode eksekusi (Sequential, OpenMP, dan OpenCL).
* Simulasi Dinamis: Setiap partikel menyimpan posisi `(x, y)` dan kecepatan `(vx, vy)`. Partikel bergerak menuju titik pusat gravitasi dan memantul ketika menyentuh batas layar.
* Parameter Dapat Dikonfigurasi: Jumlah partikel, jumlah langkah simulasi (*steps*), dan jumlah thread OpenMP dapat diubah sesuai kebutuhan pengujian.
* Analisis Performa (Benchmark): Dilengkapi pencatatan metrik valid seperti waktu eksekusi (detik), *Speedup*, dan *Efisiensi* thread.

## Parameter Simulasi

Sistem ini menggunakan argumen baris perintah (command-line arguments*) untuk mengatur jalannya simulasi secara dinamis tanpa perlu mengubah kode sumber. Berikut adalah parameter yang dapat dikonfigurasi:

| Parameter | Tipe Data | Fungsi | Nilai Standar / Rekomendasi |
| :--- | :--- | :--- | :--- |
| `numParticles` | `int` | Menentukan total jumlah objek partikel yang akan dihitung posisinya di dalam memori. | `200000` (Ideal untuk menguji batas ekstrem performa GPU) |
| `steps` | `int` | Jumlah iterasi atau langkah waktu (*time-steps*) pergerakan simulasi sebelum program mengekspor hasil akhir. | `300` langkah |
| `omp threads` | `int` | Jumlah utas paralel (*worker threads*) CPU yang dialokasikan saat mengeksekusi mode OpenMP. | `8` (Sesuaikan dengan jumlah *logical processor* CPU Anda) |
| `gravity center` | `float` | Titik koordinat pusat penarikan seluruh objek partikel berdasarkan rumus medan gaya gravitasi. | Diatur secara konstan pada titik tengah kanvas `(0.5, 0.5)` |

### Cara Menjalankan dengan Parameter

Format penulisan argumen saat mengeksekusi program di terminal adalah sebagai berikut:
```bash
.\particle_sim.exe [numParticles] [steps] [omp_threads]
```

### 4. Langkah-Langkah Menjalankan Simulasi
### Prasyarat & Persiapan Environment (Khusus Windows)
Pastikan *compiler* C++ Anda (seperti MinGW-w64 atau MSVC di Visual Studio) mendukung dan mengaktifkan fitur OpenMP, pastikan juga OpenCL SDK atau *driver* GPU (NVIDIA/AMD/Intel) sudah terpasang, serta *path include* untuk file `CL/cl.h` sudah diarahkan dengan benar pada *compiler*.

### Alur Eksekusi Program
1. Pastikan seluruh file `.cpp` dan file terkait berada di dalam folder `src/`.
2. Buka terminal/command prompt, arahkan ke direktori proyek, dan lakukan kompilasi. Jangan lupa menambahkan *flag* paralel (misalnya `-fopenmp` untuk GCC/MinGW) saat mengompilasi versi OpenMP.
3. Sesuaikan parameter simulasi (seperti jumlah partikel atau thread) pada file konfigurasi atau langsung di dalam kode jika diperlukan.
4. Eksekusi file biner hasil kompilasi. Program akan otomatis menjalankan tiga mode secara bergantian:
   * Sequential:Memproses partikel melalui satu *loop* utama di CPU (*baseline*).
   * OpenMP: Membagi beban *loop* ke banyak thread CPU menggunakan pragma `parallel for`.
   * OpenCL: Mengirim data ke *device* GPU, mengeksekusi elemen secara paralel via *kernel*, lalu mengambil kembali hasilnya ke *host*.

## Pencatatan Analisis Benchmark
Setelah simulasi selesai, catat metrik hasil pengujian yang ditampilkan oleh program ke dalam folder `test/`:
* Waktu Sequential: 2.13784 Durasi eksekusi baseline (detik).
* Waktu OpenMP: 1.16987 Durasi eksekusi dengan multi-threading CPU (detik).
* Waktu OpenCL: 2.18805 Durasi eksekusi paralel masif di GPU (detik).
* Speedup OpenMP vs Sequential:1.82742x (lebih cepat)
* Speedup OpenCL vs Sequential: 0.97705x (Mengalami penurunan performa)

## Link Demo/Simulasi
* Link Video Presentasi YouTube: https://youtu.be/WK22jg78o_0
