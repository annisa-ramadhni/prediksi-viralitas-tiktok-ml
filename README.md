# Prediksi Viralitas Konten TikTok Menggunakan MLP dan SVM

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Deskripsi Proyek

Proyek ini merupakan project akhir mata kuliah **Pembelajaran Mesin Dasar** yang berfokus pada prediksi viralitas konten TikTok berdasarkan data engagement media sosial.

Viralitas konten ditentukan berdasarkan jumlah **plays**. Konten dikategorikan sebagai **Viral** apabila jumlah plays berada pada persentil ke-75 atau termasuk dalam 25% data dengan jumlah penayangan tertinggi. Konten yang berada di bawah batas tersebut dikategorikan sebagai **Tidak Viral**.

Penelitian ini membangun dan membandingkan dua model klasifikasi, yaitu:

- **Multi-Layer Perceptron (MLP)**
- **Support Vector Machine (SVM)**

Fitur engagement yang digunakan dalam proses klasifikasi meliputi:

- Likes
- Comments
- Shares
- Plays

Tahapan pengolahan data meliputi data cleaning, normalisasi, penanganan ketidakseimbangan kelas menggunakan **SMOTE**, feature selection menggunakan **SelectKBest**, pembangunan model, serta evaluasi menggunakan beberapa metrik klasifikasi.

---

## 🎯 Tujuan

Proyek ini memiliki beberapa tujuan utama:

1. Membangun model machine learning untuk memprediksi viralitas konten TikTok berdasarkan data engagement.
2. Menganalisis fitur engagement yang paling berkontribusi terhadap klasifikasi viralitas menggunakan **SelectKBest**.
3. Membandingkan performa **Multi-Layer Perceptron (MLP)** dan **Support Vector Machine (SVM)** dalam memprediksi viralitas konten TikTok.

---

## 📊 Dataset

Dataset yang digunakan adalah **TikTok 2025 Dataset** yang diperoleh dari Kaggle.

Pada pemeriksaan awal, dataset terdiri dari:

- **7.225 baris**
- **14 variabel**
- 4 variabel bertipe integer
- 2 variabel bertipe float
- 8 variabel kategorikal/teks

Beberapa atribut dalam dataset antara lain:

- `author`
- `hashtags`
- `music`
- `description`
- `likes`
- `comments`
- `shares`
- `plays`
- `create_time`
- `fetch_time`
- `views`
- `posted_time`

Fitur utama yang digunakan untuk proses klasifikasi adalah:

`likes`, `comments`, `shares`, dan `plays`.

### Sumber Dataset

Dataset berasal dari **TikTok 2025 Dataset** yang diperoleh melalui Kaggle.

> Dataset yang digunakan dalam repository ini merupakan dataset yang telah digunakan dalam proses project dan disimpan pada folder `data/`.

---

## 🏷️ Pembentukan Label Viralitas

Label target dibuat berdasarkan nilai **persentil ke-75 dari plays**.

Aturan klasifikasi:

| Kondisi | Label |
|---|---|
| Plays ≥ persentil ke-75 | Viral |
| Plays < persentil ke-75 | Tidak Viral |

Hasil pembentukan label menghasilkan:

| Kelas | Jumlah |
|---|---:|
| Tidak Viral | 5.337 |
| Viral | 1.881 |

Distribusi tersebut menunjukkan adanya ketidakseimbangan kelas, sehingga diperlukan proses balancing sebelum model dilatih.

---

## 🔄 Tahapan Pengolahan Data

Secara umum, alur project dilakukan melalui beberapa tahapan berikut:

```text
Dataset Collection
        ↓
Data Understanding
        ↓
Data Preprocessing
        ↓
Data Balancing
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Data Splitting
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Result Analysis
        ↓
Prediction & Conclusion
```

### 1. Data Understanding

Tahap awal dilakukan untuk memahami struktur dataset, tipe data, distribusi data, missing value, serta data duplikat.

Hasil pemeriksaan menunjukkan bahwa:

- Dataset awal terdiri dari 7.225 baris dan 14 variabel.
- Tidak terdapat data duplikat.
- Beberapa variabel memiliki missing value dalam jumlah cukup besar.
- `hashtags` memiliki 2.075 missing value.
- `description` memiliki 692 missing value.
- `plays` dan `create_time` masing-masing memiliki 7 missing value.
- `fetch_time`, `views`, dan `posted_time` memiliki 7.218 missing value.

### 2. Data Cleaning

Penanganan missing value dilakukan menggunakan `dropna()` sehingga data yang memiliki nilai kosong dihapus dari dataset.

Setelah proses pembersihan, data yang digunakan untuk proses klasifikasi menjadi lebih bersih dan siap untuk tahap berikutnya.

### 3. Exploratory Data Analysis

EDA dilakukan untuk memahami karakteristik data engagement.

Fitur:

- Likes
- Comments
- Shares
- Plays

memiliki distribusi yang cenderung **right-skewed**, dengan sebagian besar nilai berada pada rentang rendah dan terdapat beberapa nilai yang sangat tinggi.

Analisis korelasi juga dilakukan menggunakan correlation heatmap.

Beberapa hubungan yang terlihat dalam analisis antara lain:

- Likes dan plays memiliki korelasi sebesar **0,79**.
- Likes dan shares memiliki korelasi sebesar **0,61**.
- Shares dan plays memiliki korelasi sebesar **0,50**.
- Comments memiliki korelasi yang lebih rendah dibandingkan fitur engagement lainnya.

---

## ⚖️ Data Balancing dengan SMOTE

Distribusi kelas sebelum balancing menunjukkan bahwa kelas **Tidak Viral** lebih banyak dibandingkan kelas **Viral**.

Untuk menangani ketidakseimbangan tersebut digunakan metode:

**Synthetic Minority Oversampling Technique (SMOTE)**

SMOTE digunakan untuk membentuk data sintetis pada kelas minoritas sehingga jumlah data pada kedua kelas menjadi seimbang.

Setelah proses SMOTE:

| Kelas | Jumlah |
|---|---:|
| Kelas 0 – Tidak Viral | 4.269 |
| Kelas 1 – Viral | 4.269 |

Data hasil balancing kemudian digunakan untuk proses feature selection dan pelatihan model.

---

## 🔎 Feature Selection dengan SelectKBest

Feature selection dilakukan menggunakan:

- `SelectKBest`
- `f_classif` / ANOVA F-value

Tujuannya adalah mengetahui fitur yang paling informatif terhadap target viralitas.

Hasil skor fitur:

| Fitur | Skor |
|---|---:|
| Plays | 1947,20 |
| Likes | 1580,22 |
| Shares | 738,02 |
| Comments | 317,73 |

Berdasarkan hasil tersebut, **plays** memperoleh skor tertinggi, diikuti oleh likes, shares, dan comments.

---

## ✂️ Data Splitting

Dataset dibagi menjadi data training dan testing menggunakan `train_test_split`.

Konfigurasi yang digunakan:

- `test_size = 0.2`
- 80% data training
- 20% data testing
- `stratify = y`
- `random_state = 42`

Penggunaan `stratify=y` dilakukan agar proporsi kelas tetap terjaga pada data training dan testing.

---

## 🤖 Model Machine Learning

Dua metode machine learning digunakan untuk melakukan klasifikasi viralitas:

### 1. Support Vector Machine (SVM)

Model SVM dibangun menggunakan `SVC` dari Scikit-learn.

Parameter utama:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)
```

Kernel **RBF (Radial Basis Function)** digunakan untuk menangani pola hubungan data yang tidak linear.

---

### 2. Multi-Layer Perceptron (MLP)

Pengujian MLP dilakukan menggunakan dua konfigurasi arsitektur.

#### Eksperimen 1 — Dua Hidden Layer

Arsitektur:

```text
Input Layer
    ↓
64 Neuron
    ↓
32 Neuron
    ↓
Output Layer
```

Parameter utama:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)
```

Input terdiri dari empat fitur engagement:

- Likes
- Comments
- Shares
- Plays

Output terdiri dari dua kelas:

- `0` = Tidak Viral
- `1` = Viral

Model dengan arsitektur `(64, 32)` menghasilkan performa **100% pada accuracy, precision, recall, dan F1-score** pada data testing.

#### Eksperimen 2 — Satu Hidden Layer

Eksperimen kedua menggunakan satu hidden layer dengan:

```text
(64)
```

Pengujian ini dilakukan untuk melihat perbedaan performa berdasarkan jumlah hidden layer pada arsitektur MLP.

---

## 📏 Evaluasi Model

Evaluasi dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM | 97,92% | 92,61% | 100% | 96,16% |
| MLP | 100% | 100% | 100% | 100% |

Berdasarkan hasil pengujian pada dataset yang digunakan dalam project, MLP menghasilkan nilai evaluasi yang lebih tinggi pada accuracy, precision, dan F1-score, sementara kedua model memiliki recall sebesar 100%.

### Confusion Matrix SVM

Pada data testing, confusion matrix SVM menunjukkan:

- **1.038** data Tidak Viral berhasil diklasifikasikan dengan benar.
- **376** data Viral berhasil diklasifikasikan dengan benar.
- **30** data Tidak Viral diprediksi sebagai Viral.
- **0** data Viral diprediksi sebagai Tidak Viral.

Dengan demikian, SVM menghasilkan **recall 100%** untuk kelas Viral.

### Confusion Matrix MLP

Model MLP memperoleh accuracy, precision, recall, dan F1-score sebesar **100%** pada data testing yang digunakan dalam project.

---

## 💡 Hasil Utama

Beberapa hasil utama dari project ini adalah:

1. Data engagement TikTok dapat digunakan untuk membangun model klasifikasi viralitas.
2. `plays` memperoleh skor feature selection tertinggi sebesar **1947,20**.
3. `likes` menjadi fitur dengan skor tertinggi kedua sebesar **1580,22**.
4. SMOTE berhasil menyeimbangkan kelas menjadi masing-masing **4.269 data**.
5. SVM menghasilkan:
   - Accuracy: **97,92%**
   - Precision: **92,61%**
   - Recall: **100%**
   - F1-score: **96,16%**
6. MLP dengan arsitektur `(64, 32)` menghasilkan:
   - Accuracy: **100%**
   - Precision: **100%**
   - Recall: **100%**
   - F1-score: **100%**

Hasil tersebut menunjukkan performa model pada **dataset dan konfigurasi eksperimen yang digunakan dalam project ini**.

---

## 🗂️ Struktur Repository

```text
prediksi-viralitas-tiktok-ml/
│
├── data/
│   └── tiktok_merged_data_deduplicated.csv
│
├── docs/
│   └── flowchart project pmd.drawio.png
│
├── notebook/
│   └── KODE_AKHIR_PROJECT_PEMBELAJARAN_MESIN_DASAR_KELOMPOK_1.ipynb
│
└── README.md
```

### Keterangan Folder

| Folder/File | Keterangan |
|---|---|
| `data/` | Dataset yang digunakan dalam project |
| `docs/` | Dokumentasi flowchart project |
| `notebook/` | Notebook yang berisi keseluruhan proses analisis dan pemodelan |
| `README.md` | Dokumentasi project |

---

## 🖼️ Flowchart Project

Flowchart yang menggambarkan keseluruhan alur penelitian tersedia pada folder `docs/`.

Alur penelitian mencakup:

1. Dataset Collection
2. Data Understanding
3. Data Preprocessing
4. Data Balancing
5. Exploratory Data Analysis
6. Feature Selection
7. Data Splitting
8. Model Development
9. Model Evaluation
10. Result Analysis
11. Output
12. Kesimpulan

---

## 🛠️ Tools & Technologies

Project ini menggunakan beberapa tools dan teknologi berikut:

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn / SMOTE**
- **Kaggle Dataset**

Metode utama yang digunakan:

- Data Preprocessing
- Exploratory Data Analysis
- SMOTE
- SelectKBest
- ANOVA F-value
- Support Vector Machine
- Multi-Layer Perceptron
- Classification Report
- Confusion Matrix

---

## 📚 Project Information

**Mata Kuliah:** Pembelajaran Mesin Dasar  
**Semester:** Genap 2025/2026  
**Kelas:** 2024B  
**Topik:** TikTok Virality Prediction

### Anggota Kelompok

1. **Yanaka Sofia Pardede** — 24031554065
2. **Ayu Wulan Anggraeni Putri** — 24031554177
3. **Annisa Ramadhani** — 24031554206

---

## 🔗 Project Links

### Repository Portfolio

Repository GitHub kelompok sebelumnya:

https://github.com/yanakapardede/PROJECT-AKHIR-PEMBELAJARAN-MESIN-DASAR-KELOMPOK-1

### Google Colab

Notebook project sebelumnya juga tersedia melalui Google Colab:

https://colab.research.google.com/drive/1BWs_q2MRWKVcG023dobpIWryVl2o9XbI?usp=sharing

---

## 👤 Author

**Annisa Ramadhani**

Data Science Student

GitHub:  
https://github.com/annisa-ramadhni

---

## 📌 Catatan

Project ini dibuat sebagai bagian dari **Project Akhir Mata Kuliah Pembelajaran Mesin Dasar** Semester Genap 2025/2026.

Hasil evaluasi yang ditampilkan pada repository merupakan hasil eksperimen pada dataset, preprocessing, pembagian data, dan konfigurasi model yang digunakan dalam project.
