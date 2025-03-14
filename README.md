# Laporan Proyek Machine Learning: Sistem Rekomendasi Hotel 

![cover](https://github.com/user-attachments/assets/f2e3abaa-2a23-4e8d-aea3-2f638e7bc947)


### Disusun oleh: Frisnadi Nurul Huda

Untuk memenuhi submission kedua Recomendation System kelas Machine Learning Terapan - Dicoding Academy.
Proyek ini bertujuan untuk membuat model ML untuk memberikan rekomendasi hotel berdasarkan kesamaan tertentu.


## Project Overview

Peningkatakan pertumbuhan industri hotel pada tiap tahunnya dan preferensi konsumen yang bervariasi dalam kebutuhan layanan hotel mengakibatkan konsumen lebih konsumtif dalam memilih hotel. Kurangnya pilihan kriteria bobot pada penyedia layanan hotel mengakibatkan konsumen mengalami kesulitan dalam memilih hotel yang sesuai dengan preferensinya, sehingga diperlukan sebuah sistem rekomendasi hotel sebagai pilihan alternatif dalam memilih hotel [[Mualiawan, 2022]](https://pdfs.semanticscholar.org/da91/a152e40a95c235e8bc176f176cc7dd2626cd.pdf). 

Sistem rekomendasi hotel dirancang untuk memberikan saran akomodasi yang dipersonalisasi berdasarkan preferensi pengguna, seperti lokasi, jenis kamar, fasilitas, dan ulasan tamu sebelumnya. Dengan memanfaatkan teknik seperti collaborative filtering dan content-based filtering, sistem ini dapat meningkatkan pengalaman pengguna dalam memilih hotel yang sesuai. Implementasi sistem rekomendasi yang efektif tidak hanya meningkatkan kepuasan pelanggan tetapi juga dapat meningkatkan tingkat hunian hotel dengan mencocokkan penawaran dengan permintaan secara lebih akurat.

Banyak Peneliti yang melakukan penelitian dan pengembangan terkait sistem rekomendasi hotel ini, seperti penelitian dengan judul ["Membangun Sistem Rekomendasi Hotel dengan Content Based Filtering Menggunakan K-Nearest Neighbor dan Haversine Formula"](https://pdfs.semanticscholar.org/da91/a152e40a95c235e8bc176f176cc7dd2626cd.pdf) oleh A. Muliawan, dkk. Selanjutnya pada artikel ilmiah dengan judul ["Sistem Rekomendasi Hotel Berbasis Collaborative Filtering Menggunakan Algortima Cosine Similarity Dengan Graph Database"](https://www.researchgate.net/profile/Rajif-Yunmar-2/publication/343123675_Sistem_Rekomendasi_Pemilihan_Hotel_dengan_Case_Based_Reasoning/links/5f178e2c92851cd5fa3be555/Sistem-Rekomendasi-Pemilihan-Hotel-dengan-Case-Based-Reasoning.pdf) yang ditulis oleh Rajif Agung Yunmar.


## Business Understanding

### Problem Statement
Permasalahan dalam proyek ini adalah tentang bagaimana sulitnya pengguna menemukan hotel yang sesuai dengan preferensi mereka secara cepat dan akurat di tengah banyaknya pilihan yang tersedia. 

### Goals
Tujuan proyek ini adalah untuk membuat sistem rekomendasi yang mempermudah pengguna dalam menemukan hotel yang sesuai dengan kebutuhannya.

### Solution Approach
Untuk mencapai tujuan tersebut, dalam proyek ini dikembangkan sistem rekomendasi berbasis **Content-Based Filtering" yang diharapkan dapat mempermudah pengguna dalam menemukan hotel yang sesuai dengan kebutuhannya.


## Data Understanding
Pada proyek ini peneliti menggunakan dataset ["Hotel Recommendation"](https://www.kaggle.com/datasets/keshavramaiah/hotel-recommendation) dari Keshav Ramaiah yang diperoleh dari situs [Kaggle](https://www.kaggle.com). Berikut informasi pada dataset :
+ Dataset memiliki 4 buah file, yang terdiri dari: Hotel_Room_attributes.csv, Hotel_details.csv, Hotel_price_min_max - Formula.csv dan Hotel_RoomPrice.csv. Namun, Peneliti hanya mengambil 2 buah file yaitu Hotel_Room_attributes.csv (atribut kamar hotel) dan Hotel_details.csv {details hotel) saja untuk keperluan pembuatan sistem rekomendasi.
+ Jumlah data atribut kamar hotel / Hotel Room Atribut: 165873
+ Jumlah data details hotel / Hotel Details : 106193

### Fitur pada Dataset Hotel Room Atribut  
1. **id** (`int64`)  
   ID unik untuk setiap kamar yang tersedia dalam dataset.  
2. **hotelcode** (`int64`)  
   Kode unik untuk mengidentifikasi hotel tempat kamar tersebut berada.  
3. **roomamenities** (`object`)  
   Daftar fasilitas atau kelengkapan kamar, seperti AC, Wi-Fi, dan TV.  
4. **roomtype** (`object`)  
   Jenis kamar yang tersedia, seperti **Standard Room, Deluxe Room, Suite**, dll.  
5. **ratedescription** (`object`)  
   Deskripsi rating kamar berdasarkan kategori tertentu.  

### Fitur pada Dataset Hotel Details
1. **id** (`int64`)  
   ID unik untuk setiap hotel dalam dataset.  
2. **hotelcode** (`int64`)  
   Kode unik hotel yang dapat digunakan untuk menghubungkan dataset kamar dan hotel.  
3. **hotelname** (`object`)  
   Nama hotel, seperti **Grand Hotel, Sunset Resort, dll.**  
4. **address** (`object`)  
   Alamat lengkap hotel, yang dapat mencakup **jalan, nomor, dan kawasan**.  
5. **city** (`object`)  
    Kota tempat hotel tersebut berada, seperti **Jakarta, Bali, Tokyo, dll.**  
6. **country** (`object`)  
    Negara tempat hotel berada, seperti **Indonesia, Jepang, Amerika Serikat, dll.**  
7. **zipcode** (`float64`)  
    Kode pos lokasi hotel.  
8. **propertytype** (`object`)  
    Jenis properti hotel, seperti **Hotel, Guesthouse, Hostel, Villa**, dll.  
9. **starrating** (`int64`)  
    Rating bintang hotel (misalnya **3, 4, atau 5 bintang**).  
10. **latitude** (`float64`)  
    Koordinat **lintang** lokasi hotel dalam derajat desimal.  
    - **Nilai positif** → Belahan bumi utara (N).  
    - **Nilai negatif** → Belahan bumi selatan (S).  
11. **longitude** (`float64`)  
    Koordinat **bujur** lokasi hotel dalam derajat desimal.  
    - **Nilai positif** → Belahan bumi timur (E).  
    - **Nilai negatif** → Belahan bumi barat (W).  
12. **Source** (`int64`)  
    Sumber data hotel, yang mungkin digunakan untuk tracking data dari berbagai platform.  
13. **url** (`object`)  
    URL halaman pemesanan hotel atau informasi resmi.  
14. **curr** (`object`)  
    Mata uang yang digunakan untuk harga kamar hotel, seperti **IDR, USD, EUR**, dll.  

## Exploratory Data analysis (EDA)
Dalam tahap awal eksplorasi data, peneliti melakukan analisis dasar dengan menggunakan dua fungsi utama, yaitu `data.info()` dan `data.head()`. Fungsi `data.info()` memberikan gambaran menyeluruh mengenai struktur dataset, meliputi jumlah baris, tipe data tiap kolom, dan jumlah nilai non-null pada setiap kolom. Informasi ini sangat penting untuk mengidentifikasi adanya missing values dan memastikan kesesuaian tipe data setiap variabel untuk analisis selanjutnya.

### Room Hotel Attributes

| #  | Column           | Non-Null Count   | Dtype  |
|----|------------------|------------------|--------|
| 0  | id               | 165873 non-null  | int64  |
| 1  | hotelcode        | 165873 non-null  | int64  |
| 2  | roomamenities    | 161054 non-null  | object |
| 3  | roomtype         | 165873 non-null  | object |
| 4  | ratedescription  | 161054 non-null  | object |

Dataset ini terdiri dari 5 kolom dengan total 165,873 entri. Kolom id dan hotelcode memiliki 165,873 nilai non-null dengan tipe data int64, sedangkan roomamenities dan ratedescription memiliki 161,054 nilai non-null, menunjukkan adanya beberapa missing values pada kolom-kolom tersebut. Kolom roomtype lengkap dengan 165,873 entri, menyediakan informasi tentang kategori kamar yang tersedia.

### Hotel Details

| #  | Column       | Non-Null Count | Dtype   |
|----|------------|----------------|--------|
| 0  | id         | 108048          | int64  |
| 1  | hotelid    | 108048          | int64  |
| 2  | hotelname  | 108048          | object |
| 3  | address    | 102955          | object |
| 4  | city       | 108047          | object |
| 5  | country    | 108048          | object |
| 6  | zipcode    | 83486           | float64 |
| 7  | propertytype | 108048        | object |
| 8  | starrating | 108048          | int64  |
| 9  | latitude   | 108048          | float64 |
| 10 | longitude  | 108048          | float64 |
| 11 | Source     | 108048          | int64  |
| 12 | url        | 107937          | object |
| 13 | curr       | 108048          | object |

Tabel ini berisi informasi details hotel dengan 14 kolom utama, termasuk id (identifikasi unik hotel), hotelname (nama hotel), address (alamat hotel), city dan country (lokasi hotel), serta propertytype (jenis properti). Selain itu, terdapat informasi seperti starrating (peringkat bintang), latitude dan longitude (koordinat lokasi), serta url yang mengarah ke halaman hotel. Beberapa kolom memiliki nilai yang hilang, seperti address dan zipcode, yang perlu ditangani sebelum analisis lebih lanjut.

Selanjutnya, peneliti menggunakan `data.head()` untuk menampilkan beberapa baris pertama dari dataset. Langkah ini memungkinkan peneliti untuk mengobservasi contoh data secara langsung, memahami format data, dan mengevaluasi integritas data yang telah dilakukan penggabungan antar kolom.

### Room Hotel Attributes

| id       | hotelcode | roomamenities                                        | roomtype               | ratedescription                                    |
|----------|----------|------------------------------------------------------|------------------------|----------------------------------------------------|
| 50677497 | 634876   | Air conditioning; Alarm clock; Carpeting; ...       | Double Room           | Room size: 15 m²/161 ft², Shower, 1 king bed      |
| 50672149 | 8328096  | Air conditioning; Closet; Fireplace; Free WiFi ...  | Vacation Home         | Shower, Kitchenette, 2 bedrooms, 1 double bed ... |
| 50643430 | 8323442  | Air conditioning; Closet; Dishwasher; Fireplace ... | Vacation Home         | Shower, Kitchenette, 2 bedrooms, 1 double bed ... |
| 50650317 | 7975     | Air conditioning; Clothes rack; Coffee/tea maker ... | Standard Triple Room  | Room size: 20 m²/215 ft², Shower, 3 single beds  |
| 50650318 | 7975     | Air conditioning; Clothes rack; Coffee/tea maker ... | Standard Triple Room  | Room size: 20 m²/215 ft², Shower, 3 single beds  |

Setiap kamar hotel dalam tabel memiliki informasi mendetail mengenai fasilitas dan jenis kamar yang tersedia. **hotelcode** menunjukkan kode unik untuk masing-masing hotel yang menyediakan kamar tersebut. Pada kolom **roomamenities**, terdapat daftar fasilitas yang melengkapi setiap kamar, seperti AC, jam alarm, karpet, lemari, WiFi gratis, mesin kopi/teh, dan lainnya, yang berbeda di setiap tipe kamar. **roomtype** mengklasifikasikan jenis kamar yang tersedia, termasuk *Double Room*, *Vacation Home*, dan *Standard Triple Room*, yang menunjukkan kapasitas dan tipe akomodasi yang ditawarkan. Sementara itu, **ratedescription** memberikan deskripsi tambahan mengenai ukuran kamar, jenis tempat tidur, serta fasilitas utama seperti shower, dapur kecil, dan jumlah kamar tidur, misalnya *Room size: 15 m²/161 ft², Shower, 1 king bed* untuk *Double Room* atau *Room size: 20 m²/215 ft², Shower, 3 single beds* untuk *Standard Triple Room*.

### Hotel Details

| id    | hotelid  | hotelname                   | address                     | city      | country  | zipcode  | propertytype  | starrating | latitude  | longitude  | Source | url                                           | curr |
|-------|---------|----------------------------|-----------------------------|-----------|---------|----------|--------------|------------|-----------|------------|--------|-----------------------------------------------|------|
| 46406 | 1771651 | Mediteran Bungalow Galeb   | Vukovarska 7                | Omis      | Croatia  | 21310.0  | Holiday parks | 4          | 43.440124 | 16.682505  | 2      | https://www.booking.com/hotel/hr/bungalow-luxu... | EUR  |
| 46407 | 177167  | Hotel Polonia              | Plac Teatralny 5            | Torun     | Poland   | NaN      | Hotels       | 3          | 53.012329 | 18.603800  | 5      | https://www.agoda.com/en-gb/hotel-polonia/hote... | EUR  |
| 46408 | 1771675 | Rifugio Sass Bece          | Belvedere del Pordoi,1      | Canazei   | Italy    | 38032.0  | Hotels       | 3          | 46.477920 | 11.813350  | 2      | http://www.booking.com/hotel/it/rifugio-sass-b... | EUR  |
| 46409 | 177168  | Madalena Hotel             | Mykonos                     | Mykonos   | Greece   | 84600.0  | Hotels       | 3          | 37.452316 | 25.329849  | 5      | https://www.agoda.com/en-gb/madalena-hotel/hot... | EUR  |
| 46410 | 1771718 | Pension Morenfeld          | Mair im Korn Strasse 2      | Lagundo   | Italy    | 39022.0  | Hotels       | 3          | 46.682780 | 11.131736  | 2      | http://www.booking.com/hotel/it/pension-morenf... | EUR  |

Setiap hotel dalam tabel memiliki informasi lengkap mengenai identitas dan lokasinya. **hotelid** berisi kode unik untuk masing-masing hotel, sementara **hotelname** mencantumkan nama hotel seperti *Mediteran Bungalow Galeb*, *Hotel Polonia*, dan *Rifugio Sass Bece*. Lokasi hotel dijelaskan dalam kolom **address**, dengan rincian seperti *Vukovarska 7* di Omis dan *Plac Teatralny 5* di Torun, serta didukung oleh informasi kota dan negara pada kolom **city** dan **country**. Beberapa hotel juga memiliki kode pos yang tercantum dalam **zipcode**, meskipun ada yang tidak tersedia. Berdasarkan jenis properti dalam **propertytype**, hotel-hotel ini diklasifikasikan sebagai *Hotels* atau *Holiday parks*, dengan peringkat bintang yang tertera di **starrating**, berkisar dari tiga hingga empat bintang. Untuk membantu dalam pencarian lokasi, koordinat geografis hotel tercantum dalam **latitude** dan **longitude**. Sumber data setiap hotel ditampilkan dalam **Source**, dengan angka tertentu yang menunjukkan platform asal informasi. Setiap hotel juga memiliki tautan pemesanan yang tersedia dalam **url**, mengarahkan pengguna ke situs seperti Booking.com atau Agoda. Semua harga dalam tabel dinyatakan dalam mata uang Euro, sebagaimana tercantum dalam **curr**.


## Data Prepocessing
Sebelum dataset diolah, kolom `hotelid` pada dataset Hotel_details diubah menjadi `hotelcode`, kemudian berdasarkan kolom `hotelcode` kedua dataset digabungkan menjadi satu.
Setelah digabungkan (merge) jumlah dataset menjadi 181415 sampel dengan 18 kolom/fitur. Kemudian peneliti hanya mengambil fitur `hotelname`, `roomtype`, `city`, dan `country`, yang digunakan untuk membangun sistem rekomendasi hotel.

## Data Preparation

### Checking Missing value and Duplicated Value

![missing_value](https://github.com/user-attachments/assets/e777468f-6f42-4b00-b395-d516e2692802)
![Duplicated](https://github.com/user-attachments/assets/9ba4edd4-ebe0-4e7c-8591-6bc4aecf9b1f)


Tidak terdapat missing value tetapi terdapat duplicated value pada dataset sehingga harus ditangani.

### Handling Duplicated Value
untuk mengatasi data yang terduplikasi digunakan metode `drop_duplicates()`. Setelah ditangani dataset berkurang menjadi 49496.

### Cleaning and Concat Dataset values
Dataset dibersihkan dari format yang tidak sesuai dengan cara menghilangkan karakter yang mengganggu (";", ";"), kemudian dijadikan lowecase semua agar seragam.
Setelah bersih, fitur `hotelname`, `roomtype`, `city`, dan `country` digabungkan menjadi satu dengan diberi nama fitur `tags`. fitur inilah yang selanjutnya diolah menjadi dasar sistem rekomendasi hotel.


## Modeling and Result
Peneliti menggunakan metode Content-Based Filtering (CBF) untuk menjadi dasar sistem rekomendasi.**Content-Based Filtering (CBF)** adalah metode rekomendasi yang menggunakan karakteristik atau fitur dari item untuk memberikan rekomendasi yang relevan bagi pengguna. Model ini membandingkan kesamaan antara item berdasarkan fitur seperti hotel, tipe kamar dan dan lokasi hotel.

### Feature Extraction
Feature extraction adalah proses mengubah data mentah menjadi representasi numerik yang dapat digunakan dalam pemodelan machine learning. Dalam pemrosesan teks, metode feature extraction bertujuan untuk mengubah teks menjadi bentuk yang dapat dianalisis oleh algoritma. Salah satu teknik yang digunakan adalah **TF-IDF (Term Frequency-Inverse Document Frequency)**, yang mengubah teks menjadi vektor numerik berbobot berdasarkan frekuensi kata dalam dokumen dan kepentingannya dalam seluruh kumpulan dokumen. Setelah representasi vektor diperoleh, teknik seperti **Truncated SVD (Latent Semantic Analysis/LSA)** diterapkan untuk mereduksi dimensi, sehingga data menjadi lebih efisien tanpa kehilangan banyak informasi. Kombinasi TF-IDF dan Truncated SVD memungkinkan ekstraksi fitur yang lebih optimal, menjaga informasi penting dari teks sekaligus mengurangi kompleksitas komputasi.

### Cosine Similarity
**Cosine Similarity** adalah teknik dalam CBF yang mengukur tingkat kesamaan antara dua vektor berdasarkan sudut di antara mereka. Nilainya berkisar dari 0 hingga 1, di mana nilai lebih tinggi menunjukkan kemiripan yang lebih besar.  

Berikut adalah rumus **Cosine Similarity**:  

$$
\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

Di mana:  
- \( A \cdot B \) adalah **dot product** antara vektor \( A \) dan \( B \).  
- \( \|A\| \) adalah **norma (panjang)** vektor \( A \).  
- \( \|B\| \) adalah **norma (panjang)** vektor \( B \).  

Norma dari masing-masing vektor dihitung sebagai berikut:  

$$
\|A\| = \sqrt{\sum_{i=1}^{n} A_i^2}, \quad \|B\| = \sqrt{\sum_{i=1}^{n} B_i^2}
$$

Sehingga rumus lengkapnya menjadi:  

$$
\cos(\theta) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Nilai **Cosine Similarity** berkisar antara **0** (tidak mirip) hingga **1** (sangat mirip).

#### **Kelebihan CBF dengan teknik Cosine Similarity**  
1. **Personalisasi tinggi** – Rekomendasi lebih sesuai dengan preferensi pengguna.  
2. **Tidak bergantung pada pengguna lain** – Bisa bekerja meskipun hanya ada satu pengguna (tidak memerlukan data kolektif).  
3. **Menangani cold start untuk item baru** – Bisa merekomendasikan item baru selama deskripsinya tersedia.  

#### **Kekurangan CBF dengan teknik Cosine Similarity**  
1. **Cold start untuk pengguna baru** – Sulit memberikan rekomendasi jika tidak ada riwayat interaksi pengguna.  
2. **Over-specialization** – Rekomendasi cenderung terbatas pada jenis item yang mirip dengan yang sudah dipilih pengguna sebelumnya.  
3. **Ketergantungan pada kualitas fitur** – Jika fitur yang digunakan kurang representatif, hasil rekomendasi bisa kurang akurat.  

Hasil cosine similarity
![cosine_similarity](https://github.com/user-attachments/assets/dddac69b-8a01-4bd4-806b-e25808d83d89)

### Recomendation Function
Fungsi `recommend()` digunakan untuk memberikan rekomendasi hotel berdasarkan tipe kamar (`roomtype`), negara (`country`), dan kota (`city`). Prosesnya diawali dengan normalisasi input untuk memastikan pencocokan tidak terpengaruh oleh perbedaan kapitalisasi huruf. Setelah itu, dataset difilter untuk menemukan hotel yang sesuai dengan kriteria pengguna. Jika tidak ditemukan hasil yang cocok, fungsi akan mengembalikan pesan bahwa tidak ada hotel yang sesuai.  

Selanjutnya, jika ada hotel yang cocok, fungsi akan mencari hotel lain yang memiliki kemiripan tinggi berdasarkan **Cosine Similarity**. Setiap hotel yang cocok akan dicari indeksnya dalam matriks kemiripan, lalu daftar hotel yang memiliki nilai kemiripan tertinggi diurutkan dari yang paling relevan. Hanya sejumlah `num_recommendations` hotel teratas yang ditampilkan. Jika tidak ada hotel serupa yang ditemukan, fungsi akan mengembalikan pesan bahwa tidak ada rekomendasi yang tersedia.  

Penggunaan **Cosine Similarity** dalam fungsi ini bertujuan untuk mengukur kesamaan antara hotel berdasarkan fitur yang tersedia, seperti fasilitas kamar dan kategori hotel. Dengan metode ini, sistem dapat memberikan rekomendasi hotel yang memiliki karakteristik serupa dengan hotel pilihan pengguna. Semakin tinggi nilai **Cosine Similarity**, semakin mirip hotel tersebut dengan preferensi pengguna, sehingga rekomendasi menjadi lebih relevan.

#### Recommendation Result
Menguji fungsi `recommend()` dengan mencari rekomendasi hotel dengan tipe kamar 'double room', negara 'italy' dan kota 'venice'. Berikut adalah hasilnya:

| hotelname            | roomtype               | country | city   | similarity |
|----------------------|-----------------------|---------|--------|------------|
| hotel antica fenice | double room           | italy   | venice | 1.00       |
| hotel dolomiti      | double or twin room   | italy   | venice | 0.93       |
| hotel bartolomeo    | double or twin room   | italy   | venice | 0.93       |
| hotel moresco       | double or twin room   | italy   | venice | 0.93       |
| hotel tivoli        | double or twin room   | italy   | venice | 0.93       |
| hotel falier        | double or twin room   | italy   | venice | 0.93       |
| hotel malibran      | standard double room  | italy   | venice | 0.93       |
| hotel bartolomeo    | standard double room  | italy   | venice | 0.93       |
| hotel malibran      | superior double room  | italy   | venice | 0.91       |
| hotel moresco       | superior double room  | italy   | venice | 0.91       |

Kemudian diuji lagi dengan mencari rekomendasi hotel dengan tipe kamar 'single', negara 'Spain' dan kota 'Barcelona'. Berikut adalah hasilnya:

| hotelname              | roomtype               | country | city       | similarity |
|------------------------|-----------------------|---------|------------|------------|
| hotel viladomat       | single                | spain   | barcelona  | 1.00       |
| hotel ciutat vella    | single                | spain   | barcelona  | 1.00       |
| astoria hotel        | single                | spain   | barcelona  | 1.00       |
| abba sants          | single                | spain   | barcelona  | 0.98       |
| hotel teatre auditori | single room           | spain   | barcelona  | 0.98       |
| magatzem 128        | interior single room   | spain   | barcelona  | 0.96       |
| hotel lleo          | single standard       | spain   | barcelona  | 0.93       |
| astoria hotel       | standard single       | spain   | barcelona  | 0.93       |
| moderno hotel       | single standard       | spain   | barcelona  | 0.93       |
| hotel onix liceo    | twin single room      | spain   | barcelona  | 0.92       |

Dapat diketahui, fungsi rekomendasi memberikan respon Top-N hotel yang memiliki kesamaan fitur tadi.

## Evaluation
Dalam sistem rekomendasi hotel dengan metode CBF ini menggunakan metrix evaluasi Precision@K. Precision@K adalah salah satu metrik evaluasi dalam content-based filtering (CBF) yang digunakan untuk mengukur seberapa relevan hasil rekomendasi dibandingkan dengan kebutuhan pengguna. Precision@K menghitung proporsi rekomendasi yang benar-benar relevan (True Positives, TP) terhadap total jumlah rekomendasi yang diberikan hingga K hasil teratas.  

Rumus precision@K dinyatakan sebagai berikut:

$$
\text{Precision@K} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

Dalam konteks rekomendasi hotel, **True Positives (TP)** adalah hotel yang relevan dengan kebutuhan pengguna berdasarkan tipe kamar yang diinginkan, sementara **False Positives (FP)** adalah hotel yang tidak sesuai tetapi tetap direkomendasikan.  

Berdasarkan hasil rekomendasi untuk kota **Venice**, seluruh 10 hotel yang diberikan dianggap relevan (**TP = 10, FP = 0**), sehingga precision@10 dihitung sebagai berikut:

$$
\text{Precision@10} = \frac{10}{10 + 0} = \frac{10}{10} = 1.0
$$

Pada hasil rekomendasi untuk kota **Barcelona**, seluruh 10 hotel yang diberikan dianggap relevan (**TP = 10, FP = 0**), sehingga hasil precision@10 nya adalah 1.0.
Hasil precision@10 sebesar **1.0 (100%)** menunjukkan bahwa sistem rekomendasi bekerja dengan sangat baik untuk skenario ini, karena semua hasil yang diberikan relevan dengan permintaan pengguna.

---

## Conclusion
Proyek ini berhasil membangun sistem rekomendasi hotel berbasis **Content-Based Filtering (CBF)** untuk membantu pengguna menemukan hotel yang sesuai dengan preferensi mereka secara lebih cepat dan akurat. Sistem ini menggunakan **TF-IDF** untuk mengubah data teks menjadi vektor numerik dan **Cosine Similarity** untuk mengukur kemiripan antar hotel. Evaluasi menggunakan **Precision@K** menunjukkan bahwa rekomendasi yang dihasilkan memiliki tingkat relevansi yang tinggi, dengan **Precision@10 mencapai 1.0** pada dataset yang diuji. Pendekatan ini terbukti efektif dalam menyaring pilihan hotel berdasarkan kemiripan fitur, sehingga meningkatkan pengalaman pengguna dalam menemukan akomodasi yang sesuai dengan kebutuhan mereka.
