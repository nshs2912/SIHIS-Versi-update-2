# core/national_dummy.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==============================================================================
# 1. MASTER DATA WILAYAH ADMINISTRASI & BOUNDING BOX KOORDINAT (Subset Nyata)
# ==============================================================================
# Format: Provinsi -> Kabupaten/Kota -> (Min_Lat, Max_Lat, Min_Lon, Max_Lon)
# Catatan: Dalam produksi, ini bisa diganti dengan load dari CSV BPS/Kemendagri penuh.
REGION_MASTER = {
    "DKI Jakarta": {
        "Kota Adm. Jakarta Pusat": {"lat": (-6.20, -6.15), "lon": (106.80, 106.88), "puskesmas": ["Puskesmas Kecamatan Gambir", "Puskesmas Kecamatan Menteng"]},
        "Kota Adm. Jakarta Selatan": {"lat": (-6.30, -6.20), "lon": (106.75, 106.85), "puskesmas": ["Puskesmas Kecamatan Kebayoran Baru", "Puskesmas Kecamatan Tebet"]},
    },
    "Jawa Barat": {
        "Kab. Bandung": {"lat": (-7.15, -6.85), "lon": (107.40, 107.90), "puskesmas": ["Puskesmas Baleendah", "Puskesmas Soreang", "Puskesmas Cicalengka"]},
        "Kota Bandung": {"lat": (-6.98, -6.85), "lon": (107.50, 107.75), "puskesmas": ["Puskesmas Coblong", "Puskesmas Bandung Kulon"]},
        "Kab. Bekasi": {"lat": (-6.40, -6.25), "lon": (106.90, 107.20), "puskesmas": ["Puskesmas Cikarang Pusat", "Puskesmas Tambun"]},
    },
    "Jawa Timur": {
        "Kota Surabaya": {"lat": (-7.35, -7.15), "lon": (112.60, 112.85), "puskesmas": ["Puskesmas Genteng", "Puskesmas Rungkut"]},
        "Kab. Malang": {"lat": (-8.10, -7.80), "lon": (112.40, 112.80), "puskesmas": ["Puskesmas Kepanjen", "Puskesmas Lawang"]},
    },
    "Sumatera Utara": {
        "Kota Medan": {"lat": (3.50, 3.70), "lon": (98.60, 98.80), "puskesmas": ["Puskesmas Medan Kota", "Puskesmas Medan Area"]},
        "Kab. Deli Serdang": {"lat": (3.40, 3.60), "lon": (98.70, 98.90), "puskesmas": ["Puskesmas Lubuk Pakam", "Puskesmas Tanjung Morawa"]},
    }
}

DISEASE_LIST = [
    "Demam Berdarah Dengue", "Tuberkulosis", "Diare Akut", 
    "ISPA", "Typhoid Abdominalis", "Malaria", "Campak"
]

AGE_GROUPS = ["<1", "1-4", "5-9", "10-14", "15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "≥85"]
GENDER_LIST = ["Laki-laki", "Perempuan"]
JOB_LIST = ["Belum Bekerja", "Pelajar/Mahasiswa", "Pegawai Negeri", "Swasta", "Wiraswasta", "Petani/Nelayan", "Lainnya"]
COMORBID_LIST = ["Tidak Ada", "Hipertensi", "Diabetes Melitus", "Penyakit Jantung", "Ginjal Kronis", "Obesitas"]

# ==============================================================================
# 2. FUNGSI HELPER GENERATOR
# ==============================================================================

def get_random_coordinate(provinsi, kabupaten):
    """Menghasilkan koordinat Lat/Lon yang VALID di dalam batas wilayah kabupaten sesungguhnya."""
    region = REGION_MASTER.get(provinsi, {}).get(kabupaten)
    if not region:
        # Fallback acak jika wilayah tidak terdaftar di master subset
        return np.random.uniform(-5.0, 5.0), np.random.uniform(95.0, 140.0)
    
    lat = np.random.uniform(region['lat'][0], region['lat'][1])
    lon = np.random.uniform(region['lon'][0], region['lon'][1])
    return round(lat, 6), round(lon, 6)

def generate_date_with_outbreak_signal(n_days=365):
    """Menghasilkan tanggal dengan sinyal musiman dan lonjakan wabah (outbreak injection)."""
    base_date = datetime.now() - timedelta(days=n_days)
    
    # 70% data tersebar normal (background noise)
    if random.random() < 0.70:
        days_offset = random.randint(0, n_days)
    # 30% data disuntikkan sebagai "Outbreak Cluster" (misal: puncak DBD bulan ke-2 dan ke-3)
    else:
        # Simulasi puncak wabah sekitar 60-90 hari yang lalu
        days_offset = random.randint(60, 90) 
        
    return (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')

def assign_realistic_comorbidity(age_group):
    """Komorbiditas lebih mungkin terjadi pada kelompok usia lanjut."""
    if age_group in ["<1", "1-4", "5-9", "10-14", "15-19", "20-24"]:
        return random.choices(COMORBID_LIST, weights=[0.90, 0.02, 0.02, 0.02, 0.02, 0.02])[0]
    elif age_group in ["65-74", "75-84", "≥85"]:
        return random.choices(COMORBID_LIST, weights=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05])[0]
    else:
        return random.choices(COMORBID_LIST, weights=[0.70, 0.10, 0.08, 0.05, 0.04, 0.03])[0]

# ==============================================================================
# 3. MAIN GENERATOR
# ==============================================================================

def generate_data_simulasi(n_samples=5000):
    """
    Menghasilkan dataset sintetis nasional yang realistis untuk pengujian pipeline SI-HIS.
    Memuat: Nama wilayah asli, koordinat valid, pola musiman, dan injeksi sinyal wabah.
    """
    data = []
    
    # Bobot probabilitas pemilihan provinsi (Jawa Barat lebih padat sebagai contoh)
    prov_weights = {"DKI Jakarta": 0.2, "Jawa Barat": 0.4, "Jawa Timur": 0.25, "Sumatera Utara": 0.15}
    provinces = list(REGION_MASTER.keys())
    
    for _ in range(n_samples):
        # 1. Pilih Wilayah
        prov = random.choices(provinces, weights=[prov_weights[p] for p in provinces])[0]
        kab = random.choice(list(REGION_MASTER[prov].keys()))
        pusk_list = REGION_MASTER[prov][kab]["puskesmas"]
        pusk = random.choice(pusk_list)
        
        # 2. Generate Koordinat Valid
        lat, lon = get_random_coordinate(prov, kab)
        
        # 3. Generate Tanggal dengan Sinyal Epidemiologi
        tgl_sakit = generate_date_with_outbreak_signal()
        
        # 4. Generate Demografi & Klinis
        umur = random.choices(AGE_GROUPS, weights=[0.02, 0.05, 0.08, 0.08, 0.10, 0.10, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])[0]
        gender = random.choice(GENDER_LIST)
        pekerjaan = random.choice(JOB_LIST)
        komorbid = assign_realistic_comorbidity(umur)
        
        # 5. Generate Diagnosis (Suntikkan DBD lebih tinggi di Jawa Barat untuk simulasi outbreak)
        if prov == "Jawa Barat" and random.random() < 0.40:
            penyakit = "Demam Berdarah Dengue"
        else:
            penyakit = random.choice(DISEASE_LIST)
            
        # 6. Generate Outcome (Meninggal)
        # CFR dasar 1%, tapi naik jika ada komorbid atau usia lanjut
        base_mortality = 0.01
        if komorbid != "Tidak Ada": base_mortality += 0.03
        if umur in ["65-74", "75-84", "≥85"]: base_mortality += 0.04
        
        meninggal = "Ya" if random.random() < base_mortality else "Tidak"
        
        # 7. Susun ke dalam baris
        data.append({
            "Provinsi": prov,
            "Kabupaten": kab,
            "Kecamatan": f"Kecamatan Simulasi {random.randint(1, 10)}", # Bisa diperluas dengan master asli
            "Desa/Kelurahan": f"Kelurahan Simulasi {random.randint(1, 20)}",
            "Puskesmas": pusk,
            "Tanggal Sakit": tgl_sakit,
            "Umur": umur,
            "Jenis Kelamin": gender,
            "Pekerjaan": pekerjaan,
            "Status Komorbid": komorbid,
            "Diagnosis Konfirm": penyakit if random.random() > 0.3 else "",
            "Diagnosis Probabel": penyakit if random.random() > 0.7 else "",
            "Diagnosis Suspek": "" ,
            "Latitude": lat,
            "Longitude": lon,
            "Meninggal": meninggal
        })
        
    df = pd.DataFrame(data)
    
    # Pastikan tipe data sesuai
    df['Tanggal Sakit'] = pd.to_datetime(df['Tanggal Sakit'])
    df = df.sort_values('Tanggal Sakit').reset_index(drop=True)
    
    return df

# Untuk pengujian lokal
if __name__ == "__main__":
    print("Generating realistic dummy data...")
    df_test = generate_data_simulasi(1000)
    print(df_test[['Provinsi', 'Kabupaten', 'Latitude', 'Longitude', 'Diagnosis Konfirm', 'Tanggal Sakit']].head(10))
    print(f"\nTotal records: {len(df_test)}")
    print(f"Unique Provinces: {df_test['Provinsi'].unique()}")
