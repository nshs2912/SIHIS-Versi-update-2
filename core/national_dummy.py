# core/national_dummy.py
"""
Generator data simulasi nasional realistis untuk SI-HIS Intelligence.
Kompatibel dengan:
  - core/data_provider.py (via generate_national_dummy)
  - core/analytics.py / app.py (via generate_data_simulasi)
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==============================================================================
# 1. MASTER DATA WILAYAH ADMINISTRASI & BOUNDING BOX KOORDINAT
# ==============================================================================
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

DISEASE_LIST = ["Demam Berdarah Dengue", "Tuberkulosis", "Diare Akut", "ISPA", "Typhoid Abdominalis", "Malaria", "Campak"]
AGE_GROUPS = ["<1", "1-4", "5-9", "10-14", "15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "≥85"]
GENDER_LIST = ["Laki-laki", "Perempuan"]
JOB_LIST = ["Belum Bekerja", "Pelajar/Mahasiswa", "Pegawai Negeri", "Swasta", "Wiraswasta", "Petani/Nelayan", "Lainnya"]
COMORBID_LIST = ["Tidak Ada", "Hipertensi", "Diabetes Melitus", "Penyakit Jantung", "Ginjal Kronis", "Obesitas"]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_random_coordinate(provinsi, kabupaten):
    region = REGION_MASTER.get(provinsi, {}).get(kabupaten)
    if not region:
        return np.random.uniform(-5.0, 5.0), np.random.uniform(95.0, 140.0)
    lat = np.random.uniform(region['lat'][0], region['lat'][1])
    lon = np.random.uniform(region['lon'][0], region['lon'][1])
    return round(lat, 6), round(lon, 6)

def generate_date_with_outbreak_signal(n_days=365):
    base_date = datetime.now() - timedelta(days=n_days)
    if random.random() < 0.70:
        days_offset = random.randint(0, n_days)
    else:
        days_offset = random.randint(60, 90)
    return (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')

def assign_realistic_comorbidity(age_group):
    if age_group in ["<1", "1-4", "5-9", "10-14", "15-19", "20-24"]:
        return random.choices(COMORBID_LIST, weights=[0.90, 0.02, 0.02, 0.02, 0.02, 0.02])[0]
    elif age_group in ["65-74", "75-84", "≥85"]:
        return random.choices(COMORBID_LIST, weights=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05])[0]
    else:
        return random.choices(COMORBID_LIST, weights=[0.70, 0.10, 0.08, 0.05, 0.04, 0.03])[0]

# ==============================================================================
# 3. MAIN GENERATOR (NAMA UTAMA: generate_data_simulasi)
# ==============================================================================
def generate_data_simulasi(n_samples=5000):
    """Generator data simulasi realistis dengan sinyal epidemiologi."""
    data = []
    prov_weights = {"DKI Jakarta": 0.2, "Jawa Barat": 0.4, "Jawa Timur": 0.25, "Sumatera Utara": 0.15}
    provinces = list(REGION_MASTER.keys())
    
    for _ in range(n_samples):
        prov = random.choices(provinces, weights=[prov_weights[p] for p in provinces])[0]
        kab = random.choice(list(REGION_MASTER[prov].keys()))
        pusk_list = REGION_MASTER[prov][kab]["puskesmas"]
        pusk = random.choice(pusk_list)
        
        lat, lon = get_random_coordinate(prov, kab)
        tgl_sakit = generate_date_with_outbreak_signal()
        
        umur = random.choices(AGE_GROUPS, weights=[0.02, 0.05, 0.08, 0.08, 0.10, 0.10, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])[0]
        gender = random.choice(GENDER_LIST)
        pekerjaan = random.choice(JOB_LIST)
        komorbid = assign_realistic_comorbidity(umur)
        
        if prov == "Jawa Barat" and random.random() < 0.40:
            penyakit = "Demam Berdarah Dengue"
        else:
            penyakit = random.choice(DISEASE_LIST)
            
        base_mortality = 0.01
        if komorbid != "Tidak Ada": base_mortality += 0.03
        if umur in ["65-74", "75-84", "≥85"]: base_mortality += 0.04
        meninggal = "Ya" if random.random() < base_mortality else "Tidak"
        
        data.append({
            "Provinsi": prov,
            "Kabupaten": kab,
            "Kecamatan": f"Kecamatan Simulasi {random.randint(1, 10)}",
            "Desa/Kelurahan": f"Kelurahan Simulasi {random.randint(1, 20)}",
            "Puskesmas": pusk,
            "Tanggal Sakit": tgl_sakit,
            "Umur": umur,
            "Jenis Kelamin": gender,
            "Pekerjaan": pekerjaan,
            "Status Komorbid": komorbid,
            "Diagnosis Konfirm": penyakit if random.random() > 0.3 else "",
            "Diagnosis Probabel": penyakit if random.random() > 0.7 else "",
            "Diagnosis Suspek": "",
            "Latitude": lat,
            "Longitude": lon,
            "Meninggal": meninggal
        })
        
    df = pd.DataFrame(data)
    df['Tanggal Sakit'] = pd.to_datetime(df['Tanggal Sakit'])
    df = df.sort_values('Tanggal Sakit').reset_index(drop=True)
    return df

# ==============================================================================
# 4. BACKWARD COMPATIBILITY ALIAS (PENTING!)
# ==============================================================================
# Alias agar core/data_provider.py tidak error saat import
generate_national_dummy = generate_data_simulasi

# Alias tambahan untuk kompatibilitas dengan berbagai variasi nama
generate_national_data = generate_data_simulasi
generate_dummy_data = generate_data_simulasi

# ==============================================================================
# 5. TESTING
# ==============================================================================
if __name__ == "__main__":
    print("Testing generate_data_simulasi...")
    df1 = generate_data_simulasi(100)
    print(f"  Rows: {len(df1)}")
    print(f"  Columns: {list(df1.columns)}")
    
    print("\nTesting generate_national_dummy (alias)...")
    df2 = generate_national_dummy(100)
    print(f"  Rows: {len(df2)}")
    print(f"  Same function? {generate_national_dummy is generate_data_simulasi}")
