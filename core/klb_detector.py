# core/klb_detector.py
"""
Deteksi sinyal KLB dan Wabah berdasarkan:
- Permenkes No. 1501/Menkes/Per/X/2010 (KLB)
- Permenkes No. 1 Tahun 2026 (Wabah sebagai eskalasi KLB)

PENTING: Output adalah SINYAL PROBABILISTIK, bukan penetapan status resmi.
Penetapan KLB/Wabah adalah kewenangan otoritas kesehatan (Dinkes / Menkes).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def compute_temporal_series(df, date_col='Tanggal Sakit', freq='W'):
    """Agregasi kasus per periode (mingguan default)."""
    if df is None or df.empty or date_col not in df:
        return pd.Series(dtype=int)
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
    work = work.dropna(subset=[date_col])
    if work.empty:
        return pd.Series(dtype=int)
    return work.set_index(date_col).resample(freq).size()

def compute_spatial_spread_rate(df, region_col='Kabupaten', date_col='Tanggal Sakit', window_days=7):
    """
    Menghitung laju penyebaran spasial: jumlah wilayah BARU yang melaporkan kasus
    dalam window_days terakhir. Ini adalah proksi algoritmik untuk "menyebar cepat".
    """
    if df is None or df.empty or region_col not in df or date_col not in df:
        return {'new_regions_7d': 0, 'total_active_regions': 0, 'spread_rate': 0.0}
    
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
    work = work.dropna(subset=[date_col])
    if work.empty:
        return {'new_regions_7d': 0, 'total_active_regions': 0, 'spread_rate': 0.0}
    
    max_date = work[date_col].max()
    recent = work[work[date_col] >= (max_date - pd.Timedelta(days=window_days))]
    previous = work[work[date_col] < (max_date - pd.Timedelta(days=window_days))]
    
    recent_regions = set(recent[region_col].dropna().unique())
    previous_regions = set(previous[region_col].dropna().unique())
    
    new_regions = recent_regions - previous_regions
    
    return {
        'new_regions_7d': len(new_regions),
        'total_active_regions': len(recent_regions),
        'spread_rate': len(new_regions) / max(len(previous_regions), 1),
        'new_regions_list': sorted(list(new_regions))
    }

def detect_klb_signals(current_series: pd.Series, historical_series: pd.Series = None) -> Dict:
    """
    Mengevaluasi 7 kriteria KLB berdasarkan data deret waktu.
    
    Input:
      - current_series: deret waktu kasus periode berjalan (index=periode, value=jumlah kasus)
      - historical_series: deret waktu kasus periode sama tahun lalu (untuk perbandingan)
    
    Output: dict dengan 'criteria_met' (list), 'severity' ('none'|'potential_klb'|'strong_klb'),
            dan 'details' untuk setiap kriteria.
    """
    result = {
        'criteria_met': [],
        'severity': 'none',
        'details': {},
        'data_adequate': False
    }
    
    if current_series is None or len(current_series) < 3:
        result['details']['note'] = 'Deret waktu terlalu pendek (minimal 3 periode) untuk evaluasi KLB.'
        return result
    
    result['data_adequate'] = True
    current_vals = current_series.values
    n = len(current_vals)
    
    # Kriteria 2: 3 periode berturut-turut meningkat
    if n >= 3:
        last3 = current_vals[-3:]
        consecutive_increase = all(last3[i] < last3[i+1] for i in range(2))
        result['details']['k2_consecutive_3_periods'] = {
            'met': consecutive_increase,
            'values': last3.tolist()
        }
        if consecutive_increase:
            result['criteria_met'].append('K2: Peningkatan 3 periode berturut-turut')
    
    # Kriteria 3: Kenaikan ≥2x lipat dibanding periode sebelumnya
    if n >= 2 and current_vals[-2] > 0:
        ratio = current_vals[-1] / current_vals[-2]
        result['details']['k3_ratio_vs_previous'] = {'met': ratio >= 2.0, 'ratio': round(ratio, 2)}
        if ratio >= 2.0:
            result['criteria_met'].append(f'K3: Kenaikan {ratio:.1f}x lipat vs periode sebelumnya')
    
    # Kriteria 4 & 5: Perbandingan dengan tahun lalu (jika tersedia)
    if historical_series is not None and len(historical_series) > 0:
        avg_last_year = historical_series.mean()
        current_mean = current_vals[-4:].mean() if n >= 4 else current_vals.mean()
        
        if avg_last_year > 0:
            ratio_yearly = current_mean / avg_last_year
            result['details']['k4_ratio_vs_last_year'] = {
                'met': ratio_yearly >= 2.0, 
                'ratio': round(ratio_yearly, 2)
            }
            if ratio_yearly >= 2.0:
                result['criteria_met'].append(f'K4/K5: Rata-rata {ratio_yearly:.1f}x lipat vs tahun lalu')
    
    # Kriteria 6: CFR naik ≥50% (dihitung terpisah, butuh data mortalitas)
    # Akan diisi oleh caller jika data mortalitas tersedia
    
    # Severity classification
    n_criteria = len(result['criteria_met'])
    if n_criteria >= 2:
        result['severity'] = 'strong_klb'
    elif n_criteria == 1:
        result['severity'] = 'potential_klb'
    
    return result

def detect_wabah_escalation(klb_result: Dict, rt_value: float, spread_rate: Dict, 
                             cfr_change_pct: Optional[float] = None) -> Dict:
    """
    Mengevaluasi sinyal eskalasi KLB → Wabah berdasarkan Permenkes No. 1 Tahun 2026.
    
    Indikator Wabah:
      1. Eskalasi dari status KLB (klb_result severity = strong_klb)
      2. Penyebaran cepat (spread_rate tinggi, banyak wilayah baru)
      3. Rₜ konsisten tinggi (>1.2)
      4. CFR meningkat signifikan (opsional, jika data tersedia)
    
    PENTING: Fungsi ini TIDAK menetapkan status Wabah. Hanya memberikan sinyal peringatan.
    """
    result = {
        'escalation_signal': False,
        'signal_strength': 'none',  # none | moderate | strong
        'indicators': [],
        'legal_note': 'Penetapan Wabah adalah kewenangan eksklusif Menteri Kesehatan (Permenkes 1/2026).'
    }
    
    # Indikator 1: KLB sudah kuat
    if klb_result.get('severity') == 'strong_klb':
        result['indicators'].append('Status KLB kuat terdeteksi (≥2 kriteria terpenuhi)')
    
    # Indikator 2: Rₜ tinggi dan konsisten
    if rt_value > 1.2:
        result['indicators'].append(f'Rₜ = {rt_value:.2f} (>1.2, transmisi ekspansif)')
    
    # Indikator 3: Penyebaran spasial cepat
    if spread_rate.get('new_regions_7d', 0) >= 3:
        result['indicators'].append(
            f"Penyebaran cepat: {spread_rate['new_regions_7d']} wilayah baru terdampak dalam 7 hari"
        )
    if spread_rate.get('spread_rate', 0) > 0.5:
        result['indicators'].append(
            f"Laju ekspansi wilayah: {spread_rate['spread_rate']*100:.0f}% dari baseline"
        )
    
    # Indikator 4: CFR meningkat
    if cfr_change_pct is not None and cfr_change_pct >= 50:
        result['indicators'].append(f'CFR meningkat {cfr_change_pct:.0f}% (K6 KLB terpenuhi)')
    
    # Klasifikasi kekuatan sinyal
    n_indicators = len(result['indicators'])
    if n_indicators >= 3 and klb_result.get('severity') == 'strong_klb':
        result['escalation_signal'] = True
        result['signal_strength'] = 'strong'
    elif n_indicators >= 2:
        result['escalation_signal'] = True
        result['signal_strength'] = 'moderate'
    
    return result
