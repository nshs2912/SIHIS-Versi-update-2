import warnings
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from core.analytics import REQUIRED_COLUMNS, generate_excel_template, generate_data_simulasi, get_wib_time
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope

warnings.filterwarnings('ignore')
st.set_page_config(page_title='SI-HIS Intelligence', page_icon='🧠', layout='wide')

st.markdown('''
<div style="background:#f0e68c; padding:18px 22px; border-radius:12px; border:2px solid #d4c886;">
    <h1 style="margin:0; color:#1e3a8a; font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1>
    <p style="margin:6px 0 0; color:#475569; font-weight:600;">Early Detection, Smarter Intervention</p>
</div>
''', unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")

engine = SIHISIntelligenceEngine()

def _fmt_number(v, digits=4):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return 'Tidak dapat dihitung'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.{digits}f}'
    return str(v)

def show_resume(text, detail='tabel'):
    st.info(text or 'Belum tersedia interpretasi untuk scope/data ini.')
    st.caption(f"Untuk lebih detail bisa dilihat pada {detail} di bawah ini.")

def render_value(value, title=None):
    if title: st.markdown(f'#### {title}')
    if isinstance(value, pd.DataFrame):
        if value.empty: st.info('Belum ada data untuk ditampilkan.')
        else: st.dataframe(value, use_container_width=True, hide_index=True)
        return
    if isinstance(value, dict):
        if not value: st.info('Belum ada hasil.'); return
        for key, item in value.items():
            st.markdown(f'**{key}**')
            if isinstance(item, pd.DataFrame):
                if item.empty: st.caption('Tidak ada hasil yang dapat dihitung.')
                else: st.dataframe(item, use_container_width=True, hide_index=True)
            elif isinstance(item, dict): render_value(item)
            elif isinstance(item, list) and item and all(isinstance(x, dict) for x in item):
                st.dataframe(pd.DataFrame(item), use_container_width=True, hide_index=True)
            elif item is None or (isinstance(item, (float, np.floating)) and np.isnan(item)):
                st.caption('Tidak dapat dihitung dari data yang tersedia.')
            else: st.write(item)
        return
    if isinstance(value, list):
        if not value: st.info('Tidak ada temuan pada bagian ini.')
        elif all(isinstance(x, dict) for x in value):
            st.dataframe(pd.DataFrame(value), use_container_width=True, hide_index=True)
        else: st.write(value)
        return
    if value is None: st.info('Belum tersedia untuk scope/data ini.')
    else: st.write(value)

def risk_summary(risk):
    if not isinstance(risk, dict): return pd.DataFrame()
    rows = []
    for var, obj in risk.items():
        if var.startswith('MULTIVARIAT') or not isinstance(obj, dict): continue
        p = obj.get('p_value')
        valid = p is not None and pd.notna(p)
        sig = bool(valid and float(p) < .05)
        rows.append({
            'Faktor': var,
            'p-value': round(float(p), 4) if valid else None,
            'Interpretasi': 'Ada asosiasi statistik (p<0,05)' if sig else ('Tidak ada bukti asosiasi statistik pada α=0,05' if valid else 'Uji tidak dapat dihitung')
        })
    return pd.DataFrame(rows)

# ==============================================================================
# PILAR 1: EXPERT VOICE NARRATIVES (Rigorous, Cautious, Scientifically Accurate)
# ==============================================================================

def descriptive_narrative(result, label):
    ov = result.get('overview', {}) if isinstance(result, dict) else {}
    total = int(ov.get('total_cases', 0) or 0)
    deaths = int(ov.get('deaths', 0) or 0)
    cfr = (deaths / total * 100) if total > 0 else 0.0
    
    parts = [f"Beban penyakit di **{label}** mencatat **{total:,} morbiditas** dan **{deaths:,} mortalitas** (CFR kasar: **{cfr:.2f}%**)."]
    
    top = result.get('top10_diseases') if isinstance(result, dict) else None
    if isinstance(top, pd.DataFrame) and not top.empty:
        r = top.iloc[0]
        disease_name = r.get('Nama Penyakit', '-')
        cases = int(r.get('Jumlah Kasus', 0))
        cfr_top = float(r.get('CFR', 0))
        parts.append(f"Dominasi kasus oleh **{disease_name}** ({cases:,} kasus). ")
        if cfr_top > 5.0:
            # Koreksi: CFR > 5% bukan otomatis keganasan tinggi, bisa karena bias pelaporan, denominator, atau struktur umur.
            parts.append(f"CFR sebesar {cfr_top:.2f}% merupakan sinyal yang memerlukan investigasi lebih lanjut. Angka ini dapat mencerminkan keganasan klinis, namun juga sangat rentan terhadap bias pelaporan (underreporting kasus ringan), kelengkapan data outcome, atau struktur umur populasi yang rentan. Validasi terhadap denominator populasi berisiko sangat diperlukan.")
            
    if isinstance(result.get('province_distribution'), pd.DataFrame) and not result['province_distribution'].empty:
        p = result['province_distribution'].iloc[0]
        parts.append(f"Secara geografis, konsentrasi kasus tertinggi tercatat di **{p.get('Provinsi', '-')}** ({int(p.get('Jumlah Kasus', 0)):,} kasus).")
        
    parts.append("Temuan ini bersifat deskriptif. Interpretasi risiko yang valid memerlukan denominator populasi berisiko untuk menghitung *Attack Rate* atau *Incidence Rate*, bukan hanya mengandalkan jumlah kasus absolut.")
    return ' '.join(parts)

def ews_narrative(ews, rt, mortality_trend=None):
    """
    Narasi Early Warning System (EWS) yang mengacu pada Permenkes No. 1 Tahun 2026 
    tentang KLB, Wabah, dan Krisis Kesehatan.
    """
    if not isinstance(ews, dict):
        return "Data deret waktu belum memenuhi syarat minimum (panjang seri dan kelengkapan) untuk evaluasi kriteria KLB/Wabah."
    
    score = ews.get('ews_score', ews.get('score', ews.get('EWS', 0)))
    trend = ews.get('trend_pct', ews.get('trend', 0))
    rt_val = float(ews.get('rt_recent', 1.0)) if isinstance(ews.get('rt_recent'), (int, float)) else 1.0
    
    text = "Sistem mendeteksi anomali temporal yang dievaluasi berdasarkan indikator kewaspadaan dini. "
    
    # 1. Evaluasi Sinyal KLB
    criteria_met = []
    if trend >= 100:
        criteria_met.append("peningkatan kasus ≥2 kali lipat dibanding periode sebelumnya")
    if trend > 0:
        criteria_met.append("tren peningkatan kasus berturut-turut")
        
    if criteria_met:
        text += f"Sinyal sistem menunjukkan potensi pemenuhan kriteria **Kejadian Luar Biasa (KLB)**, yaitu: {', '.join(criteria_met)}. "
    else:
        text += "Saat ini, tren kasus belum menunjukkan pola yang secara otomatis memenuhi ambang batas kriteria KLB berdasarkan data yang tersedia. "

    # 2. Evaluasi Sinyal Eskalasi menuju Wabah (Berdasarkan Permenkes No. 1 Tahun 2026)
    escalation_warning = False
    if rt_val > 1.2 and trend > 50: # Ambang batas heuristik untuk "menyebar cepat"
        escalation_warning = True
        
    if escalation_warning:
        text += "⚠️ **PERINGATAN ESKALASI**: Kombinasi antara akselerasi kasus yang tinggi dan nilai Rₜ yang konsisten di atas 1 mengindikasikan pola penyebaran yang cepat dan meluas. Secara epidemiologis, ini merupakan **sinyal peringatan dini eskalasi KLB menuju status Wabah**."
    else:
        text += f"Estimasi Rₜ terkini berada di angka **{rt_val:.2f}**. "
        if rt_val > 1.0:
            text += "Nilai Rₜ > 1 mengindikasikan *potensi* pertumbuhan kasus, namun belum tentu memenuhi kriteria eskalasi cepat yang disyaratkan untuk status Wabah. "

    # 3. Penegasan Batasan Metodologis dan Kewenangan Hukum (Rigor)
    text += "\n\n**Penting (Batasan Sistem & Regulasi):** "
    text += "1. Output ini adalah sinyal probabilistik berbasis data surveilans. Penetapan status **KLB** maupun **Wabah** tidak dapat dilakukan secara otomatis oleh algoritma. "
    text += "2. Berdasarkan **Permenkes Nomor 1 Tahun 2026**, penetapan status **'Wabah'** merupakan **kewenangan eksklusif Menteri Kesehatan** setelah memverifikasi eskalasi jumlah kasus/kematian yang signifikan dan kecepatan penyebaran di masyarakat. "
    text += "3. Sinyal ini wajib segera ditindaklanjuti dengan verifikasi lapangan, konfirmasi diagnosis, dan pelaporan berjenjang ke Dinas Kesehatan setempat untuk evaluasi penetapan status resmi."
    
    return text

def spatial_narrative(spatial):
    if not isinstance(spatial, pd.DataFrame) or spatial.empty:
        return "Data koordinat tidak memadai atau tidak valid untuk pemodelan kerapatan spasial."
    
    counts = spatial['Cluster'].value_counts() if 'Cluster' in spatial else pd.Series(dtype=int)
    clusters = counts.drop(index=-1, errors='ignore')
    noise = int(counts.get(-1, 0))
    
    # Koreksi: DBSCAN adalah density-based clustering, BUKAN uji autokorelasi spasial (seperti Moran's I)
    text = f"Algoritma DBSCAN mengidentifikasi **{len(clusters)} area dengan kepadatan kasus tinggi (cluster)** dan **{noise} titik yang tersebar (noise/outlier)**. "
    text += "**Penting:** DBSCAN hanyalah algoritma pengelompokan berbasis kepadatan (*density-based*), **bukan** uji statistik autokorelasi spasial (seperti Moran's I atau Getis-Ord Gi*). "
    
    if len(clusters) > 0:
        text += "Keberadaan cluster kepadatan tinggi menghasilkan hipotesis tentang kemungkinan transmisi lokal (*person-to-person*) atau paparan terhadap sumber infeksi bersama (*common source point/continuous source*) di wilayah tersebut. "
        text += "**Rekomendasi:** Hipotesis ini harus diverifikasi melalui investigasi epidemiologi lapangan (wawancara, *contact tracing*, dan pengambilan sampel lingkungan) untuk membedakan antara klaster transmisi aktif vs. klaster akibat pelaporan yang terkonsentrasi di satu fasilitas kesehatan."
    else:
        text += "Kasus tersebar secara acak (pola dispersi). Hal ini dapat mengindikasikan transmisi komunitas yang luas (*community transmission*) atau paparan lingkungan yang homogen, namun juga bisa merupakan artefak dari ketidakakuratan data koordinat."
    return text

def risk_narrative(risk):
    s = risk_summary(risk)
    if s.empty:
        return "Model statistik belum dapat menghasilkan estimasi yang valid pada subset data ini (kemungkinan karena ukuran sampel kecil, variasi data yang rendah, atau *zero-cell count*)."
    
    sig = s[s['p-value'].notna() & (s['p-value'] < .05)]
    if sig.empty:
        return "Pada ambang signifikansi α=0.05, belum ditemukan variabel dengan asosiasi statistik yang signifikan terhadap outcome. Namun, ketiadaan signifikansi statistik (bisa akibat *low statistical power* atau ukuran sampel kecil) tidak boleh diinterpretasikan sebagai bukti ketiadaan risiko biologis/epidemiologis."
    
    factors = ', '.join(sig['Faktor'].astype(str))
    # Koreksi: p < 0.05 saja tidak cukup. Perlu OR, CI 95%, arah asosiasi, dan confounding.
    text = f"Analisis bivariat menunjukkan adanya asosiasi statistik (p < 0.05) pada variabel: **{factors}**. "
    text += "**Peringatan Interpretasi:** Nilai p < 0.05 **saja tidak cukup** untuk menyimpulkan variabel tersebut sebagai 'faktor risiko'. Evaluasi wajib mencakup: "
    text += "1) Besaran efek dan arah asosiasi (Odds Ratio / Adjusted OR), "
    text += "2) Presisi estimasi (Interval Kepercayaan 95%), "
    text += "3) Kontrol terhadap variabel perancu (*confounding*) dalam model multivariat, dan "
    text += "4) Pertimbangan bias seleksi atau bias informasi dalam pengumpulan data. "
    text += "Asosiasi statistik tidak ekuivalen dengan hubungan kausal."
    return text

def render_risk_factors(risk):
    if not isinstance(risk, dict): render_value(risk); return
    for factor, obj in risk.items():
        st.markdown(f'#### {factor}')
        if isinstance(obj, pd.DataFrame): render_value(obj); continue
        if not isinstance(obj, dict): st.write(obj); continue
        p = obj.get('p_value'); chi = obj.get('chi2')
        c1, c2 = st.columns(2)
        c1.metric('Chi-square', _fmt_number(chi))
        c2.metric('p-value', _fmt_number(p))
        if p is not None and pd.notna(p):
            if float(p) < .05: 
                st.info('Terdapat asosiasi statistik (p<0,05). Besaran, arah, dan signifikansi klinis tetap harus dinilai dari Odds Ratio (OR) dan Interval Kepercayaan 95% (CI 95%).')
            else: 
                st.info('Tidak ada bukti asosiasi statistik pada α=0,05. Ini bukan bukti bahwa faktor tersebut tidak berpengaruh (pertimbangkan power analisis).')
        else: 
            st.warning('Uji Chi-square tidak dapat dihitung secara valid. Kemungkinan adanya sel dengan frekuensi nol (*zero-cell count*) atau variasi yang terlalu rendah. Pada kondisi data yang *sparse*, metode *Firth’s penalized likelihood* atau *Exact Logistic Regression* direkomendasikan sebagai fallback.')
        
        ct = obj.get('crosstab')
        if isinstance(ct, pd.DataFrame): st.markdown('**Tabel silang**'); st.dataframe(ct, use_container_width=True)
        ors = obj.get('or_by_group')
        if isinstance(ors, pd.DataFrame) and not ors.empty: st.markdown('**Odds Ratio menurut kelompok**'); st.dataframe(ors, use_container_width=True, hide_index=True)

def curve_narrative(curve, disease):
    if not isinstance(curve, (tuple, list)) or len(curve) < 4:
        return "Morfologi kurva epidemik belum dapat diklasifikasikan karena data temporal tidak memadai."
    label, short, meaning, implication = curve[:4]
    text = f"Morfologi kurva epidemik diklasifikasikan sebagai **{label}**. {short} "
    text += f"**Implikasi:** {implication} "
    text += f"Pada kasus **{disease}**, bentuk kurva tidak boleh digunakan secara isolasi untuk menyimpulkan mekanisme transmisi. Interpretasi harus selalu mempertimbangkan masa inkubasi penyakit, periode serial, dan potensi *underreporting* atau *backlog* pelaporan pada fase awal wabah."
    return text

def forecast_render(fc):
    if not isinstance(fc, dict): render_value(fc); return
    table = pd.DataFrame({
        'Tanggal': pd.to_datetime(fc.get('dates', []), errors='coerce'),
        'Forecast': fc.get('forecast', []),
        'Lower 95%': fc.get('lower', []),
        'Upper 95%': fc.get('upper', [])
    })
    model = fc.get('model', 'Holt-Winters')
    trend = fc.get('trend', '-')
    show_resume(f"Model peramalan **{model}** memproyeksikan tren **{trend}**. Interval kepercayaan 95% (Lower/Upper) merepresentasikan ketidakpastian inheren dalam proyeksi temporal. Forecast adalah estimasi probabilistik, bukan jumlah kasus yang pasti terjadi.")
    if not table.empty:
        st.line_chart(table.set_index('Tanggal')[['Forecast', 'Lower 95%', 'Upper 95%']])
        st.dataframe(table, use_container_width=True, hide_index=True)

def vulnerable_narrative(v):
    if not isinstance(v, list) or not v:
        return "Belum ditemukan profil populasi rentan yang memenuhi batas minimal analisis statistik."
    d = pd.DataFrame(v)
    top = d.iloc[0]
    age = top.get('Age_Group', '-')
    job = top.get('Pekerjaan', '-')
    comorb = top.get('Status Komorbid', '-')
    total = int(top.get('Total', 0))
    cfr = float(top.get('CFR (%)', 0))
    
    text = f"Stratifikasi risiko multidimensi (Umur × Pekerjaan × Komorbid) mengisolasi profil populasi dengan beban tertinggi. "
    text += f"Profil teratas (**{age} × {job} × {comorb}**) menunjukkan CFR **{cfr:.2f}%** dengan n={total}. "
    text += "Angka ini menyoroti ketimpangan kerentanan. Namun, interpretasi pada strata dengan populasi kecil (n kecil) harus dilakukan dengan hati-hati karena rentan terhadap varians statistik yang tinggi dan estimasi CFR yang tidak stabil."
    return text

def ml_narrative(ml):
    if not isinstance(ml, dict) or not ml:
        return "Layer Machine Learning belum dijalankan atau data tidak memadai untuk inferensi."
    text = "Output Machine Learning berfungsi sebagai *Decision Support System* (DSS), bukan alat diagnostik atau prediktif yang definitif. "
    text += "Validitas model ini bergantung pada asumsi stasioneritas data. Sebelum implementasi operasional, model wajib melalui evaluasi ketat: validasi temporal/eksternal, analisis kalibrasi, deteksi *data drift*, audit bias algoritmik, dan harus selalu disertai *human-in-the-loop oversight* oleh tenaga kesehatan berwenang."
    return text

# ==============================================================================
# UI & CONTROL FLOW
# ==============================================================================

st.sidebar.header('⚙️ Panel Kontrol & Filter')
source = st.sidebar.radio('Sumber Data', ['Gunakan Data Simulasi (AI-Ready)', 'Upload File Excel/CSV Custom'])

if source.startswith('Gunakan'):
    df_raw = generate_data_simulasi().copy()
    st.sidebar.success(f'✅ {len(df_raw):,} data dimuat.')
else:
    uploaded = st.sidebar.file_uploader('Upload File Kasus', type=['xlsx', 'csv'])
    if uploaded is None:
        st.info('Upload file kasus untuk memulai.')
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f'Error membaca file: {exc}')
        st.stop()
    
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.error(f'Kolom wajib kurang: {missing}')
        st.stop()

try:
    st.sidebar.download_button('📥 Download Template Excel Standard', generate_excel_template(), 'Template_Data_Surveilans.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
except Exception:
    pass

st.sidebar.markdown('---')
st.sidebar.subheader('📍 Filter Analisis Epidemiologi')

provinces = sorted(df_raw['Provinsi'].dropna().astype(str).unique()) if 'Provinsi' in df_raw else []
sel_prov = st.sidebar.selectbox('1. Provinsi', ['Semua Provinsi'] + provinces)
dfp = df_raw if sel_prov == 'Semua Provinsi' else df_raw[df_raw['Provinsi'].astype(str).eq(sel_prov)]

districts = sorted(dfp['Kabupaten'].dropna().astype(str).unique()) if 'Kabupaten' in dfp else []
sel_kab = st.sidebar.selectbox('2. Kabupaten/Kota', ['Semua Kabupaten/Kota'] + districts)
dfk = dfp if sel_kab == 'Semua Kabupaten/Kota' else dfp[dfp['Kabupaten'].astype(str).eq(sel_kab)]

disease_values = set()
for col in ['Diagnosis Konfirm', 'Diagnosis Probabel', 'Diagnosis Suspek']:
    if col in df_raw:
        disease_values.update(str(x).strip() for x in df_raw[col].dropna().unique() if str(x).strip().lower() not in {'', 'nan', 'bukan', 'none', 'tidak ada', '-'})

sel_disease = st.sidebar.selectbox('3. Diagnosis Penyakit', ['Semua Penyakit'] + sorted(disease_values))

with st.sidebar.expander('Drill-down wilayah (opsional)'):
    kecs = sorted(dfk['Kecamatan'].dropna().astype(str).unique()) if 'Kecamatan' in dfk else []
    sel_kec = st.selectbox('Kecamatan', ['Semua Kecamatan'] + kecs)
    dbase = dfk if sel_kec == 'Semua Kecamatan' else dfk[dfk['Kecamatan'].astype(str).eq(sel_kec)]
    
    villages = sorted(dbase['Desa/Kelurahan'].dropna().astype(str).unique()) if 'Desa/Kelurahan' in dbase else []
    sel_desa = st.selectbox('Desa/Kelurahan', ['Semua Desa/Kelurahan'] + villages)
    vbase = dbase if sel_desa == 'Semua Desa/Kelurahan' else dbase[dbase['Desa/Kelurahan'].astype(str).eq(sel_desa)]
    
    pusk = sorted(vbase['Puskesmas'].dropna().astype(str).unique()) if 'Puskesmas' in vbase else []
    sel_pusk = st.selectbox('Puskesmas', ['Semua Puskesmas'] + pusk)

include_ml = st.sidebar.checkbox('Aktifkan ML layer', False)

scope = QueryScope(
    province=None if sel_prov == 'Semua Provinsi' else sel_prov,
    district=None if sel_kab == 'Semua Kabupaten/Kota' else sel_kab,
    kecamatan=None if sel_kec == 'Semua Kecamatan' else sel_kec,
    village=None if sel_desa == 'Semua Desa/Kelurahan' else sel_desa,
    puskesmas=None if sel_pusk == 'Semua Puskesmas' else sel_pusk,
    disease=None if sel_disease == 'Semua Penyakit' else sel_disease,
    period_days=3650
)

if sel_disease == 'Semua Penyakit':
    result = engine.descriptive(dfk)
    label = sel_kab if sel_kab != 'Semua Kabupaten/Kota' else (sel_prov if sel_prov != 'Semua Provinsi' else 'Indonesia')
    st.markdown(f'## 📊 Analisis Deskriptif — {label}')
    
    ov = result['overview']
    a, b = st.columns(2)
    a.metric('Total Kunjungan Pasien', f"{ov['total_cases']:,}")
    b.metric('Kasus Meninggal', f"{ov['deaths']:,}")
    
    st.markdown('### Resume Epidemiologi')
    show_resume(descriptive_narrative(result, label))
    
    st.markdown('### 🏆 10 Besar Penyakit')
    show_resume('Tabel ini menunjukkan penyakit dengan beban kasus terbesar dalam scope yang dipilih. Jumlah kasus menggambarkan beban absolut; CFR menggambarkan proporsi kematian di antara kasus dan tidak boleh ditafsirkan sebagai risiko populasi tanpa denominator yang sesuai.')
    render_value(result['top10_diseases'])
    
    st.markdown('### Distribusi')
    show_resume('Distribusi berikut memperlihatkan komposisi kasus menurut penyakit, jenis kelamin, kelompok umur, provinsi, dan kabupaten/kota. Perbedaan jumlah kasus adalah temuan deskriptif dan tidak otomatis menunjukkan perbedaan risiko.')
    render_value(result['disease_distribution'])
    render_value(result['sex_distribution'], 'Jenis Kelamin')
    render_value(result['age_distribution'], 'Kelompok Umur')
    render_value(result['province_distribution'], 'Provinsi')
    render_value(result['district_distribution'].head(50), 'Kabupaten/Kota')

else:
    result = engine.analyze(df_raw, scope=scope, include_ml=include_ml, mode='epidemiology')
    label = sel_kab if sel_kab != 'Semua Kabupaten/Kota' else (sel_prov if sel_prov != 'Semua Provinsi' else 'Indonesia')
    st.markdown(f'## 🧬 Analisis Epidemiologi — {sel_disease}')
    st.caption(f'Scope: **{sel_disease} — {label}** | TIME + PERSON + PLACE')
    
    if not result.get('eligible', False):
        st.warning('Analisis epidemiologi belum dapat dijalankan.')
        render_value(result.get('eligibility'))
        st.stop()
        
    total = int(result['overview']['total_cases'])
    mortality = result.get('mortality')
    deaths = int(mortality.get('deaths', mortality.get('meninggal', 0)) or 0) if isinstance(mortality, dict) else 0
    cfr = deaths / total * 100 if total else 0
    
    a, b, c = st.columns(3)
    a.metric(f'Total {sel_disease}', f'{total:,}')
    b.metric('Meninggal', f'{deaths:,}')
    c.metric('CFR', f'{cfr:.2f}%')
    
    tabs = st.tabs(['📊 TIME + PERSON + PLACE', '🧪 Faktor Risiko', '🚨 Early Warning / KLB', '🗺️ Spatial / DBSCAN', '📈 Kurva Epidemik', '👥 Vulnerable Population', '🧠 ML'])
    
    with tabs[0]:
        tri = result.get('trias_summary', {})
        show_resume(tri.get('narrative', 'TIME + PERSON + PLACE menjelaskan kapan, siapa dan di mana kasus terjadi.'))
        top10 = tri.get('top10_province') if isinstance(tri, dict) else None
        if isinstance(top10, pd.DataFrame) and not top10.empty: render_value(top10, '10 Besar Provinsi')
        
        render_value(result.get('place'), 'PLACE')
        render_value(result.get('person'), 'PERSON')
        
        t = result.get('time')
        if isinstance(t, pd.DataFrame) and not t.empty:
            chart = t.copy()
            chart['Tanggal Sakit'] = pd.to_datetime(chart['Tanggal Sakit'], errors='coerce')
            st.line_chart(chart.dropna(subset=['Tanggal Sakit']).set_index('Tanggal Sakit')['Jumlah Kasus'])
            show_resume('Distribusi waktu memperlihatkan kapan kasus terjadi dan membantu mengenali perubahan tren, puncak, atau pola gelombang.')
            render_value(t, 'TIME')
            
        show_resume('Distribusi mortalitas dan CFR menjelaskan beban kematian relatif terhadap jumlah kasus yang dianalisis. CFR perlu dibaca bersama ukuran sampel, kelengkapan outcome, dan karakteristik kasus.')
        render_value(mortality, 'MORTALITY / CFR')
        
    with tabs[1]:
        st.markdown('### Resume Faktor Risiko')
        show_resume(risk_narrative(result.get('risk_factors')))
        s = risk_summary(result.get('risk_factors'))
        if not s.empty: st.dataframe(s, use_container_width=True, hide_index=True)
        
        st.markdown('### Hasil Analisis Bivariat & Multivariat Selengkapnya')
        render_risk_factors(result.get('risk_factors'))
        st.caption('OR, CI 95%, p-value dan adjusted OR harus dibaca bersama desain studi, confounding, bias dan ukuran sampel.')
        
    with tabs[2]:
        st.markdown('### Interpretasi Epidemiologi')
        show_resume(ews_narrative(result.get('ews'), result.get('rt')))
        render_value(result.get('ews'), 'Indikator EWS')
        render_value(result.get('rt'), 'Rₜ')
        
    with tabs[3]:
        st.markdown('### Interpretasi Spatial')
        show_resume(spatial_narrative(result.get('spatial')))
        spatial = result.get('spatial')
        render_value(spatial, 'Hasil DBSCAN')
        
        if isinstance(spatial, pd.DataFrame) and not spatial.empty and {'Latitude', 'Longitude'}.issubset(spatial.columns):
            geo = spatial.dropna(subset=['Latitude', 'Longitude'])
            if not geo.empty:
                m = folium.Map(location=[float(geo.Latitude.mean()), float(geo.Longitude.mean())], zoom_start=9)
                for _, row in geo.head(500).iterrows():
                    folium.CircleMarker(
                        [float(row.Latitude), float(row.Longitude)],
                        radius=4,
                        popup=f"{row.get('Desa/Kelurahan', '')} | Cluster {row.get('Cluster', '')}"
                    ).add_to(m)
                st_folium(m, width=None, height=500)
                
        show_resume('Episentrum/titik prioritas harus dibaca sebagai lokasi konsentrasi spasial dalam dataset, bukan otomatis sebagai sumber penularan.')
        render_value(result.get('epicenters'), 'Episentrum / Titik Prioritas')
        
    with tabs[4]:
        curve = result.get('epidemic_curve_classification')
        st.markdown('### Interpretasi Kurva Epidemik')
        show_resume(curve_narrative(curve, sel_disease))
        st.markdown('**Jenis umum:** Point Source, Common Source Continuous, Intermittent Source, Propagated/Multi-Wave, dan Mixed/Unclassified. Interpretasi harus mempertimbangkan masa inkubasi dan mekanisme penyakit.')
        
        t = result.get('time')
        if isinstance(t, pd.DataFrame) and not t.empty:
            chart = t.copy()
            chart['Tanggal Sakit'] = pd.to_datetime(chart['Tanggal Sakit'], errors='coerce')
            st.line_chart(chart.dropna(subset=['Tanggal Sakit']).set_index('Tanggal Sakit')['Jumlah Kasus'])
            
        st.markdown('### Forecast 14 Hari')
        forecast_render(result.get('forecast'))
        
    with tabs[5]:
        v = result.get('vulnerable')
        st.markdown('### Interpretasi Vulnerable Population')
        show_resume(vulnerable_narrative(v))
        render_value(v, 'Profil Rentan')
        
    with tabs[6]:
        if include_ml:
            st.markdown('### Interpretasi ML')
            show_resume(ml_narrative(result.get('ml')))
            render_value(result.get('ml'))
        else:
            show_resume('ML layer tidak diaktifkan. Analisis epidemiologi non-ML tetap dapat digunakan sesuai kecukupan data.')

st.caption('SI-HIS Intelligence — epidemiological decision-support with TIME + PERSON + PLACE.')
