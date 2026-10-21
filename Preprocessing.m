%% 1.1 LOAD DATA RAW
data = readtable('dataset_exoplanet_raw.xlsx');

[n, p] = size(data);
disp('=== DATA AWAL ===')
disp(['Jumlah baris : ' num2str(n)])
disp(['Jumlah kolom : ' num2str(p)])

%% 1.2 HAPUS KOLOM TIDAK RELEVAN
kolom_hapus = { ...
    'rowid', 'kepid', 'kepler_name', ...
    'koi_pdisposition', 'koi_tce_delivname', ...
    'koi_period_err1',  'koi_period_err2', ...
    'koi_time0bk',      'koi_time0bk_err1', 'koi_time0bk_err2', ...
    'koi_impact_err1',  'koi_impact_err2', ...
    'koi_duration_err1','koi_duration_err2', ...
    'koi_depth_err1',   'koi_depth_err2', ...
    'koi_prad_err1',    'koi_prad_err2', ...
    'koi_teq_err1',     'koi_teq_err2', ...
    'koi_insol_err1',   'koi_insol_err2', ...
    'koi_steff_err1',   'koi_steff_err2', ...
    'koi_slogg_err1',   'koi_slogg_err2', ...
    'koi_srad_err1',    'koi_srad_err2', ...
    'koi_fpflag_nt',    'koi_fpflag_ss', ...
    'koi_fpflag_co',    'koi_fpflag_ec', ...
    'ra', 'dec', 'koi_kepmag', 'koi_tce_plnt_num'};

data = removevars(data, kolom_hapus);

[n, p] = size(data);
disp('=== SETELAH HAPUS KOLOM ===')
disp(['Jumlah baris : ' num2str(n)])
disp(['Jumlah kolom : ' num2str(p)])

%% 1.3 SELEKSI FITUR PENTING
fitur_svm = {'koi_period', 'koi_impact', 'koi_duration', ...
             'koi_depth',  'koi_model_snr', 'koi_score', ...
             'koi_steff',  'koi_slogg', 'koi_srad', ...
             'koi_prad',   'koi_teq',   'koi_insol'};

kolom_penting = [{'kepoi_name', 'koi_disposition'}, fitur_svm];
data = data(:, kolom_penting);

[n, p] = size(data);
disp('=== SETELAH SELEKSI FITUR ===')
disp(['Jumlah baris : ' num2str(n)])
disp(['Jumlah kolom : ' num2str(p)])

%% 1.4 CEK MISSING VALUES
kolom_numerik = fitur_svm;

disp('=== MISSING VALUES PER KOLOM ===')
total_mv = 0;
for i = 1:length(kolom_numerik)
    col = data.(kolom_numerik{i});
    mv  = sum(isnan(col));
    if mv > 0
        disp([kolom_numerik{i} ' : ' num2str(mv) ' missing'])
        total_mv = total_mv + mv;
    end
end
disp(['Total missing values : ' num2str(total_mv)])

%% 1.5 HAPUS BARIS YANG MENGANDUNG NULL
mask_bersih = true(height(data), 1);
for i = 1:length(kolom_numerik)
    col = data.(kolom_numerik{i});
    mask_bersih = mask_bersih & ~isnan(col);
end

data = data(mask_bersih, :);

disp(['Baris tersisa : ' num2str(height(data))])

%% 1.6 KONVERSI NILAI KE SKALA YANG BENAR
data.koi_period    = data.koi_period    / 1e9;
data.koi_impact    = data.koi_impact    / 10000;
data.koi_duration  = data.koi_duration  / 100000;
data.koi_depth     = data.koi_depth     / 10000;
data.koi_model_snr = data.koi_model_snr / 100;
data.koi_score     = data.koi_score     / 10000;
data.koi_steff     = data.koi_steff     / 100;
data.koi_slogg     = data.koi_slogg     / 1000;
data.koi_srad      = data.koi_srad      / 10000;
data.koi_prad      = data.koi_prad      / 100;
data.koi_teq       = data.koi_teq       / 10;
data.koi_insol     = data.koi_insol     / 10000;

%% 1.7 VERIFIKASI NILAI
disp('=== VERIFIKASI NILAI SAMPLE ===')
disp(['koi_period (baris 1) : ' num2str(data.koi_period(1))])
disp(['koi_prad   (baris 1) : ' num2str(data.koi_prad(1))])
disp(['koi_teq    (baris 1) : ' num2str(data.koi_teq(1))])
disp(['koi_score  (baris 1) : ' num2str(data.koi_score(1))])
disp(['koi_steff  (baris 1) : ' num2str(data.koi_steff(1))])

%% 1.8 CEK DISTRIBUSI LABEL
confirmed = sum(strcmp(data.koi_disposition, 'CONFIRMED'));
candidate = sum(strcmp(data.koi_disposition, 'CANDIDATE'));
false_pos = sum(strcmp(data.koi_disposition, 'FALSE POSITIVE'));

disp('=== DISTRIBUSI LABEL ===')
disp(['Jumlah baris   : ' num2str(height(data))])
disp(['CONFIRMED      : ' num2str(confirmed)])
disp(['CANDIDATE      : ' num2str(candidate)])
disp(['FALSE POSITIVE : ' num2str(false_pos)])

%% 1.9 VISUALISASI DISTRIBUSI LABEL
figure;
label_counts = [confirmed, candidate, false_pos];
pie(label_counts, {'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'});
title('Distribusi Label koi\_disposition');
colormap([0.18 0.55 0.34; 0.20 0.40 0.72; 0.85 0.20 0.20]);

disp('=== PREPROCESSING SELESAI ===')
disp(['Data awal   : ' num2str(n) ' baris'])
disp(['Data bersih : ' num2str(height(data)) ' baris'])
disp(['Fitur       : ' num2str(length(kolom_numerik)) ' kolom'])

%% 1.10 SIMPAN DATASET BERSIH
writetable(data, 'dataset_exoplanet_clean.xlsx');