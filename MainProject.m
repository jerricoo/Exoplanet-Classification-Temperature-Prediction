clear; clc;

%% 2.1 LOAD DATA BERSIH
data = readtable('dataset_exoplanet_clean.xlsx');

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

disp('=== DATA LOADED ===')
disp(['Jumlah baris : ' num2str(height(data))])

%% 2.2 ENCODING LABEL
% CONFIRMED = 1, CANDIDATE = 2, FALSE POSITIVE = 0
n     = height(data);
label = zeros(n, 1);
for i = 1:n
    if strcmp(data.koi_disposition{i}, 'CONFIRMED')
        label(i) = 1;
    elseif strcmp(data.koi_disposition{i}, 'CANDIDATE')
        label(i) = 2;
    else
        label(i) = 0;
    end
end

disp('=== ENCODING LABEL ===')
disp(['CONFIRMED      (1) : ' num2str(sum(label == 1))])
disp(['CANDIDATE      (2) : ' num2str(sum(label == 2))])
disp(['FALSE POSITIVE (0) : ' num2str(sum(label == 0))])

%% 2.3 PERSIAPAN FITUR
fitur_svm = {'koi_period', 'koi_impact', 'koi_duration', ...
             'koi_depth',  'koi_model_snr', 'koi_score', ...
             'koi_steff',  'koi_slogg', 'koi_srad', ...
             'koi_prad',   'koi_teq',   'koi_insol'};

X = table2array(data(:, fitur_svm));
y = label;

%% 2.4 NORMALISASI Z-SCORE
X_mean = mean(X);
X_std  = std(X);
X_std(X_std == 0) = 1;
X_norm = (X - X_mean) ./ X_std;

disp('=== NORMALISASI Z-SCORE ===')
disp(['Mean rata-rata : ' num2str(round(mean(mean(X_norm)), 4))])
disp(['Std rata-rata  : ' num2str(round(mean(std(X_norm)), 4))])

%% 2.5 STRATIFIED SPLIT 80:20
rng(42);
idx_0 = find(y == 0);
idx_1 = find(y == 1);
idx_2 = find(y == 2);

idx_0 = idx_0(randperm(length(idx_0)));
idx_1 = idx_1(randperm(length(idx_1)));
idx_2 = idx_2(randperm(length(idx_2)));

n0_train = round(0.8 * length(idx_0));
n1_train = round(0.8 * length(idx_1));
n2_train = round(0.8 * length(idx_2));

idx_train = [idx_0(1:n0_train); idx_1(1:n1_train); idx_2(1:n2_train)];
idx_test  = [idx_0(n0_train+1:end); idx_1(n1_train+1:end); idx_2(n2_train+1:end)];

X_train = X_norm(idx_train, :);
y_train = y(idx_train);
X_test  = X_norm(idx_test, :);
y_test  = y(idx_test);

disp('=== STRATIFIED SPLIT 80:20 ===')
disp(['Training : ' num2str(length(idx_train)) ' data'])
disp(['  - FALSE POSITIVE : ' num2str(sum(y_train==0))])
disp(['  - CONFIRMED      : ' num2str(sum(y_train==1))])
disp(['  - CANDIDATE      : ' num2str(sum(y_train==2))])
disp(['Testing  : ' num2str(length(idx_test)) ' data'])
disp(['  - FALSE POSITIVE : ' num2str(sum(y_test==0))])
disp(['  - CONFIRMED      : ' num2str(sum(y_test==1))])
disp(['  - CANDIDATE      : ' num2str(sum(y_test==2))])

%% 2.6 CLASS WEIGHTS
n_total = length(y_train);
w0 = n_total / (3 * sum(y_train == 0));
w1 = n_total / (3 * sum(y_train == 1));
w2 = n_total / (3 * sum(y_train == 2));

bobot = zeros(length(y_train), 1);
bobot(y_train == 0) = w0;
bobot(y_train == 1) = w1;
bobot(y_train == 2) = w2;

disp('=== CLASS WEIGHTS ===')
disp(['Bobot FALSE POSITIVE : ' num2str(round(w0, 4))])
disp(['Bobot CONFIRMED      : ' num2str(round(w1, 4))])
disp(['Bobot CANDIDATE      : ' num2str(round(w2, 4))])

%% 2.7 TRAINING SVM LINEAR
disp('=== TRAINING SVM LINEAR ===')

t = templateSVM('KernelFunction', 'linear', ...
                'Standardize', false, ...
                'BoxConstraint', 10);

mdl_svm = fitcecoc(X_train, y_train, ...
                   'Learners', t, ...
                   'Weights', bobot);
%% 2.8 PREDIKSI & EVALUASI
y_pred   = predict(mdl_svm, X_test);
accuracy = sum(y_pred == y_test) / length(y_test) * 100;

disp('=== EVALUASI SVM LINEAR ===')
disp(['Accuracy : ' num2str(round(accuracy, 2)) ' %'])

figure;
confusionchart(y_test, y_pred, ...
    'Title', 'Confusion Matrix — SVM Linear', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

kelas      = [0, 1, 2];
nama_kelas = {'FALSE POSITIVE', 'CONFIRMED', 'CANDIDATE'};

disp('=== PRECISION, RECALL, F1 PER KELAS ===')
for k = 1:length(kelas)
    TP = sum(y_pred == kelas(k) & y_test == kelas(k));
    FP = sum(y_pred == kelas(k) & y_test ~= kelas(k));
    FN = sum(y_pred ~= kelas(k) & y_test == kelas(k));

    if (TP+FP) == 0; precision = 0; else; precision = TP/(TP+FP); end
    if (TP+FN) == 0; recall    = 0; else; recall    = TP/(TP+FN); end
    if (precision+recall) == 0; f1 = 0;
    else; f1 = 2*(precision*recall)/(precision+recall); end

    disp([nama_kelas{k} ':'])
    disp(['  Precision : ' num2str(round(precision*100, 2)) ' %'])
    disp(['  Recall    : ' num2str(round(recall*100, 2))    ' %'])
    disp(['  F1-Score  : ' num2str(round(f1*100, 2))        ' %'])
end

%% Analisis prediksi per kelas
disp('=== ANALISIS PREDIKSI PER KELAS ===')

% CONFIRMED
total_conf      = sum(y_test == 1);
conf_benar      = sum(y_pred == 1 & y_test == 1);
conf_jadi_cand  = sum(y_pred == 2 & y_test == 1);
conf_jadi_fp    = sum(y_pred == 0 & y_test == 1);

disp('CONFIRMED:')
disp(['  Total di testing                  : ' num2str(total_conf) ' data'])
disp(['  Diprediksi benar sebagai CONFIRMED: ' num2str(conf_benar) ...
      ' data (' num2str(round(conf_benar/total_conf*100,2)) '%)'])
disp(['  Diprediksi sebagai CANDIDATE      : ' num2str(conf_jadi_cand) ...
      ' data (' num2str(round(conf_jadi_cand/total_conf*100,2)) '%)'])
disp(['  Diprediksi sebagai FALSE POSITIVE : ' num2str(conf_jadi_fp) ...
      ' data (' num2str(round(conf_jadi_fp/total_conf*100,2)) '%)'])

% CANDIDATE
total_cand     = sum(y_test == 2);
cand_benar     = sum(y_pred == 2 & y_test == 2);
cand_jadi_conf = sum(y_pred == 1 & y_test == 2);
cand_jadi_fp   = sum(y_pred == 0 & y_test == 2);

disp('CANDIDATE:')
disp(['  Total di testing                  : ' num2str(total_cand) ' data'])
disp(['  Diprediksi benar sebagai CANDIDATE: ' num2str(cand_benar) ...
      ' data (' num2str(round(cand_benar/total_cand*100,2)) '%)'])
disp(['  Diprediksi sebagai CONFIRMED      : ' num2str(cand_jadi_conf) ...
      ' data (' num2str(round(cand_jadi_conf/total_cand*100,2)) '%)'])
disp(['  Diprediksi sebagai FALSE POSITIVE : ' num2str(cand_jadi_fp) ...
      ' data (' num2str(round(cand_jadi_fp/total_cand*100,2)) '%)'])

% FALSE POSITIVE
total_fp      = sum(y_test == 0);
fp_benar      = sum(y_pred == 0 & y_test == 0);
fp_jadi_conf  = sum(y_pred == 1 & y_test == 0);
fp_jadi_cand  = sum(y_pred == 2 & y_test == 0);

disp('FALSE POSITIVE:')
disp(['  Total di testing                      : ' num2str(total_fp) ' data'])
disp(['  Diprediksi benar sebagai FALSE POSITIVE: ' num2str(fp_benar) ...
      ' data (' num2str(round(fp_benar/total_fp*100,2)) '%)'])
disp(['  Diprediksi sebagai CONFIRMED           : ' num2str(fp_jadi_conf) ...
      ' data (' num2str(round(fp_jadi_conf/total_fp*100,2)) '%)'])
disp(['  Diprediksi sebagai CANDIDATE           : ' num2str(fp_jadi_cand) ...
      ' data (' num2str(round(fp_jadi_cand/total_fp*100,2)) '%)'])

figure;
akurasi_kelas = zeros(1, length(kelas));
for k = 1:length(kelas)
    benar = sum(y_pred == kelas(k) & y_test == kelas(k));
    total = sum(y_test == kelas(k));
    akurasi_kelas(k) = benar / total * 100;
end
bar(akurasi_kelas);
set(gca, 'XTickLabel', nama_kelas);
ylabel('Accuracy (%)');
title('Accuracy per Kelas — SVM Linear');
ylim([0 100]);
grid on;

%% 2.9 STRATIFIED K-FOLD CROSS VALIDATION (K=5)
disp('=== STRATIFIED K-FOLD CV (K=5) ===')

K          = 5;
kelas_cv   = [0, 1, 2];
acc_cv     = zeros(K, 1);
f1_cv      = zeros(K, 3);

rng(42);
idx_kelas = cell(3, 1);
idx_kelas{1} = find(y == 0);
idx_kelas{2} = find(y == 1);
idx_kelas{3} = find(y == 2);

for c = 1:3
    idx_kelas{c} = idx_kelas{c}(randperm(length(idx_kelas{c})));
end

for k = 1:K
    test_idx  = [];
    train_idx = [];

    for c = 1:3
        n_c         = length(idx_kelas{c});
        fold_size_c = floor(n_c / K);
        idx_start   = (k-1) * fold_size_c + 1;
        idx_end     = k * fold_size_c;
        if k == K; idx_end = n_c; end

        test_idx  = [test_idx;  idx_kelas{c}(idx_start:idx_end)];
        train_idx = [train_idx; idx_kelas{c}([1:idx_start-1, idx_end+1:end])];
    end

    X_cv_train = X_norm(train_idx, :);
    y_cv_train = y(train_idx);
    X_cv_test  = X_norm(test_idx, :);
    y_cv_test  = y(test_idx);

    n_fold = length(y_cv_train);
    w0_cv  = n_fold / (3 * sum(y_cv_train == 0));
    w1_cv  = n_fold / (3 * sum(y_cv_train == 1));
    w2_cv  = n_fold / (3 * sum(y_cv_train == 2));

    bobot_cv = zeros(n_fold, 1);
    bobot_cv(y_cv_train == 0) = w0_cv;
    bobot_cv(y_cv_train == 1) = w1_cv;
    bobot_cv(y_cv_train == 2) = w2_cv;

    t_cv = templateSVM('KernelFunction', 'linear', ...
                       'Standardize', false, ...
                       'BoxConstraint', 10);

    mdl_cv    = fitcecoc(X_cv_train, y_cv_train, ...
                         'Learners', t_cv, ...
                         'Weights', bobot_cv);

    y_cv_pred = predict(mdl_cv, X_cv_test);
    acc_cv(k) = sum(y_cv_pred == y_cv_test) / length(y_cv_test) * 100;

    for c = 1:3
        TP = sum(y_cv_pred == kelas_cv(c) & y_cv_test == kelas_cv(c));
        FP = sum(y_cv_pred == kelas_cv(c) & y_cv_test ~= kelas_cv(c));
        FN = sum(y_cv_pred ~= kelas_cv(c) & y_cv_test == kelas_cv(c));

        if (TP+FP) == 0; prec = 0; else; prec = TP/(TP+FP); end
        if (TP+FN) == 0; rec  = 0; else; rec  = TP/(TP+FN); end
        if (prec+rec) == 0; f1_cv(k,c) = 0;
        else; f1_cv(k,c) = 2*(prec*rec)/(prec+rec)*100; end
    end

    disp(['Fold ' num2str(k) ' | Acc: ' num2str(round(acc_cv(k),2)) '%' ...
          ' | F1-FP: '   num2str(round(f1_cv(k,1),2)) '%' ...
          ' | F1-CONF: ' num2str(round(f1_cv(k,2),2)) '%' ...
          ' | F1-CAND: ' num2str(round(f1_cv(k,3),2)) '%'])
end

disp('=== HASIL RATA-RATA STRATIFIED K-FOLD CV ===')
disp(['Accuracy rata-rata         : ' num2str(round(mean(acc_cv),2)) ' %'])
disp(['Std Accuracy               : ' num2str(round(std(acc_cv),2))  ' %'])
disp(['F1 FALSE POSITIVE rata-rata: ' num2str(round(mean(f1_cv(:,1)),2)) ' %'])
disp(['F1 CONFIRMED rata-rata     : ' num2str(round(mean(f1_cv(:,2)),2)) ' %'])
disp(['F1 CANDIDATE rata-rata     : ' num2str(round(mean(f1_cv(:,3)),2)) ' %'])

figure;
boxplot(f1_cv, 'Labels', {'FALSE POSITIVE','CONFIRMED','CANDIDATE'});
ylabel('F1-Score (%)');
title('Distribusi F1-Score — Stratified K-Fold CV (K=5)');
grid on;

figure;
plot(1:K, acc_cv, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(mean(acc_cv), 'r--', 'LineWidth', 1.5);
xlabel('Fold ke-');
ylabel('Accuracy (%)');
title('Accuracy per Fold — Stratified K-Fold CV');
legend('Accuracy per Fold', 'Rata-rata', 'Location', 'southeast');
ylim([70 100]);
grid on;

disp('=== SVM LINEAR SELESAI ===')
disp(['Accuracy keseluruhan : ' num2str(round(accuracy, 2)) ' %'])

%% 2.10 FILTER PLANET CONFIRMED
data_confirmed = data(label == 1, :);

disp('=== FILTER PLANET CONFIRMED ===')
disp(['Total planet CONFIRMED : ' num2str(height(data_confirmed)) ' data'])

%% STEP 3 — PCR (Principal Component Regression)
%% 3.1 PERSIAPAN DATA CONFIRMED
fitur_pcr  = {'koi_steff', 'koi_srad', 'koi_period', ...
              'koi_insol', 'koi_slogg', 'koi_duration', ...
              'koi_model_snr'};
target_pcr = 'koi_teq';

X_pcr = table2array(data_confirmed(:, fitur_pcr));
Y_pcr = data_confirmed.(target_pcr);
n_pcr = size(X_pcr, 1);

disp('=== PERSIAPAN DATA PCR ===')
disp(['Jumlah planet CONFIRMED : ' num2str(n_pcr)])
disp(['Fitur input             : ' num2str(length(fitur_pcr))])
disp(['Target output           : koi_teq (suhu planet)'])

%% 3.2 STANDARISASI Z-SCORE
Xmean = mean(X_pcr);
Xstd  = std(X_pcr);
Xstd(Xstd == 0) = 1;
Zr = (X_pcr - Xmean) ./ Xstd;

disp('=== STANDARISASI Z-SCORE ===')
disp(['Mean rata-rata : ' num2str(round(mean(mean(Zr)), 4))])
disp(['Std rata-rata  : ' num2str(round(mean(std(Zr)), 4))])

%% 3.3 MATRIKS KOVARIANS
Zcov = (Zr' * Zr) / (n_pcr - 1);

disp('=== MATRIKS KOVARIANS ===')
disp(['Ukuran : ' num2str(size(Zcov,1)) ' x ' num2str(size(Zcov,2))])

%% 3.4 EIGENVALUE & EIGENVECTOR
[V, D]      = eig(Zcov);
eigenvalues = diag(D);
[eigenvalues, idx_sort] = sort(eigenvalues, 'descend');
V         = V(:, idx_sort);
total_var = sum(eigenvalues);
proporsi  = eigenvalues / total_var * 100;
kumulatif = cumsum(proporsi);

disp('=== HASIL PCA ===')
for i = 1:length(eigenvalues)
    disp(['PC' num2str(i) ...
          ' | Eigenvalue: ' num2str(round(eigenvalues(i),4)) ...
          ' | Proporsi: '   num2str(round(proporsi(i),2))  '%' ...
          ' | Kumulatif: '  num2str(round(kumulatif(i),2)) '%'])
end

%% 3.5 PILIH KOMPONEN (kumulatif >= 90%)
n_komponen = find(kumulatif >= 90, 1);

disp('=== PEMILIHAN KOMPONEN ===')
disp(['Jumlah komponen : ' num2str(n_komponen)])
disp(['Kumulatif       : ' num2str(round(kumulatif(n_komponen),2)) '%'])

%% 3.6 TRANSFORMASI PCA
VA = V(:, 1:n_komponen);
W  = Zr * VA;

disp('=== TRANSFORMASI PCA ===')
disp(['Dimensi asli   : ' num2str(size(Zr,2)) ' fitur'])
disp(['Dimensi PCA    : ' num2str(size(W,2))  ' komponen'])

%% 3.7 SPLIT 80:20
rng(42);
idx_r       = randperm(n_pcr);
n_train_r   = round(0.8 * n_pcr);
idx_train_r = idx_r(1:n_train_r);
idx_test_r  = idx_r(n_train_r+1:end);

x0          = ones(n_pcr, 1);   
W_intercept = [x0 W];

X_train_r = W_intercept(idx_train_r, :);
Y_train_r = Y_pcr(idx_train_r);
X_test_r  = W_intercept(idx_test_r, :);
Y_test_r  = Y_pcr(idx_test_r);

disp('=== SPLIT DATA PCR 80:20 ===')
disp(['Training : ' num2str(length(idx_train_r)) ' data'])
disp(['Testing  : ' num2str(length(idx_test_r))  ' data'])

%% 3.8 REGRESI LINEAR OLS
b_reg = inv(X_train_r' * X_train_r) * (X_train_r' * Y_train_r);
Yhat  = X_train_r * b_reg;
error = Y_train_r - Yhat;

disp('=== KOEFISIEN REGRESI ===')
disp(['Intercept : ' num2str(round(b_reg(1),4))])
for i = 1:n_komponen
    disp(['PC' num2str(i) '      : ' num2str(round(b_reg(i+1),4))])
end

%% 3.9 EVALUASI ANOVA
n_tr   = length(Y_train_r);
[~,px] = size(X_train_r);

ssTot = Y_train_r'*Y_train_r - n_tr*mean(Y_train_r)^2;
ssReg = b_reg'*(X_train_r'*Y_train_r) - n_tr*mean(Y_train_r)^2;
ssRes = ssTot - ssReg;
R2    = (ssReg/ssTot)*100;
dfReg = px-1; dfRes = n_tr-px;
msReg = ssReg/dfReg; msRes = ssRes/dfRes;
Fhit  = msReg/msRes;

Yhat_test  = X_test_r * b_reg;
error_test = Y_test_r - Yhat_test;
RMSE       = sqrt(mean(error_test.^2));

disp('=== TABEL ANOVA ===')
disp(['SS Regresi  : ' num2str(round(ssReg,2))])
disp(['SS Residual : ' num2str(round(ssRes,2))])
disp(['SS Total    : ' num2str(round(ssTot,2))])
disp(['MS Regresi  : ' num2str(round(msReg,2))])
disp(['MS Residual : ' num2str(round(msRes,2))])
disp(['F hitung    : ' num2str(round(Fhit,4))])
disp('=== EVALUASI PCR ===')
disp(['R-squared (R2) : ' num2str(round(R2,4))   ' %'])
disp(['RMSE           : ' num2str(round(RMSE,4)) ' Kelvin'])

%% 3.10 UJI NORMALITAS RESIDUAL
disp('=== UJI NORMALITAS RESIDUAL ===')
disp(['Mean residual : ' num2str(round(mean(error),4))])
disp(['Std residual  : ' num2str(round(std(error),4))])

if abs(mean(error)) < 1
    disp('Mean residual mendekati 0 — Asumsi normalitas terpenuhi')
else
    disp('Mean residual jauh dari 0 — Asumsi normalitas perlu diperiksa')
end

figure;
histogram(error, 30, 'FaceColor', '#2166AC');
xlabel('Residual (Kelvin)');
ylabel('Frekuensi');
title('Distribusi Residual — PCR');
grid on;

%% 3.11 PLOT RESIDUAL VS FITTED
dalam_2std = sum(abs(error) <= 2*std(error)) / length(error) * 100;

disp('=== UJI HOMOSKEDASTISITAS ===')
disp(['Residual dalam 2 Std : ' num2str(round(dalam_2std,2)) '%'])
if dalam_2std >= 95
    disp('95% residual dalam 2 Std — Homoskedastisitas terpenuhi')
else
    disp('Kurang dari 95% — Ada indikasi heteroskedastisitas')
end

figure;
scatter(Yhat, error, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
yline(0, 'r--', 'LineWidth', 2);
yline(2*std(error),  'g--', 'LineWidth', 1.5);
yline(-2*std(error), 'g--', 'LineWidth', 1.5);
xlabel('Fitted Values (Kelvin)');
ylabel('Residual (Kelvin)');
title('Residual vs Fitted Values — PCR');
legend('Residual','Garis Nol','+2 Std','-2 Std','Location','northeast');
grid on;

%% 3.12 PREDIKSI SUHU SEMUA PLANET CONFIRMED
W_all    = [ones(n_pcr,1) W];
Yhat_all = W_all * b_reg;

disp('=== PREDIKSI SUHU PLANET CONFIRMED ===')
disp(['Range : ' num2str(round(min(Yhat_all),2)) ...
      ' - '     num2str(round(max(Yhat_all),2)) ' K'])

%% 3.13 VISUALISASI SCREE PLOT
figure;
plot(1:length(proporsi), kumulatif, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(90, 'r--', 'LineWidth', 1.5);
xlabel('Jumlah Komponen');
ylabel('Kumulatif Variansi (%)');
title('Scree Plot PCA');
legend('Kumulatif Variansi','Threshold 90%','Location','southeast');
grid on;

%% 3.14 VISUALISASI PREDIKSI VS AKTUAL
figure;
scatter(Y_test_r, Yhat_test, 20, 'b', 'filled');
hold on;
plot([min(Y_test_r) max(Y_test_r)], ...
     [min(Y_test_r) max(Y_test_r)], 'r--', 'LineWidth', 2);
xlabel('Suhu Aktual koi\_teq (K)');
ylabel('Suhu Prediksi (K)');
title('PCR — Prediksi vs Aktual Suhu Planet');
legend('Data','Garis Ideal','Location','northwest');
grid on;

disp('=== PCR SELESAI ===')
disp(['R2   : ' num2str(round(R2,4))   ' %'])
disp(['RMSE : ' num2str(round(RMSE,4)) ' Kelvin'])

%% STEP 4 — HABITABILITY SCORING
%% 4.1 PREDIKSI & FILTER NILAI NEGATIF
Yhat_all(Yhat_all < 0) = 0;

disp('=== PREDIKSI SUHU PLANET CONFIRMED ===')
disp(['Range : ' num2str(round(min(Yhat_all),2)) ...
      ' - '     num2str(round(max(Yhat_all),2)) ' K'])

%% 4.2 AMBIL NILAI ASLI
insol_asli  = data_confirmed.koi_insol;
prad_asli   = data_confirmed.koi_prad;
nama_planet = data_confirmed.kepoi_name;

%% 4.3 HITUNG SKOR PER KRITERIA
teq_ideal   = 300;
insol_ideal = 1.0;
prad_ideal  = 1.0;
teq_max     = 200;
insol_max   = 2.5;
prad_max    = 1.5;

skor_teq   = max(0, 100 - (abs(Yhat_all - teq_ideal)    / teq_max)   * 100);
skor_insol = max(0, 100 - (abs(insol_asli - insol_ideal) / insol_max) * 100);
skor_prad  = max(0, 100 - (abs(prad_asli - prad_ideal)   / prad_max)  * 100);

%% 4.4 SKOR FINAL
% Bobot: suhu 50%, radiasi 30%, ukuran 20%
skor_final = (skor_teq * 0.5) + (skor_insol * 0.3) + (skor_prad * 0.2);

disp('=== SKOR LAYAK HUNI ===')
disp(['Range skor : ' num2str(round(min(skor_final),2)) ...
      ' - '          num2str(round(max(skor_final),2))])
disp(['Rata-rata  : ' num2str(round(mean(skor_final),2))])

%% 4.5 KATEGORISASI
kategori = cell(length(skor_final), 1);
for i = 1:length(skor_final)
    if skor_final(i) >= 70
        kategori{i} = 'Sangat Layak Huni';
    elseif skor_final(i) >= 40
        kategori{i} = 'Cukup Layak Huni';
    else
        kategori{i} = 'Tidak Layak Huni';
    end
end

n_sangat = sum(strcmp(kategori, 'Sangat Layak Huni'));
n_cukup  = sum(strcmp(kategori, 'Cukup Layak Huni'));
n_tidak  = sum(strcmp(kategori, 'Tidak Layak Huni'));

disp('=== DISTRIBUSI KATEGORI ===')
disp(['Sangat Layak Huni : ' num2str(n_sangat) ' planet'])
disp(['Cukup Layak Huni  : ' num2str(n_cukup)  ' planet'])
disp(['Tidak Layak Huni  : ' num2str(n_tidak)  ' planet'])

%% 4.6 RANKING TOP 10
[skor_urut, idx_urut] = sort(skor_final, 'descend');

disp('=== TOP 10 PLANET KANDIDAT LAYAK HUNI ===')
disp('Rank | Nama Planet  | Skor   | Kategori')
disp('-----|--------------|--------|------------------')
for i = 1:min(10, length(skor_urut))
    disp([num2str(i,'%2d') '   | ' ...
          char(nama_planet(idx_urut(i))) '  | ' ...
          num2str(round(skor_urut(i),2),'%6.2f') ' | ' ...
          kategori{idx_urut(i)}])
end

%% 4.7 VISUALISASI PIE CHART
figure;
pie([n_sangat, n_cukup, n_tidak], ...
    {'Sangat Layak Huni','Cukup Layak Huni','Tidak Layak Huni'});
title('Distribusi Kategori Layak Huni — Planet CONFIRMED');
colormap([0.18 0.55 0.34; 0.20 0.40 0.72; 0.85 0.20 0.20]);

%% 4.8 VISUALISASI BAR CHART TOP 10
figure;
top10_nama = cell(10,1);
top10_skor = zeros(10,1);
for i = 1:10
    top10_nama{i} = char(nama_planet(idx_urut(i)));
    top10_skor(i) = skor_urut(i);
end
barh(top10_skor);
yticks(1:10);
yticklabels(top10_nama);
xlabel('Skor Layak Huni (0-100)');
title('Top 10 Planet Kandidat Layak Huni');
xline(70, 'r--', 'Sangat Layak Huni', 'LineWidth', 1.5);
xline(40, 'y--', 'Cukup Layak Huni',  'LineWidth', 1.5);
xlim([0 100]);
grid on;

%% 4.9 VISUALISASI SCATTER ZONA LAYAK HUNI
figure;
scatter(insol_asli, Yhat_all, 20, skor_final, 'filled');
colorbar;
colormap(jet);
hold on;
xline(0.2, 'k--', 'LineWidth', 1.5);
xline(2.0, 'k--', 'LineWidth', 1.5);
yline(200,  'r--', 'LineWidth', 1.5);
yline(400,  'r--', 'LineWidth', 1.5);
xlabel('Insolation Flux koi\_insol (x Bumi)');
ylabel('Prediksi Suhu koi\_teq (K)');
title('Zona Layak Huni — Planet CONFIRMED');
legend('Planet CONFIRMED','Batas Insol','','Batas Suhu','Location','northeast');
grid on;

disp('=== HABITABILITY SCORING SELESAI ===')
disp(['Total planet CONFIRMED : ' num2str(n_pcr)])
disp(['Sangat Layak Huni      : ' num2str(n_sangat) ' planet'])
disp(['Cukup Layak Huni       : ' num2str(n_cukup)  ' planet'])
disp(['Tidak Layak Huni       : ' num2str(n_tidak)  ' planet'])
disp('=== PROJECT SELESAI ===')