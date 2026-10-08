% clear; clc; close all;

% 1. 路徑設定
OUT = "C:\Lab\brain_age_test_retest\output";
PLOT_DIR = fullfile(OUT, "plots_diff");
if ~isfolder(PLOT_DIR)
    mkdir(PLOT_DIR);
end

% 2. 要分析的資料集
DATASETS = ["MMRR", "OAS1"];

% MMRR 已知 motion 受試者
MOTION_SUBJECT_BASE = "sub_742";

OAS1_RESNET_OVERLAP = [
    "OAS1_0061"
    "OAS1_0092"
    "OAS1_0111"
    "OAS1_0150"
    "OAS1_0156"
    "OAS1_0202"
    "OAS1_0230"
    "OAS1_0353"
    "OAS1_0368"
    "OAS1_0379"
];
% 3. 固定 y 軸設定
USE_FIXED_YLIM = true;
COMMON_YLIM = [0 12];

% 4. Model 設定（依指定 x 軸順序）
% 顯示名稱（第 1 欄）已將 GMP/WMP 改為 GM/WM；Sheet 辨識名稱（第 3 欄）維持原 Excel 命名
CFG = {
    "SFCN-reg T1WI",    "SFCN_reg.xlsx", "T1WI";
    "SFCN-reg GM",      "SFCN_reg.xlsx", "GMP";      % 圖上顯示改為 GM
    "SFCN-reg WM",      "SFCN_reg.xlsx", "WMP";      % 圖上顯示改為 WM
    "SFCN-sm T1WI",     "SFCN_sm.xlsx",  "T1WI";
    "SFCN-sm GM",       "SFCN_sm.xlsx",  "GMP";      % 圖上顯示改為 GM
    "SFCN-sm WM",       "SFCN_sm.xlsx",  "WMP";      % 圖上顯示改為 WM
    "ResNet T1",        "ResNet_batch1.xlsx",   "T1";
    "ResNet T1 B0",     "ResNet_batch1.xlsx",   "T1_B0";
    "ResNet T1 SEX",    "ResNet_batch1.xlsx",   "T1_SEX";
    "ResNet T1 SEX B0", "ResNet_batch1.xlsx",   "T1_SEX_B0";
};
nModels = size(CFG, 1);

% 5. 顏色設定
MODEL_COLORS = [
    0.00 0.45 0.15;   % SFCN-reg T1WI   深綠
    0.20 0.65 0.25;   % SFCN-reg GM     中綠
    0.55 0.82 0.55;   % SFCN-reg WM     淺綠
    0.70 0.00 0.00;   % SFCN-sm T1WI    深紅
    0.88 0.28 0.28;   % SFCN-sm GM      中紅
    0.97 0.65 0.65;   % SFCN-sm WM      淺紅
    0.00 0.25 0.75;   % ResNet T1       深藍
    0.15 0.45 0.90;   % ResNet T1 B0    中深藍
    0.40 0.65 0.95;   % ResNet T1 SEX   中淺藍
    0.72 0.84 0.98;   % ResNet T1 SEX B0 淺藍
];

% 6. 逐一分析 MMRR 與 OAS1
for d = 1:numel(DATASETS)
    db = DATASETS(d);
    
    diffData   = cell(nModels, 1);
    motionFlag = cell(nModels, 1);
    
    Model   = strings(nModels, 1);
    N       = zeros(nModels, 1);
    MAD     = nan(nModels, 1);
    MinDiff = nan(nModels, 1);
    MaxDiff = nan(nModels, 1);
    
    for m = 1:nModels
        Model(m) = CFG{m,1};
        file = fullfile(OUT, CFG{m,2});
        sheet = db + "_" + CFG{m,3};   % Excel 命名格式：資料集_模型
        
        if ~isfile(file)
            warning("找不到檔案：%s", file);
            continue;
        end
        if ~ismember(sheet, string(sheetnames(file)))
            warning("找不到工作表：%s | %s", file, sheet);
            continue;
        end
        
        % 讀取並建立 scan-rescan pair
        % 讀取並建立 scan-rescan pair
        R = loadRetestDifference(file, sheet, db, MOTION_SUBJECT_BASE);
        
        % ============================================================
        % OASIS-1:
        % ResNet training data 與 10 位 test-retest subjects overlap
        % Primary repeatability analysis 排除這 10 位 subject 的完整 pair
        % SFCN 不受影響，仍使用完整 OAS1 test-retest dataset
        % ============================================================
        if db == "OAS1" && startsWith(Model(m), "ResNet")
        
            isOverlap = ismember( ...
                upper(string(R.SubjectID)), ...
                upper(OAS1_RESNET_OVERLAP));
        
            fprintf( ...
                "%s | OAS1 ResNet: excluding %d overlapping subjects\n", ...
                Model(m), sum(isOverlap));
        
            R = R(~isOverlap, :);
        end
        
        diffData{m}   = R.AbsDiff;
        motionFlag{m} = R.IsMotionPair;
        N(m)          = height(R);
        
        if ~isempty(R.AbsDiff)
            MAD(m)     = mean(R.AbsDiff, 'omitnan');
            MinDiff(m) = min(R.AbsDiff, [], 'omitnan');
            MaxDiff(m) = max(R.AbsDiff, [], 'omitnan');
        end
    end
    
    % 7. 繪製 absolute difference 分布圖
    f = figure('Color', 'w', 'Position', [100 100 1650 720]);
    hold on;
    
    for m = 1:nModels
        y = diffData{m};
        if isempty(y)
            continue;
        end
        n = numel(y);
        
        % 固定 jitter 分布
        if n == 1
            jitter = 0;
        else
            jitter = linspace(-0.20, 0.20, n)';
        end
        x = m + jitter;
        
        isMotion = motionFlag{m};
        isNormal = ~isMotion;
        
        % 一般受試者點
        scatter(x(isNormal), y(isNormal), 44, ...
            'Marker', 'o', ...
            'MarkerFaceColor', MODEL_COLORS(m,:), ...
            'MarkerEdgeColor', MODEL_COLORS(m,:), ...
            'MarkerFaceAlpha', 0.65, ...
            'MarkerEdgeAlpha', 0.80);
        
        % MMRR motion 受試者 (sub_742)
        if db == "MMRR" && any(isMotion)
            scatter(x(isMotion), y(isMotion), 130, ...
                'Marker', 'p', ...
                'MarkerFaceColor', [1.00 0.70 0.15], ...
                'MarkerEdgeColor', [0.30 0.30 0.30], ...
                'LineWidth', 1.0);
        end
        
        % MAD 黑色虛線
        madValue = MAD(m);
        if ~isnan(madValue)
            plot([m-0.23, m+0.23], [madValue, madValue], '--', ...
                'Color', [0 0 0], 'LineWidth', 2.2);
            
            % MAD 數值標註
            text(m, madValue + 0.18, sprintf('%.2f', madValue), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 10, ...
                'FontWeight', 'bold', ...
                'Color', [0 0 0]);
        end
    end
    
    % 8. 圖形軸與標籤設定
    xlim([0.45, nModels + 0.55]);
    xticks(1:nModels);
    xticklabels(Model);
    xtickangle(35);
    ylabel('|Scan 1 - Scan 2| (years)', 'FontSize', 15);
    xlabel('Model', 'FontSize', 15);
    title(char(db), 'FontSize', 20, 'FontWeight', 'bold');
    
    if USE_FIXED_YLIM
        ylim(COMMON_YLIM);
        yticks(COMMON_YLIM(1):1:COMMON_YLIM(2));
    else
        allY = vertcat(diffData{:});
        if ~isempty(allY)
            yTop = max(allY, [], 'omitnan');
            ylim([0, max(1, ceil(yTop * 1.12))]);
        end
    end
    
    set(gca, ...
        'FontSize', 12, ...
        'LineWidth', 1.0, ...
        'TickDir', 'out');
    grid on;
    box on;
    
    % Legend 設定
    hMAD = plot(nan, nan, '--k', 'LineWidth', 2.2);
    if db == "MMRR"
        hMotion = scatter(nan, nan, 120, 'p', 'filled', ...
            'MarkerFaceColor', [1.00 0.70 0.15], ...
            'MarkerEdgeColor', [0.30 0.30 0.30], ...
            'LineWidth', 1.0);
        legend([hMAD, hMotion], {'MAD', 'sub\_742 (motion)'}, ...
            'Location', 'northeast');
    else
        legend(hMAD, {'MAD'}, 'Location', 'northeast');
    end
    
    % 9. 儲存輸出
    figPath = fullfile(PLOT_DIR, db + "_AbsoluteDifference_Distribution.png");
    exportgraphics(f, figPath, 'Resolution', 300);
    
    AnalysisSet = strings(nModels,1);

    for m = 1:nModels
        if db == "OAS1" && startsWith(Model(m), "ResNet")
            AnalysisSet(m) = "Independent subset";
        else
            AnalysisSet(m) = "Full test-retest set";
        end
    end
    
    Summary = table( ...
        Model, ...
        AnalysisSet, ...
        N, ...
        MAD, ...
        MinDiff, ...
        MaxDiff);
    csvPath = fullfile(PLOT_DIR, db + "_AbsoluteDifference_Summary.csv");
    writetable(Summary, csvPath);
    
    fprintf('\n========== %s absolute difference summary ==========\n', db);
    disp(Summary);
    fprintf('圖檔輸出：%s\n', figPath);
    fprintf('報表輸出：%s\n', csvPath);
end

% ================================================================
%  Local function: 配對 Scan-Rescan 並計算 Absolute Difference
% ================================================================
function R = loadRetestDifference(file, sheet, db, motionSubjectBase)
    T = readtable(file, 'Sheet', sheet, 'VariableNamingRule', 'preserve');
    if width(T) < 3
        error('%s | %s：Excel 至少需要前四欄資料。', file, sheet);
    end
    
    T = T(:, 1:3);
    T.Properties.VariableNames = ...
        {'SubjectID', 'TrueAge', 'PredictedAge'};
    
    ok = ~ismissing(T.PredictedAge) & ...
         strlength(strtrim(string(T.SubjectID))) > 0;
    T = T(ok,:);
    
    id   = string(T.SubjectID);
    pred = T.PredictedAge;
    
    if db == "MMRR"
        base = regexprep(id, '_[^_]+$', '');
    elseif db == "OAS1"
        base = regexprep(id, '(?i)_MR[12]$', '');
    else
        error('此函式目前只支援 MMRR 與 OAS1。');
    end
    
    subjects = unique(base, 'stable');
    
    SubjectID    = strings(0,1);
    Scan1ID      = strings(0,1);
    Scan2ID      = strings(0,1);
    Scan1        = zeros(0,1);
    Scan2        = zeros(0,1);
    AbsDiff      = zeros(0,1);
    IsMotionPair = false(0,1);
    
    for i = 1:numel(subjects)
        idx = find(base == subjects(i));
        if numel(idx) < 2
            continue;
        end
        
        if db == "OAS1"
            idx1 = idx(endsWith(upper(id(idx)), '_MR1'));
            idx2 = idx(endsWith(upper(id(idx)), '_MR2'));
            if ~isempty(idx1) && ~isempty(idx2)
                i1 = idx1(1);
                i2 = idx2(1);
            else
                i1 = idx(1);
                i2 = idx(2);
            end
        else
            i1 = idx(1);
            i2 = idx(2);
        end
        
        SubjectID(end+1,1) = subjects(i);
        Scan1ID(end+1,1)   = id(i1);
        Scan2ID(end+1,1)   = id(i2);
        Scan1(end+1,1)     = pred(i1);
        Scan2(end+1,1)     = pred(i2);
        AbsDiff(end+1,1)   = abs(pred(i1) - pred(i2));
        IsMotionPair(end+1,1) = (subjects(i) == motionSubjectBase);
    end
    
    R = table(SubjectID, Scan1ID, Scan2ID, Scan1, Scan2, AbsDiff, IsMotionPair);
end