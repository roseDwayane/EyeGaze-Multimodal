% 創建一個 1920x1080 的空白圖片 (黑色背景)
image_width = 1920;
image_height = 1080;
image = zeros(image_height, image_width, 3); % RGB 圖片 (黑色背景)

trial = 5;

% 讀取 Excel 文件數據
data = readtable('G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\位置ans.xlsx');

% 數字資料
%numbers = [12, 8, 4, 12, 12, 6, 1, 9, 3, 7, 4, 1, 7, 6, 8, 10];
numbers = myxdf{1,2}.time_series(1:17,trial)';

% 假設的 gaze_position 資料 (取自 EEG)
x1 = EEG.data(1,:,trial)'; % X 軸資料
y1 = EEG.data(2,:,trial)'; % Y 軸資料

x2 = EEG.data(3,:,trial)'; % X 軸資料
y2 = EEG.data(4,:,trial)'; % Y 軸資料

% 繪製方塊和軌跡
%figure;
figure('Color', 'k', 'Position', [0, 0, image_width, image_height]); % 黑色背景，設定視窗尺寸
axes('Position', [0, 0, 1, 1]); % 填滿整個畫布
imshow(image, 'InitialMagnification', 'fit'); % 顯示圖片，保持比例
%axis on;
hold on; % 保持繪圖

% 設置軸範圍 (將原點設置到左下角)
xlim([1, image_width]);
ylim([1, image_height]);
set(gca, 'YDir', 'normal'); % 將 y 軸方向設為正向

% 繪製方塊
for i = 1:height(data)
    % 獲取方塊頂點
    vertices_x = [data.x___x(i), data.x___x_1(i), data.x___x_2(i), data.x___x_3(i), data.x___x(i)];
    vertices_y = [data.x___y(i), data.x___y_1(i), data.x___y_2(i), data.x___y_3(i), data.x___y(i)];
    
    % 繪製紅色方塊
    %fill(vertices_x, vertices_y, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    
    % 計算方塊中心位置
    center_x = mean(vertices_x(1:4));
    center_y = mean(vertices_y(1:4));
    
    % 計算新的數字
    new_value = numbers(i) + 1;
    
    % 將 11, 12, 13 替換為 J, Q, K
    if new_value == 11
        text_value = 'J';
    elseif new_value == 12
        text_value = 'Q';
    elseif new_value == 13
        text_value = 'K';
    elseif new_value == 1
        text_value = 'A';
    else
        text_value = num2str(new_value);
    end
    
    % 在方塊中心填入數字或字母
    text(center_x, center_y, text_value, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontSize', 48, 'Color', 'white', 'FontWeight', 'bold');
end

% 儲存圖片為正確大小
frame = getframe(gca); % 擷取目前繪圖的框架
bmp_image = imresize(frame.cdata, [image_height, image_width]); % 強制調整尺寸
imwrite(bmp_image, 'output.bmp'); % 儲存成 BMP 格式
disp('圖片已成功儲存為 output.bmp');


% 繪製眼動軌跡
%plot(x1 + 960, 540 - y1, 'b', 'LineWidth', 1.5); % 第一條軌跡 (藍色)
%plot(x2 + 960, 540 - y2, 'g', 'LineWidth', 1.5); % 第二條軌跡 (綠色)

%hold off;

% 添加標題和軸標籤
%title('1920x1080 圖片中的方塊與眼動軌跡');
%xlabel('X 軸 (1:1920)');
%ylabel('Y 軸 (1:1080)');
