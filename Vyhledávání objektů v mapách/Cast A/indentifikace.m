% Start basics
clear, clc;
format long g;
format compact;
close all;


% Image prep
file = "MMC_sk1.jpg"

rgbImage = imread(file);
rgbImage = double(rgbImage);
rgbImage = rgbImage/256;

[rows, columns, numberOfColorBands] = size(rgbImage);

% % Plotting of images
% Original
figure;
imshow(rgbImage, []);
axis on
title('Original')

% Manual input of searched input image
templateWidth = 30;
templateHeight = 82;
smallSubImage = imcrop(rgbImage, [4802, 5705, templateWidth, templateHeight]);

subplot(2, 5, 6)
imshow(smallSubImage(:,:,3), []);
axis on;
title('Vzor 1')

% Multiple examples
smallSubImage2 = imcrop(rgbImage, [6884, 5096, templateWidth, templateHeight]);
smallSubImage3 = imcrop(rgbImage, [4273, 3750, templateWidth, templateHeight]);
smallSubImage4 = imcrop(rgbImage, [5444, 4401, templateWidth, templateHeight]);
smallSubImage5 = imcrop(rgbImage, [7122, 4673, templateWidth, templateHeight]);

% Example mean
templateAll = cat(4,smallSubImage, smallSubImage2, smallSubImage3, smallSubImage4, smallSubImage5);
meanTemplate = mean(templateAll,4);
smallSubImage = meanTemplate;

% Searched Images

subplot(2, 5, [1,2,3,4,5])
imshow(smallSubImage(:,:,3), []);
axis on;
title('Prumer ze vzorku')

subplot(2, 5, 7)
imshow(smallSubImage2(:,:,3), []);
axis on;
title('Vzor 2')

subplot(2, 5, 8)
imshow(smallSubImage3(:,:,3), []);
axis on;
title('Vzor 3')

subplot(2, 5, 9)
imshow(smallSubImage4(:,:,3), []);
axis on;
title('Vzor 4')

subplot(2, 5, 10)
imshow(smallSubImage5(:,:,3), []);
axis on;
title('Vzor 5')


% % Correlation
% Choosing channel, 3 = blue
channleToCorrelate = 3;
correlationOutput = normxcorr2(smallSubImage(:,:,channleToCorrelate), rgbImage(:,:,channleToCorrelate));


%
figure;
imshow(correlationOutput, []);
axis on;
title('Normaovany korelacni koeficient')

% Ammount of correlation
korelace = 0.575;

[yk,xk] = size(correlationOutput);
pocet = 0;
vysledky = [];

for radek=1:yk
    for sloupec=1:xk
        if abs(correlationOutput(radek,sloupec)) > korelace
            pocet=pocet+1;
            vysledky(pocet,1)=radek;
            vysledky(pocet,2)=sloupec;
        end
    end
end

% % Find unique values only
% uniquetol(A,tol)
tol = 1e-3;

vysledky = uniquetol(vysledky,tol, 'ByRows', true);
pocet = size(vysledky, 1);

figure;
imshow(rgbImage(:,:,3), []);
axis on;
for i=1:pocet
    corr_offset = [vysledky(i,2)-size(smallSubImage,2) vysledky(i,1)-size(smallSubImage,1)];
    boxRect = [corr_offset(1) corr_offset(2) templateWidth, templateHeight];
    rectangle("position",boxRect, 'edgecolor','g','linewidth',2);
end
title("Nalezeno " + num2str(pocet) + " vysledku v obrazu")


vysledky
