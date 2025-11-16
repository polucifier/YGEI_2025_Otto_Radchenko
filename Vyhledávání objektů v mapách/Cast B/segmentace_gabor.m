% Start basics
clear, clc;
format long g;
format compact;
close all;

%% Krok 1 - Načtení a převod barev //////////////////////////////
labImage = rgb2lab(imread("TM25_sk1.jpg"));

%% Krok 2.1 - Příprava Gaborových filtrů
% Gaborovy filtry fungují nejlépe na kanálu světlosti (grayscale)
% Použijeme L* kanál z našeho L*a*b* obrázku
L_channel = labImage(:,:,1);
a_channel = labImage(:,:,2);

% Vytvoříme sadu ("banku") Gaborových filtrů
% Můžete experimentovat s počtem směrů (orientations) a "velikostí" 
% textury (wavelengths - vlnové délky)
orientations = 0:45:135;  % 4 směry (0, 45, 90, 135 stupňů)
wavelengths = [8 12 16];   % 3 různé vlnové délky (od jemné po hrubou texturu)

% Tímto příkazem vytvoříme banku filtrů
g = gabor(wavelengths, orientations);

% Aplikujeme filtry na L* kanál. 
% Výstup 'gaborMag' bude matice M x N x 12 (protože 4 směry * 3 vlnové délky = 12 filtrů)
% 'gaborMag' popisuje "sílu" textury v daném místě pro každý z 12 filtrů
[gaborMag, ~] = imgaborfilt(L_channel, g);
%% Krok 2.2 - Priprava dat pro imsegkmeans() ///////////////////////
% Získame pouze barevné kanály a* a b*
abChannels = labImage(:,:,2:3);

% Spojíme 2 barevné kanály (a*, b*) s 12 Gaborovými kanály (textura)
% Výsledkem je matice M x N x 14 (14 příznaků pro každý pixel)
allFeatures = cat(3, abChannels, gaborMag);

% Kanály a*b* mají jiný rozsah hodnot (cca -100 až 100) než Gaborova magnituda (cca 0 až X).
% Aby k-Means dávalo všem příznakům stejnou váhu, musíme je normalizovat (standardizovat).
% Použijeme z-skóre (průměr 0, směr. odchylka 1 pro každý příznak).
[M, N, F] = size(allFeatures); % F = 14 v našem případě

% Převedeme 3D matici M x N x F na 2D matici (M*N) x F (pixely x příznaky)
features_2d = reshape(single(allFeatures), M*N, F);

% Vypočítáme z-skóre pro každý příznak (sloupec)
features_scaled = zscore(features_2d);

% Převedeme normalizovaná data zpět do 3D tvaru M x N x F
features_scaled_3d = reshape(features_scaled, M, N, F);

% Nastavime počet tříd
k = 6; 

%% Krok 3 - segmentace ///////////////////////////////////////////
% Nyní segmentujeme s využitím VŠECH 14 příznaků (barva + textura)
% Používáme 'features_scaled_3d' místo 'single(abChannels)'
pixelLabels = imsegkmeans(features_scaled_3d, k, 'NumAttempts', 5); 
% Přidali jsme 'NumAttempts' pro robustnější výsledek

% Zobrazime si výsledek
figure;
imshow(pixelLabels, []); 
title('Výsledek k-Means (segmentované třídy)');
%imwrite(uint8(pixelLabels),'segmentovane_tridy.png')

%% Krok 4 - Identifikace clusteru a maska ///////////////////////

clusterMeans = zeros(k,1);
for i=1:k
    clusterMeans(i) = mean(a_channel(pixelLabels==i));
end
[~, idLesa] = min(clusterMeans);
forestMask = (pixelLabels==idLesa);

% Zobrazime vysledek
figure;
imshow(forestMask, []);
title('Binarni maska lesa (pred cistenim)');

%% Krok 5.1 - Cisteni sumu a tenkych car ////////////////////////
% 1. Odstrani vsechny male bile objekty (sum) mensi nez 100 pixelu
% To cislo 100 si muzes zmenit, kdyby to bylo malo nebo moc
cleanMask = bwareaopen(forestMask, 100);

cleanMask = imopen(cleanMask, strel('disk',4));
% Zobrazime si mezivysledek
figure;
imshow(cleanMask, []);
title('Maska po odstraneni sumu a car');

%% Krok 5.2 - Plneni der (s vyjimkou velkych) /////////////////////
%odstraneni mrizky a malych cest
closedMask = imclose(cleanMask, strel('disk', 5));

% 1. Vyplnime VSECHNY diry v masce
filledMask = imfill(closedMask, 'holes');

% 2. Najdeme, co bylo vyplneno (to jsou ty diry)
% (pixel je dira, pokud je bily ve filledMask, ale cerny v cleanMask)
holes = filledMask & ~cleanMask;

% vyplnime vrstevnice a popisy
largeHoles = imopen(holes, strel('disk',5));

% 4. Z finalni masky tyhle velke diry zase "vyvrtame" zpet
finalMask = filledMask & ~largeHoles;

% Zobrazime finalni vysledek.
figure;
imshow(finalMask, []);
title('Finalni maska lesa (po vyplneni der)');

%% Krok 6 - Export pixelovych souradnic ////////////////////////////

[rows, cols] = find(finalMask);

% Spojime to do jedne matice N x 2 (N = pocet pixelu)
vysledneSouradnice = [rows, cols];

% vypiseme si prvnich 10
%disp(vysledneSouradnice(1:10,:));
overlay = labeloverlay(imread("TM25_sk1.jpg"), finalMask, "Colormap",[0,1,0],"Transparency",0.7);
%imwrite(overlay,'overlay.jpg');
imwrite(finalMask, 'maska_gabor.png');
