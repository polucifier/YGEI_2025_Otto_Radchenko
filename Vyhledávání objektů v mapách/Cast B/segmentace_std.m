% start basics
clear, clc;
format long g;
format compact;
close all;

%% krok 1 - nacteni a prevod barev //////////////////////////////
% nacteni obrazku a prevod z rgb do barevneho prostoru l*a*b*
labImage = rgb2lab(imread("TM25_sk1.jpg"));

%% krok 2.1 - priprava std filtru
% std filtr funguje nejlepe na kanalu svetlosti (grayscale)
% pouzijeme l* kanal (svetlost) z naseho l*a*b* obrazku
L_channel = labImage(:,:,1);
% a* kanal (zelena-cervena) si nechame pro krok 4
a_channel = labImage(:,:,2); 

% definujeme sousedstvi (masku) pro vypocet standardni odchylky
% velikost sousedstvi (zde 9x9) ovlivnuje, jak "hrubou" texturu filtr zachyti
nhood = ones(9, 9); 

% vypocet lokalni standardni odchylky (textury) pro kazdy pixel
% vystup 'stdfeatures' bude matice m x n x 1
% popisuje lokalni "hrubost" textury v danem miste
stdFeatures = stdfilt(L_channel, nhood);

%% krok 2.2 - priprava dat pro imsegkmeans() ///////////////////////
% ziskame pouze barevne kanaly a* a b*
abChannels = labImage(:,:,2:3);

% spojime 2 barevne kanaly (a*, b*) s 1 kanalem textury (std)
% vysledkem je matice m x n x 3 (3 priznaky pro kazdy pixel)
allFeatures = cat(3, abChannels, stdFeatures);

% kanaly a*b* a std kanal maji ruzne rozsahy hodnot
% pro spravnou funkci k-means je treba priznaky normalizovat (standardizovat)
% pouzijeme z-skore (prumer 0, smer. odchylka 1 pro kazdy priznak)
[M, N, F] = size(allFeatures); % f = 3 v nasem pripade (a*, b*, std)

% prevedeme 3d matici (obraz) na 2d matici (seznam pixelu)
features_2d = reshape(single(allFeatures), M*N, F);

% vypocet z-skore pro kazdy priznak (sloupec)
features_scaled = zscore(features_2d);

% prevedeme normalizovana data zpet do 3d tvaru (obrazu)
features_scaled_3d = reshape(features_scaled, M, N, F);

% nastavime pocet trid (clusteru)
k = 2; 

%% krok 3 - segmentace ///////////////////////////////////////////
% nyni segmentujeme s vyuzitim vsech 3 priznaku (a*, b*, std)
% pouzivame 'features_scaled_3d' misto puvodniho obrazku
pixelLabels = imsegkmeans(features_scaled_3d, k, 'NumAttempts', 5); 
% 'numattempts' zvysuje sanci na nalezeni optimalniho vysledku

% zobrazeni vysledku segmentace (mapa trid)
figure;
imshow(pixelLabels, []); 
title('vysledek k-means (segmentovane tridy)');
%imwrite(uint8(pixelLabels),'segmentovane_tridy.png')

%% krok 4 - identifikace clusteru a maska ///////////////////////
% pro kazdy cluster (tridu) spocitame prumernou hodnotu a* (zelenost)
clusterMeans = zeros(k,1);
for i=1:k
    clusterMeans(i) = mean(a_channel(pixelLabels==i));
end

% cluster s nejnizsi hodnotou a* (nejvice zeleno-modry) je les
[~, idLesa] = min(clusterMeans);

% vytvorime binarni masku lesa
forestMask = (pixelLabels==idLesa);

% zobrazeni binarni masky lesa (pred cistenim)
figure;
imshow(forestMask, []);
title('binarni maska lesa (pred cistenim)');

%% krok 5.1 - cisteni sumu a tenkych car ////////////////////////
% odstraneni malych bilych objektu (sumu) mensich nez 100 pixelu
% (hodnotu 100 lze upravit podle potreby)
cleanMask = bwareaopen(forestMask, 100);

% morfologicke otevreni (imopen) odstrani tenke cary a "chlupy"
cleanMask = imopen(cleanMask, strel('disk',4));

% zobrazeni masky po prvnim cisteni
figure;
imshow(cleanMask, []);
title('maska po odstraneni sumu a car');

%% krok 5.2 - plneni der (s vyjimkou velkych) /////////////////////
% vyplnime vsechny diry v masce (i velke pruseky)
filledMask = imfill(cleanMask, 'holes');

% najdeme, ktere pixely byly prave vyplneny
% (je to rozdil mezi vyplnenou a puvodni maskou)
holes = filledMask & ~cleanMask;

% najdeme jen velke diry (pruseky) tim, ze odstranime male vyplnene diry
% 'disk',13) je zde pouzito k odfiltrovani mensich der
largeHoles = imopen(holes, strel('disk',13));
% (alternativa: pouzit bwareaopen pro filtraci podle poctu pixelu)
% largeHoles = bwareaopen(holes, 50);

% z kompletne vyplnene masky "odebereme" velke diry
finalMask = filledMask & ~largeHoles;

% zobrazeni finalni masky lesa
figure;
imshow(finalMask, []);
title('finalni maska lesa (po vyplneni der)');

%% krok 6 - export pixelovych souradnic ////////////////////////////
% najdeme radkove a sloupcove souradnice vsech bilych pixelu
[rows, cols] = find(finalMask);

% spojime souradnice do jedne matice [radky, sloupce]
vysledneSouradnice = [rows, cols];

% vypiseme si prvnich 10
disp(vysledneSouradnice(1:10,:));

%% --- Krok 7 - oriznuti a ulozeni //////////////////////////////////
x_min = 362;
x_max = 4560;
y_min = 465;
y_max = 4840;
% vytvoreni prekryvneho obrazku pro vizualni kontrolu
overlay = labeloverlay(imread("TM25_sk1.jpg"), finalMask, "Colormap",[0,1,0],"Transparency",0.7);
overlay = overlay(y_min:y_max, x_min:x_max, :);
% ulozeni prekryvneho obrazku
imwrite(overlay,'overlay.jpg');

% ulozeni finalni binarni masky
finalMask = finalMask(y_min:y_max, x_min:x_max);
imwrite(finalMask, 'maska_std.png');