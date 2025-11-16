 % Start basics

clear, clc;

format long g;

format compact;

close all;

%% Krok 1 - Načtení a převod barev //////////////////////////////

labImage = rgb2lab(imread("TM25_sk1.jpg"));

%% Krok 2 - Priprava dat pro imsegkmeans() ///////////////////////

% Získame pouze barevné kanály a* a b*

abChannels = labImage(:,:,2:3);

% Nastavime počet tříd

k = 2;

%% Krok 3 - segmentace ///////////////////////////////////////////

% Funkce vrátí obrázek, kde každé číslo (1,2,3...)

% představuje ID jedné třídy (les, voda, pole...)

pixelLabels = imsegkmeans(single(abChannels), k);

% Zobrazime si výsledek

figure;

imshow(pixelLabels, []);

title('Výsledek k-Means (segmentované třídy)');

%% Krok 4 - Identifikace clusteru a maska ///////////////////////

idLesa = 2;

forestMask = (pixelLabels == idLesa);

% Zobrazime vysledek

figure;

imshow(forestMask, []);

title('Binarni maska lesa (pred cistenim)');

%% Krok 5.1 - Cisteni sumu a tenkych car ////////////////////////

% 1. Odstrani vsechny male bile objekty (sum) mensi nez 100 pixelu

% To cislo 100 si muzes zmenit, kdyby to bylo malo nebo moc

cleanMask = bwareaopen(forestMask, 50);

cleanMask = imopen(cleanMask, strel('disk',4));

% Zobrazime si mezivysledek

figure;

imshow(cleanMask, []);

title('Maska po odstraneni sumu a car');

%% Krok 5.2 - Plneni der (s vyjimkou velkych) /////////////////////

% 1. Vyplnime VSECHNY diry v masce

filledMask = imfill(cleanMask, 'holes');

% 2. Najdeme, co bylo vyplneno (to jsou ty diry)

% (pixel je dira, pokud je bily ve filledMask, ale cerny v cleanMask)

holes = filledMask & ~cleanMask;

% vyplnime vrstevnice a popisy

largeHoles = imopen(holes, strel('disk',10));

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
disp('První 10 souřadnic:')
disp(vysledneSouradnice(1:10,:)); 

imwrite(finalMask, 'maska_bez_bu.png'); 