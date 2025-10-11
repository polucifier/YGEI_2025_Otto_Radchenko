clc
clear
format long g

%Load 
fig = imread('Image2.bmp');
imshow(fig);

%RGB components
R = double(fig(:,:,1));
G = double(fig(:,:,2));
B = double(fig(:,:,3));

%%%%%%%%%%%%%%%%%%%%%%%%%
%JPEG compression
%%%%%%%%%%%%%%%%%%%%%%%%%

%convert RGB to YCbCr
Y = 0.2990*R + 0.5870*G + 0.1140*B;
Cb = -0.1687*R - 0.3313*G + 0.5000*B + 128;
Cr = 0.5000*R - 0.4187*G - 0.0813*B + 128;

%Interval transformation
Y = 2*Y - 255;
Cb = 2*Cb - 255;
Cr = 2*Cr - 255;

%Raster resampling - pouze pro barevné složky - 1. ztratovy krok
Cb = resample_chrominance_2x2(Cb);
Cr = resample_chrominance_2x2(Cr);

%Quantisation matrix: Y
Qy = [16 11 10 16 24 40 51 61;
12 12 14 19 26 58 60 55;
14 13 16 24 40 87 69 56;
14 17 22 29 51 87 80 62;
18 22 37 26 68 109 103 77;
24 35 55 64 81 104 113 92;
49 64 78 87 103 121 120 101;
72 92 95 98 112 100 103 99];

%Quantisation matrix: Cb, Cr
Qc = [17 18 24 47 66 99 99 99
18 21 26 66 99 99 99 99
24 26 56 99 99 99 99 99
47 69 99 99 99 99 99 99
99 99 99 99 99 99 99 99
99 99 99 99 99 99 99 99
99 99 99 99 99 99 99 99
99 99 99 99 99 99 99 99];

%Compression factor
q = 50;
Qyf = 50*Qy/q;
Qcf = 50*Qc/q;

%division to submatrices
[m,n] = size(Y);

%Process lines
for i = 1:8:m-7
    %Process columns
    for j = 1:8:n-7
        %Get submatrices
        Ys = Y(i:i+7, j:j+7);
        Cbs = Cb(i:i+7, j:j+7);
        Crs = Cr(i:i+7, j:j+7);

        %Apply DCT
        Ys_dct = dct(Ys);
        Cbs_dct = dct(Cbs);
        Crs_dct = dct(Crs);

        %Quantization - 2. ztratovy krok
        Ys_q = round(Ys_dct ./ Qyf);
        Cbs_q = round(Cbs_dct ./ Qcf);
        Crs_q = round(Crs_dct ./ Qcf);

        %Update transformed matrix
        Y(i:i+7,j:j+7) = Ys_q;
        Cb(i:i+7,j:j+7) = Cbs_q;
        Cr(i:i+7,j:j+7) = Crs_q;
    end
end

%Codebook generation
codebookY = huffman_codebook(Y(:));
codebookCb = huffman_codebook(Cb(:));
codebookCr = huffman_codebook(Cr(:));

%Zig-Zag + Huffman coding
Ys_encoded = {};
Cbs_encoded = {};
Crs_encoded = {};
%Process lines
for i = 1:8:m-7
    %Process columns
    for j = 1:8:n-7
        %Get submatrices
        Ys = Y(i:i+7, j:j+7);
        Cbs = Cb(i:i+7, j:j+7);
        Crs = Cr(i:i+7, j:j+7);

        %Convertion to ZIG-ZAG sequence
        Ys_zigzag = zigzag(Ys);
        Cbs_zigzag = zigzag(Cbs);
        Crs_zigzag = zigzag(Crs);

        %Huffman encoding
        Ys_encoded{end+1} = huffman_encode(Ys_zigzag,codebookY);
        Cbs_encoded{end+1} = huffman_encode(Cbs_zigzag,codebookCb);
        Crs_encoded{end+1} = huffman_encode(Crs_zigzag,codebookCr);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%JPEG decompression
%%%%%%%%%%%%%%%%%%%%%%%%%

k = 1; % index for encoded data
%Process lines
for i = 1:8:m-7
    %Process columns
    for j = 1:8:n-7
        %Huffman decoding
        Ys_zigzag = huffman_decode(Ys_encoded{k},codebookY);
        Cbs_zigzag = huffman_decode(Cbs_encoded{k},codebookCb);
        Crs_zigzag = huffman_decode(Crs_encoded{k},codebookCr);
        k = k + 1;

        %Convertion from ZIG-ZAG sequence to matrix
        Ys = inverse_zigzag(Ys_zigzag);
        Cbs = inverse_zigzag(Cbs_zigzag);
        Crs = inverse_zigzag(Crs_zigzag);

        %Dequantization
        Ys_dq = Ys .* Qyf;
        Cbs_dq = Cbs .* Qcf;
        Crs_dq = Crs .* Qcf;

        %Apply IDCT
        Ys_idct = idct(Ys_dq);
        Cbs_idct = idct(Cbs_dq);
        Crs_idct = idct(Crs_dq);

        %Update transformed matrix
        Y(i:i+7,j:j+7) = Ys_idct;
        Cb(i:i+7,j:j+7) = Cbs_idct;
        Cr(i:i+7,j:j+7) = Crs_idct;
    end
end

%Interval transformation
Y = (Y + 255)/2;
Cb = (Cb + 255)/2;
Cr = (Cr + 255)/2;

%YCbCr -> RGB
Rd = Y + 1.4020*(Cr-128);
Gd = Y - 0.3441*(Cb-128) - 0.7141*(Cr-128);
Bd = Y + 1.7720*(Cb-128) - 0.0001*(Cr-128);

%Convert double to uint8
Ri=uint8(Rd);
Gi=uint8(Gd);
Bi=uint8(Bd);

%Assemble RGB image
imgj(:,:,1)=Ri;
imgj(:,:,2)=Gi;
imgj(:,:,3)=Bi;

%Show compressed image
figure, imshow(imgj);

%Standard deviations for RGB components
dR = R - Rd;
dG = G - Gd;
dB = B - Bd;

R2 = dR.^2;
G2 = dG.^2;
B2 = dB.^2;

sigmaR = sqrt(sum(R2(:))/(m*n))
sigmaG = sqrt(sum(G2(:))/(m*n))
sigmaB = sqrt(sum(B2(:))/(m*n))

%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions
%%%%%%%%%%%%%%%%%%%%%%%%%

function [imgt] = dct(img)

imgt = img;

%Process lines
for u = 0:7

    %compute cu
    if (u == 0)
        cu = 2^(0.5)/2;
    else
        cu = 1;
    end

    %Process columns
    for v = 0:7

        %compute cv
        if (v == 0)
            cv = 2^(0.5)/2;
        else
            cv = 1;
        end

        %Process lines
        fuv = 0;
        for x = 0:7
            %Process columns
            for y = 0:7
                fuv = fuv+0.25*cu*cv*img(x+1,y+1)*...
                    cos((2*x+1)*u*pi/16)*cos((2*y+1)*v*pi/16);
            end
        end
        %Update raster
        imgt(u+1,v+1)=fuv;
    end
end
end

function [imgt] = idct(img)

imgt = img;

%Process lines
for x = 0:7

    %Process columns
    for y = 0:7

        %Process lines
        fuv = 0;
        for u = 0:7
            %compute cu
            if (u == 0)
                cu = 2^(0.5)/2;
            else
                cu = 1;
            end

            %Process columns
            for v = 0:7
                %compute cv
                if (v == 0)
                    cv = 2^(0.5)/2;
                else
                    cv = 1;
                end
                fuv = fuv+0.25*cu*cv*img(u+1,v+1)*...
                    cos((2*x+1)*u*pi/16)*cos((2*y+1)*v*pi/16);
            end
        end
        %Update raster
        imgt(x+1,y+1)=fuv;
    end
end
end

function C_resampled = resample_chrominance_2x2(C)
    [m, n] = size(C);
    % Připravíme výslednou matici
    C_resampled = zeros(m, n);
    
    % Projdeme políčka 2x2 a spočítáme průměr
    for i = 1:2:m-1
        for j = 1:2:n-1
            block = C(i:i+1, j:j+1);
            C_resampled(i:i+1, j:j+1) = mean(block(:));
        end
    end
end

function Z = zigzag(M)
    [rows, cols] = size(M);
    Z = zeros(1, rows * cols);
    index = 1;
    
    for s = 0:(rows + cols - 2)
        if mod(s, 2) == 0
            % Odd sum index: traverse from top-right to bottom-left
            for j = max(0, s - rows + 1):min(s, cols - 1)
                i = s - j;
                if i < rows
                    Z(index) = M(i + 1, j + 1);
                    index = index + 1;
                end
            end
        else
            % Even sum index: traverse from bottom-left to top-right
            for i = max(0, s - cols + 1):min(s, rows - 1)
                j = s - i;
                if j < cols
                    Z(index) = M(i + 1, j + 1);
                    index = index + 1;
                end
            end
        end
    end
end

function M = inverse_zigzag(Z)
    n = sqrt(length(Z));
    M = zeros(n, n);
    index = 1;
    
    for s = 0:(n + n - 2)
        if mod(s, 2) == 0
            % Odd sum index: traverse from top-right to bottom-left
            for j = max(0, s - n + 1):min(s, n - 1)
                i = s - j;
                if i < n
                    M(i + 1, j + 1) = Z(index);
                    index = index + 1;
                end
            end
        else
            % Even sum index: traverse from bottom-left to top-right
            for i = max(0, s - n + 1):min(s, n - 1)
                j = s - i;
                if j < n
                    M(i + 1, j + 1) = Z(index);
                    index = index + 1;
                end
            end
        end
    end
end

function codebook = huffman_codebook(data)
    % Vstup: data - vektor celých čísel (symbolů)
    % Výstup: codebook - struktura se symboly a kódy
    
    % Najdi unikátní symboly a jejich četnosti
    symbols = unique(data);
    counts = zeros(size(symbols));
    for i = 1:length(symbols)
        counts(i) = sum(data == symbols(i));
    end

    % Vytvoř základní uzly pro každý symbol
    nodes = struct('symbol', num2cell(symbols), ...
                   'count', num2cell(counts), ...
                   'left', [], ...
                   'right', [], ...
                   'parent', []);

    % Seznam uzlů, které nejsou sloučeny
    active = num2cell(1:length(nodes));
    tree = nodes;
    
    % Rekurentní spojování uzlů
    while length(active) > 1
        % Najdi dva uzly s nejmenší četností
        counts_active = [tree([active{:}]).count];
        [~, idx] = sort(counts_active);
        a = active{idx(1)};
        b = active{idx(2)};
        
        % Slouč do nového "rodiče" (předka)
        newnode.symbol = [];
        newnode.count = tree(a).count + tree(b).count;
        newnode.left = a;
        newnode.right = b;
        newnode.parent = [];
        tree(end+1) = newnode;
        
        % Nastav rodiče původním uzlům
        tree(a).parent = length(tree);
        tree(b).parent = length(tree);

        % Aktualizuj aktivní seznam
        active([idx(1) idx(2)]) = [];
        active{end+1} = length(tree);
    end

    % Projeď strom zpět a přidej kódy
    codebook = struct('symbol', {}, 'code', {});
    function traverse(node_idx, code)
        if isempty(tree(node_idx).left) && isempty(tree(node_idx).right)
            codebook(end+1).symbol = tree(node_idx).symbol;
            codebook(end).code = code;
        else
            traverse(tree(node_idx).left, [code, '1']);
            traverse(tree(node_idx).right, [code, '0']);
        end
    end

    traverse(length(tree), '');
end

function encoded = huffman_encode(data, codebook)
    encoded = ''; % Výsledný binární řetězec ve formě stringu
    for i = 1:length(data)
        idx = find([codebook.symbol] == data(i), 1); % Najdi index odpovídajícího symbolu v codebooku
        encoded = [encoded codebook(idx).code]; % Připoj kód k výstupnímu binárnímu řetězci
    end
end

function decoded = huffman_decode(encoded, codebook)
    decoded = []; % Výstupní vektor dekomprimovaných symbolů
    idx = 1; % Aktuální pozice v binárním řetězci
    while idx <= length(encoded)
        found = false;
        % Procházej všechny položky v codebooku a hledej odpovídající prefix
        for i = 1:length(codebook)
            L = length(codebook(i).code);
            % Porovnej aktuální úsek kódu se záznamem v codebooku
            if idx+L-1 <= length(encoded) && strcmp(encoded(idx:idx+L-1), codebook(i).code)
                % Pokud odpovídá, přidej symbol do výsledku
                decoded(end+1) = codebook(i).symbol;
                idx = idx + L; % Pokračuj za tento úsek
                found = true;
                break;
            end
        end
    end
end