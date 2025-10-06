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

%store components
Y_old = Y;
Cb_old = Cb;
Cr_old = Cr;

%division to submatrices
[m,n] = size(Y_old);

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

        %Quantization - 1. ztratovy krok
        Ys_q = round(Ys_dct ./ Qyf);
        Cbs_q = round(Cbs_dct ./ Qcf);
        Crs_q = round(Crs_dct ./ Qcf);

        %Update transformed matrix
        Y(i:i+7,j:j+7) = Ys_q;
        Cb(i:i+7,j:j+7) = Cbs_q;
        Cr(i:i+7,j:j+7) = Crs_q;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%JPEG decompression
%%%%%%%%%%%%%%%%%%%%%%%%%

%Process lines
for i = 1:8:m-7
    %Process columns
    for j = 1:8:n-7
        %Get submatrices
        Ys = Y(i:i+7, j:j+7);
        Cbs = Cb(i:i+7, j:j+7);
        Crs = Cr(i:i+7, j:j+7);

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
imshow(imgj);

%Standard deviations for RGB components
dR = Rd - R;
dG = Gd - G;
dB = Bd - B;

R2 = dR.^2;
G2 = dG.^2;
B2 = dB.^2;

sigmaR = sqrt(sum(R2(:))/(m*n))
sigmaG = sqrt(sum(G2(:))/(m*n))
sigmaB = sqrt(sum(B2(:))/(m*n))

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