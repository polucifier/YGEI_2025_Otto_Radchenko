%-------------------------------------------------------------
% Manual K-means Clustering Demonstration
%-------------------------------------------------------------
clear; clc; close all; format long g;

% % ---------------------- DATA GENERATION -----------------------
cType = 5;          % 1 - One large Cluster, 2 - Two medium Clusters,
M = genPts(cType);  % 3 - Three smaller Clusters, 4 - Spiral, 5 - Ring

% Initial plot for selecting centers
figure; hold on;
plot(M(:,1), M(:,2), 'r*');
title('Select Initial Cluster Centers, and press ENTER');
xlabel('X'); ylabel('Y');

% % ---------------------- USER INPUT ----------------------------
[xi, yi] = getpts;
C = [xi, yi];       % initial centers
k = size(C,1);

% % ---------------------- PARAMETERS ----------------------------
maxIter = 20;
tol = 0.01;
iter = 0;
moveFlag = true;

% % ---------------------- K-MEANS LOOP --------------------------
while moveFlag && iter < maxIter
    
    % Distance matrix: N x k
    distM = pdist2(M, C);
    
    % Assign clusters
    [~, lbl] = min(distM, [], 2);
    
    % Begin figure
    figure; hold on;
    h = gscatter(M(:,1), M(:,2), lbl);
    
    % Prepare for new centers
    nCenter = zeros(k,2);
    moveFlag = false;
    cnt = zeros(k,1);
    
    for j = 1:k
        pts = M(lbl == j, :);
        cnt(j) = size(pts,1);
        nCenter(j,:) = mean(pts,1);
        
        % Plot cluster centroid
        plot(nCenter(j,1), nCenter(j,2), 'kx', ...
            'MarkerSize', 12, 'LineWidth', 2);
    end
    
    % Add legend
    lg = legend;
    title(sprintf('Iteration %d', iter+1));
    
    % ------------------- CLUSTER COUNT BOX -----------------------
    txt = sprintf('Cluster sizes:\n');
    for j = 1:k
        txt = sprintf('%s  Cluster %d: %d pts\n', txt, j, cnt(j));
    end
    
    lgPos = lg.Position;
    
    annotation('textbox', ...
        [lgPos(1), lgPos(2)-0.12, lgPos(3), 0.1], ...
        'String', txt, ...
        'FitBoxToText', 'on', ...
        'BackgroundColor', [0.95 0.95 0.95], ...
        'EdgeColor', [0.4 0.4 0.4], ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'left');
    % --------------------------------------------------------------
    
    hold off;
    
    % Check centroid movement
    for j = 1:k
        if norm(C(j,:) - nCenter(j,:)) > tol
            moveFlag = true;
        end
    end
    
    C = nCenter;
    iter = iter + 1;
end

fprintf('K-means finished after %d iterations.\n', iter);


%-------------------------------------------------------------
%       POINT GENERATOR FUNCTION
%-------------------------------------------------------------
function M = genPts(t)

    switch t
        
        case 1  % One cluster
            M = randn(150,2);
            
        case 2  % Two clusters
            A = randn(50,2);
            B = randn(50,2)*1.2 + [10, 2];
            M = [A; B];
            
        case 3  % Three clusters
            A = randn(15,2);
            B = randn(35,2)*1.2 + [10, 2];
            C = randn(40,2)*1.2 + [10,10];
            M = [A; B; C];
        
        case 4  % Spiral cluster
            t = linspace(0,4*pi,300)';
            spiral = [t.*cos(t) + 0.5*randn(size(t)), ...
                      t.*sin(t) + 0.5*randn(size(t))];
            M = [spiral];

        case 5 % Ring cluster
            ang = 2*pi*rand(200,1);
            r = 3 + 0.5*randn(200,1);
            ring = [r.*cos(ang)+5, r.*sin(ang)+10];
            M = [ring];
            
        otherwise
 
            return

    end
end
