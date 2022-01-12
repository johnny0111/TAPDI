function HoughCircleDetection_Farhat
%% Complete Algorithm Steps
% 1. Reading an Image
% 2. Convert to GrayScale
% 3. Detecting Edges
% 4. Defining an Accumulator Matrix
% 5. Finding the Circles Centers with Equation of Circle Using value of Radius
% 6. Assigning Values in the Accumulator Matrix
% 7. Finding Peak Values (Cen of Circles)
% 8. Plotting Circles on Original Colour Image
%%
close all
clear all
clc
% MinR = 27; MaxR=28; %Value of Rmin and Rmax
MinR = 50; MaxR=70; 
%% Step 1-3
% image = imread('cars3.jpg');
image = imread('coins.png');
figure, imshow(image);
grayimage = rgb2gray(image);
figure, imshow(grayimage);
% bwimage = im2bw(image, 0.4); %Almost similar results in Edge Image of gray image
% figure, imshow(bwimage);
% Using Canny Edge Detector. Set the Hyteresis Thresholding Lvl on Hit and
% Trial Method. [] Empty array gives erroneous results 
% edgeimage = edge(grayimage, 'Canny', [0.3, 0.35]); 
edgeimage = edge(grayimage, 'Canny', [0.1, 0.2]); 
figure, imshow(edgeimage);
[row, colm] = size(edgeimage);

%% Step - 4 (Accumulator) (xi, yi, ri)
% Accumulator with Zeroes, rows and colm extended by 2 times MaxR (for
% border circles which are out of range of image)
HoughSpace = zeros(row+2*MaxR, colm+2*MaxR, MaxR - MinR + 1);
HoughSpaceSize = size(HoughSpace);
%% Step-5 (Locating All Possible Circles Centers in Image Region)
[E_y, E_x] = find(edgeimage); % All nonzero edge Pixels
[X, Y] = meshgrid(0:MaxR*2, 0:MaxR*2); % creation of grid for storing the values of circle centers coordinates
Radii = round(sqrt((X - MaxR).^2 + (Y - MaxR).^2)); % Calc all the possible radii with eqn of circle
Radii(Radii<MinR | Radii>MaxR) = 0; %discard out of range radii
[CC_y,  , R] = find(Radii);% All nonzero radii for centers
%% Step - 6 Filling up Accumulator HoughSpace with all Possible Circle Centers satisfying the Eqn
for i = 1 : length(E_x) 
    % Linear Indices (array size, y-dimension, x-dimension, radius
    % dimension) cc_votes(Circle Center)
  cc_votes = sub2ind(HoughSpaceSize, CC_y+E_y(i) - 1, CC_x+E_x(i) - 1, R-MinR + 1);
  HoughSpace(cc_votes) = HoughSpace(cc_votes)+1; 
end
%% Step-7 Finding Peak Value of Circles to be Plotted on Original Image
DetectedCircles = zeros(0,4);    
for radius = MinR : MaxR % Loop from minimal to maximal radius
  PeakCircles = HoughSpace(:,:,radius-MinR + 1);  % Loop for radius greater than MinR
  twoPiR = 2* pi * radius * 0.333; % 0.33 is parameter value for threshold to minimize the candidate circles
  PeakCircles(PeakCircles<twoPiR) = 0;  % discarding smaller circles
  [y, x, radii_all] = find(PeakCircles); % all non-zero enteries for radii
  DetectedCircles = [DetectedCircles; [(x - MaxR), (y - MaxR), (radius*ones(length(x),1)), (radii_all/twoPiR)]]; %storing values of circle centers with radii
end

%% Step-8 Draw circles on Original Image
% Reference to draw circles is taken from ARTICLE: How do I plot a circle with a given radius and center?
% https://www.mathworks.com/matlabcentral/answers/98665-how-do-i-plot-a-circle-with-a-given-radius-and-center
figure, imshow(image), hold on;
for i=1:size(DetectedCircles,1)
    x = DetectedCircles(i,1)-DetectedCircles(i,3);
    y = DetectedCircles(i,2)-DetectedCircles(i,3);
    rr = 2*DetectedCircles(i,3);
    rectangle('Position', [x y rr rr], 'EdgeColor', 'red', 'Curvature', [1,1]);
    %drawcircle('Center', [x y], 'Radius', rr, 'EdgeColor', 'r');
end
  hold off;  
end

