function draw_confusion_matrix(cm_mat, TickLabels)

if nargin < 2,
  TickLabels = [];
end

assert(size(cm_mat, 1) == size(cm_mat, 2));

N = size(cm_mat, 1);

% cm_mat = rand(N);           %# A 5-by-5 matrix of random values from 0 to 1
% cm_mat(3,3) = 0;            %# To illustrate
% cm_mat(5,2) = 0;            %# To illustrate
imagesc(cm_mat);            %# Create a colored plot of the matrix values
axis image
% colorbar
%colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
colormap('jet');                         

textStrings = num2str(cm_mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding


%% ## New code: ###
idx = find(strcmp(textStrings(:), '0.00'));
textStrings(idx) = {'   '};


%% ################

[x,y] = meshgrid(1:N);   %# Create x and y coordinates for the strings
% hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
%                 'HorizontalAlignment','center');
% midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
% textColors = repmat(cm_mat(:) > midValue,1,3);  %# Choose white or black for the
%                                              %#   text color of the strings so
%                                              %#   they can be easily seen over
%                                              %#   the background color
% set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca, 'XAxisLocation','top',...
         'TickLength',[0 0]);
       
 
XLabels = cellfun(@(x) ['A',num2str(x)], num2cell([1:N]), 'UniformOutput', false);
YLabels = cellfun(@(x,y) [x,'(',y,')'], TickLabels, XLabels', 'UniformOutput', false);
 

if ~isempty(TickLabels)
  set(gca, ...
        'XTickLabel', '', ...
        'YTick', 1:N, ...
        'YTickLabel', TickLabels, ...
        'FontSize', 14);
%          'XTick', 1:N, ...
%       rotateXLabels(gca, 45);

end

      
      
      