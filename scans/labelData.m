
for num =000:168
    numString = num2str(num,'%03.f');
    fileBase = strcat("hokuyoScan",numString);
    filename = strcat(fileBase, ".pcd");
    delimiterIn = ' ';
    headerlinesIn = 11;
    rawdata = importdata(filename,delimiterIn,headerlinesIn);
    points = rawdata.data;
    rtipoints = zeros(0,4);
    for i=1:1:size(points)
        r = sqrt(points(i,1).^2 + points(i,2).^2);
        theta = atan (points(i,1).^2 / points(i,2).^2);
        intensity = points(i,4);
        rtipoints = [rtipoints;[r, theta, intensity, 0]];
    end
    
    hold on
    test = scatter(points(:,1), points(:,2))
    
    %brush on
    disp('Hit Enter in comand window when done brushing')
    pause
    
    data = test.BrushData;
    
    if size(data)~=0
        rtipoints(:,4) = transpose(data);
    end
    
    indicies = find(data);
    clf
    
    outBase = strcat("labeledData",numString);
    outName = strcat(outBase, ".txt");
    
    dlmwrite(outName,rtipoints)
end
