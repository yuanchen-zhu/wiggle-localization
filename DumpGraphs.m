function list = DumpGraphs(listFilename)
% Dump the list of files in listFilename
f = fopen(listFilename, 'rb');
c = textscan(f, '%s');
fclose(f);
list = c{1,1};

for i =1:size(list,1),
    list{i,1}=['cache/', list{i,1}];
end