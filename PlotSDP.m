function PlotSDP(listfn, fn)
% get the graphs
f = fopen(fn, 'wb');

list = DumpGraphs(listfn);
for i=1:size(list,1)
    [err, v, meanEdges, meterPerUnit] = LocalizeSDP(list{i,1}, 0)
    fprintf(f, '%g %d %g\n', meanEdges, v, err);
end
fclose(f);
