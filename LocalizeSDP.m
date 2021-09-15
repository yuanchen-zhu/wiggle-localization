function [err, v, meanEdges, meterPerUnit] = LocalizeSDP(filename, dumpToFn)
% Load in the graph represetend as filename and localize it using the ESDPD
% routine
f = fopen(filename, 'rb');
tmp = fscanf(f, '%d%d%d%f%f', 5);
v = tmp(1)
d = tmp(2);
e = tmp(3);
meanEdges = tmp(4)
meterPerUnit = tmp(5)

edges = reshape(fscanf(f,'%d%d%f', 3*e), 3, e);

pp = reshape(fscanf(f,'%f', d * v), d, v);
fclose(f);

[~, anchor0] = min(pp(1,:));
distFromAnchors = zeros(1,v);
for i=1:v,
    d1 = pp(:,anchor0)-pp(:,i);
    distFromAnchors(i) = dot(d1,d1);
end
[~, anchor1] = max(distFromAnchors);
distFromAnchors(1,:) = 0;
for i=1:v,
    if i~=anchor0 && i~=anchor1,
        d1 = pp(:,anchor0)-pp(:,i);
        d2 = pp(:,anchor1)-pp(:,i);
        distFromAnchors(i) = sqrt(dot(d1,d1))+sqrt(dot(d2,d2));
    end
end
[~, anchor2] = max(distFromAnchors);

anchors = sort([anchor0, anchor1, anchor2]);
cur2orig = 1:v;
cur2orig(anchors)=[];
cur2orig = [cur2orig, anchors];

orig2cur = 1:v;
orig2cur(cur2orig) = 1:v;

pp = pp(:,cur2orig);

edges(1,:) = edges(1,:)+1;
edges(2,:) = edges(2,:)+1;

for i=1:size(edges,2),
    edges(1:2,i)=sort(orig2cur(edges(1:2,i)));
end

dist = sparse(edges(1,:), edges(2,:), edges(3,:), v, v);

loc = FULLSDPD(pp, 3, dist);

loc = [loc, pp(:,v-2:v)];


R = OptimalRigid(loc, pp);
qq = R * [loc; ones(1,v)];

errBeforeR = norm(pp - loc, 'fro')/sqrt(v) * meterPerUnit
err = norm(pp - qq(1:2,:), 'fro')/sqrt(v) * meterPerUnit

if (dumpToFn ~= 0)
    r = qq(1:2,orig2cur);
    f = fopen([filename, '-sdpout'], 'wb');
    fprintf(f, '%d %d %d\n', anchors - [1 1 1]);
    fprintf(f, '%g %g\n', r);
    fclose(f);
end
    

