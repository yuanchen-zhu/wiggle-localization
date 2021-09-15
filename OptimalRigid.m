function R = OptimalRigid(p, q)
% Get the optimal rigid transformation from p to q

pStar = mean(p,2);
qStar = mean(q,2);

n = size(p, 2);
pp = p - repmat(pStar, 1, n);
qq = q - repmat(qStar, 1, n);
[U,S,V] = svd(qq * pp');

M = eye(3);
M(1:2,1:2) = U * V;

pT = eye(3);
pT(1:2,3) = -pStar;

qT = eye(3);
qT(1:2,3) = qStar;

R = qT * M * pT;


