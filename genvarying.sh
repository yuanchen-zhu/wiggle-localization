#!/bin/sh

for p in mu N delta D; do
    sed -e 's/1:2/1:2/g' varying-$p.gp > varying-$p-p.gp
    sed -e 's/1:2/1:3/g' varying-$p.gp > varying-$p-d.gp
done

gnuplot parameters.gp
epsmp varying.mp
