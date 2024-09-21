#!/bin/bash

for k in {0..4}
do
    for i in {0..3}
    do
        ./isky-OT-F r=1 p=11000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=1 & ./isky-OT-F r=2 p=11000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=1
        sleep 1
    done
done
for i in {0..3}
do
    ./isky-OT-F r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=1 & ./isky-OT-F r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=1
    sleep 1
done
for r in {2..4}
do
    for i in {0..3}
    do
        ./isky-OT-F r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=1 & ./isky-OT-F r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=1
        sleep 1
    done
done
for j in {1..5}
do
    for
    do i in {0..3}
        /isky-OT-F r=1 p=11000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=1 & /isky-OT-F r=2 p=11000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=1
        sleep 1
    done
done
