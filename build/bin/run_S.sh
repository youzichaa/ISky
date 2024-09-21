#!/bin/bash

for i in {0..3}
do
    ./isky-OT-B-ALL r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=0 & ./isky-OT-B-ALL r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=0
    sleep 1
    for r in {2..4}
    do
        ./isky-OT-B-ALL r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=0 & ./isky-OT-B-ALL r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=0
        sleep 1
    done
    ./isky-OT-B-ALL r=1 p=11000 Pdim=0 Pname=$i Psize=0 Prate=1 Pthread=0 & ./isky-OT-B-ALL r=2 p=11000 Pdim=0 Pname=$i Psize=0 Prate=1 Pthread=0
    sleep 1
    for k in {2..4}
    do
       ./isky-OT-B-ALL r=1 p=11000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=0 & ./isky-OT-B-ALL r=2 p=11000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=0
        sleep 1
    done
done
for i in {0..3}
do
    for j in {0..5}
    do
        ./isky-OT-B-ALL r=1 p=11000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=0 & ./isky-OT-B-ALL r=2 p=11000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=0
        sleep 1
    done
done

