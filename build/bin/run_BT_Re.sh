#!/bin/bash

for i in {0..3}
do
    ./isky-OT-B-Re r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=1 & ./isky-OT-B-Re r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=1
    sleep 1
    ./isky-OT-B-Re r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=5 & ./isky-OT-B-Re r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=5
    sleep 1
    ./isky-OT-B-Re r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=10 & ./isky-OT-B-Re r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 Ptheta=10
    sleep 1
done
for t in {1..3}
do
    for i in {0..3}
    do
        ./isky-OT-B-Re r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0 Pmu=$t & ./isky-OT-B-Re r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0 Pmu=$t
        sleep 1
    done
done
