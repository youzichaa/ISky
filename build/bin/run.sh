#!/bin/bash

for i in {0..3}
do
   ./isky-OT-B r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0 & ./isky-OT-B r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0
    sleep 1
   ./isky-OT-BT r=1 p=12000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 & ./isky-OT-BT r=2 p=12000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1
    sleep 1
   ./isky-OT-F r=1 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0 & ./isky-OT-F r=2 p=11000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=0
    sleep 1
   ./isky-OT-FT r=1 p=12000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1 & ./isky-OT-FT r=2 p=12000 Pdim=1 Pname=$i Psize=0 Prate=1 Pthread=1
    sleep 1
done


