!/bin/bash

#for r in {0..4}
#do
#    for k in {0..4}
#    do
#        for i in {0..2}
#        do
#            for j in {0..5}
#            do
#                ./isky-OT-F r=1 p=10000 Pdim=1 Pname=$i Psize=0 Prate=$r & ./isky-OT-F r=2 p=10000 Pdim=1 Pname=$i Psize=0 Prate=$r
#                sleep 1
 #           done
#        done
 #   done
#done
for k in {0..4}
do
    for i in {0..3}
    do
        ./isky-OT-FT r=1 p=12000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=1 & ./isky-OT-FT r=2 p=12000 Pdim=$k Pname=$i Psize=0 Prate=1 Pthread=1
        sleep 1
    done
done
for i in {0..3}
do
    ./isky-OT-FT r=1 p=12000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=1 & ./isky-OT-FT r=2 p=12000 Pdim=1 Pname=$i Psize=0 Prate=0 Pthread=1
    sleep 1
done
for r in {2..4}
do
    for i in {0..3}
    do
        ./isky-OT-FT r=1 p=12000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=1 & ./isky-OT-FT r=2 p=12000 Pdim=1 Pname=$i Psize=0 Prate=$r Pthread=1
        sleep 1
    done
done
for j in {1..5}
do
    for
    do i in {0..3}
        /isky-OT-FT r=1 p=12000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=1 & /isky-OT-FT r=2 p=12000 Pdim=1 Pname=$i Psize=$j Prate=1 Pthread=1
        sleep 1
    done
done
