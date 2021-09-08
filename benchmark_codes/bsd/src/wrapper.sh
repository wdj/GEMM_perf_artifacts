#!/bin/bash

#module load cuda

nodeid=$(hostname | sed -e 's/.*h41n//')

#num_trials=2
num_trials=6

percent=100
randseed=1

# warmup
#line=$(env OMP_NUM_THREADS=7 ./kLine 16384 16384 1 2>/dev/null | cut -f8,10 -d' ')
line=$(env OMP_NUM_THREADS=7 ./kLine 10 10 1 2>/dev/null | cut -f8,10 -d' ')

gpuid=$(hostname)-$(echo "$line" | cut -f2 -d' ')-$(echo "$line" | cut -f1 -d' ')

echo $gpuid

im1=714025
ia1=4096
ic1=150889

im2=233280
ia2=9301
ic2=49297

rand1=$(( $RANDOM + 32768 * $RANDOM ))
rand2=$(( $RANDOM + 32768 * $RANDOM ))
#rand1=21231
#rand2=1904
echo "Initial seeds: $rand1 $rand2"

outfile=outs/out-${gpuid}-${rand1}-${rand2}.txt

count=0

while [ 1 = 1 ] ; do

  kmax=28215744
  mmax=38960

  rand1=$(perl -e "print int( ( $rand1 * $ia1 + $ic1 ) % $im1 ) ;")
  rk=$rand1
  rkmax=$im1
  rand2=$(perl -e "print int( ( $rand2 * $ia2 + $ic2 ) % $im2 ) ;")
  rm=$rand2
  rmmax=$im2
  k=$(perl -e "print int( ( $rk * 1. * $kmax ) / $rkmax ) ;")
  m=$(perl -e "print int( ( $rm * 1. * $mmax ) / $rmmax ) ;")

  k=$(( ( ( $k + 8 ) / 8 ) * 8 ))
  m=$(( ( ( $m + 8 ) / 8 ) * 8 ))

  kbase=$k
  mbase=$m

  for offset in $(seq 0 $(( ( 1 << ( ( $count / 2 ) % 6 ) - 1 ) )) ) ; do

    if [ $(( $count % 2 )) = 0 ] ; then
      k=$(( $kbase + 8 * offset ))
    else
      m=$(( $mbase + 8 * offset ))
    fi

    mem=$(( 2 * 2 * $k * $m + 1 * 4 * $m * $m ))
    if [ $mem -gt $(( 1024 * 1024 * 1024 * 40 * 19 / 20 )) ] ; then
      continue
    fi

    load=$(nvidia-smi |grep -A100 Processes | grep '^|    [0-7]' | grep -v kLine | grep -v "C   -" | wc -l)

    date | tee -a $outfile

    echo "$k $m  $mem $(( $mem / 1000000 ))" | tee -a $outfile

    echo ./run $m $k $num_trials $randseed $percent | tee -a $outfile

#    env OMP_NUM_THREADS=7 ./run $m $k $num_trials $randseed $percent | tee -a $outfile
    env OMP_NUM_THREADS=1 ./run $m $k $num_trials $randseed $percent | sed -e 's/$'"/ mem $(( $mem / ( 1024 * 1024 * 1024 ) )) otherprocs $load/" | tee -a $outfile

  done

  count=$(( $count + 1 ))

done

