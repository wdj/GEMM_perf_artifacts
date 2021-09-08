#!/bin/bash

nodeid=$(hostname | sed -e 's/.*h41n//')

num_trials=2

percent=100
randseed=1

# warmup
line=$(env OMP_NUM_THREADS=7 ./run 100 100 1 2>/dev/null | cut -f8,10 -d' ')

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

mkdir -p outs
outfile=outs/out-${gpuid}-${rand1}-${rand2}.txt

count=0
cycle=0

while [ 1 = 1 ] ; do

  cycle=$(( $cycle + 1 ))

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

  round=1
  k=$(( ( ( $k + $round ) / $round ) * $round ))
  m=$(( ( ( $m + $round ) / $round ) * $round ))

  kbase=$k
  mbase=$m

  for offset in $(seq 0 $(( ( 1 << ( ( $count / 2 ) % 6 ) - 1 ) )) ) ; do

    if [ $(( $count % 2 )) = 0 ] ; then
      k=$(( $kbase + $round * offset ))
    else
      m=$(( $mbase + $round * offset ))
    fi

    mem=$(( 2 * 2 * $k * $m + 1 * 4 * $m * $m ))
    #if [ $mem -gt $(( 1024 * 1024 * 1024 * 40 * 9 / 10 )) ] ; then
    if [ $mem -gt $(( 1024 * 1024 * 1024 * 32 * 10 / 10 )) ] ; then
      continue
    fi

    echo "$m $k  $mem $(( $mem / 1000000 ))     $cycle" | tee -a $outfile

    echo ./run $m $k $num_trials $randseed $percent

    env OMP_NUM_THREADS=4 ./run $m $k $num_trials $randseed $percent 2>&1 | tee -a $outfile

  done

  count=$(( $count + 1 ))

done

