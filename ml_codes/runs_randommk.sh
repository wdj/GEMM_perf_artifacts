#!/bin/bash
#SBATCH -A stf006
#SBATCH -N 1
#SBATCH -t 48:00:00

# NOTE: these runs are done on andes.olcf.ornl.gov

# (cd logs; env method=dtree  sbatch -J dtree2  ../runs_randommk.sh)
# (cd logs; env method=etree  sbatch -J etree2  ../runs_randommk.sh)
# (cd logs; env method=gboost sbatch -J gboost2 ../runs_randommk.sh)
# (cd logs; env method=ranfor sbatch -J ranfor2 ../runs_randommk.sh)

cd /gpfs/alpine/gen006/proj-shared/$USER/andes
outdir=outs_randommk
mkdir -p $outdir

module load python

if [ "$method" = dtree ] ; then
  stub=runs12-peak2M   j=4; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs10-bsd        ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-bsd        ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-spock4.2   ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-iris       ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  echo >/dev/null
fi

if [ "$method" = etree ] ; then
  stub=runs12-peak2M    j=1; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs10-bsd         ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-bsd         ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-spock4.2 j=4; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-iris        ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  echo >/dev/null
fi

if [ "$method" = gboost ] ; then
  stub=runs12-peak2M   j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs10-bsd      j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-bsd      j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-spock4.2 j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-iris     j=8; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  echo >/dev/null
fi

if [ "$method" = ranfor ] ; then
  stub=runs12-peak2M   j=1; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs10-bsd        ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-bsd        ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-spock4.2 j=1; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  stub=runs12-iris       ; ( time srun -N1 -n1 -c32 ./train ${stub}.csv $method "$j" $outdir) 2>&1 | tee $outdir/out_${stub}_${method}.txt
  echo >/dev/null
fi


