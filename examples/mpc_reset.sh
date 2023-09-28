#!/bin/sh

# $1 : folder
# $2 : n_awes

# reset initial state 
orig=$1"x001_init.csv"
name=$1"x$(printf "%03d" $2).csv"
cp $orig $name

# reset MPC initial guess
orig=$1"w001_init.csv"
name=$1"w$(printf "%03d" $2).csv"
cp $orig $name

# Initialize control file
orig=$1"u001_init.csv"
name=$1"u$(printf "%03d" $2).csv"
cp $orig $name

# reset MPC log file
orig=$1"log001_init.csv"
name=$1"log$(printf "%03d" $2).csv"
cp $orig $name

# reset MPC output file
orig=$1"log001_init.csv"
name=$1"out$(printf "%03d" $2).csv"
cp $orig $name

# reset xa output file
orig=$1"xa001_init.csv"
name=$1"xa$(printf "%03d" $2).csv"
cp $orig $name

# reset xddot output file
orig=$1"dx001_init.csv"
name=$1"dx$(printf "%03d" $2).csv"
cp $orig $name

echo "mpc reset done..."
