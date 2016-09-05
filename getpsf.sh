#!/bin/bash

ATOM=$1

if [ ! -n "$ATOM" ]; then
    echo "No atom?"
    exit
fi

curl http://departments.icmab.es/leem/siesta/Databases/Pseudopotentials/Pseudos_GGA_Abinit/${ATOM}_html/${ATOM}.psf > ${ATOM}.psf
