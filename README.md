# QuantumParse

This is just a small Python script for converting between file formats for quantum chemistry software. 
It is customized for our lab and I am not a programmer, but it seems to work ok.

## Requirements
- python 3.5+
- colorama
- ase
- matplotlib
- Artaios (for artaios output)

## Usage

To read a linear molecule projected along the x, y, or z axis from <FILE> in XYZ, Gaussian or siesta format, attach
two Au electrodes and write an output file suitable for a trasport calculation in transiesta:
```
QuantumParse --transport --build Au -o siesta <FILE>
```

To prepare a Gaussian or Orca calculation for further transport calculatoins in Artaios:
```
QuantumParsae --transport -o {gaussian,orca} <FILE>
```

To parse the output of a Gaussian or Orca calculation for Artaios:
```
QuantumParse -o artaios <FILE>
```

To convert between file formats:
```
QuantumParse -o {orca,gaussian,siesta,xyz} <FILE>
```

# ArtaiosParallel

This is a Python script that runs an Artaios job in parallel across multiple threads of execution. It is very basic; make sure the input file is named 'transport.in' and execute ArtaoisParallel.py from the directory containing the Artaios job. It will create a directory called artaios_parallel and a BASH script called artaios_parallel.sh that runs the job and concatenates the output into the parent directory. Note that the datapoints in the transmission output file will be out of order.
