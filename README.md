## QuantumParse

This is just a small Python script for converting between file formats for quantum chemistry software. 
It is customized for our lab and I am not a programmer, but it seems to work ok.

# Requirements
- python 3.5+
- colorama
- ase
- matplotlib

#Usage

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
