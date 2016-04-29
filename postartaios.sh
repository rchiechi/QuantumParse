#!/bin/bash

if [ ! -n "$1" ]; then
	printf "I need a filename for the output movie.\n"
	exit
fi

printf "Runing sed..\n"

sed -i 's/ \/src\/tools\///' *.pov
sed -i 's/location < 0.0, 0.0, 10.0 >/location < 0.0, 20.0, 10.0 >/g' *.pov
cp /usr/local/lib/artaios_beta_020914/src/tools/*.ttf ./
gnuplot *.gpin

printf "Generating povray images...\n"

printf '%s\0' flux*.pov | xargs -0 -P4 -n1 povray +A0.3 +W1600 +H1200 +Q11 display=false
printf '%s\0' flux*.png | xargs -0 -P4 -n1 -t -I {} convert -trim {} {}

printf "Creating movie..\n"

mencoder mf://*.png -mf w=800:h=600:fps=1:type=png -ovc lavc -o ${1}.avi
