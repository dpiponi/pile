cp $1 0.png                               

convert -flop 0.png 1.png
convert -flop -flip 0.png 2.png
convert -flip 0.png 3.png

magick 1.png 0.png  +append 10.png
magick 2.png 3.png  +append 23.png

magick 23.png 10.png -append 2310.png

cp 2310.png $2
