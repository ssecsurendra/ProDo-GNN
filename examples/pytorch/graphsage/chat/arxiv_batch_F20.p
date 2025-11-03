# Set the terminal and output file
#set terminal postscript eps enhanced color solid font 'Arial' size 14,6.0
#set terminal pdfcairo enhanced font 'Arial,12.5'
#set output 'arxiv_batch.pdf'
# Set terminal to EPS and output file name
set terminal postscript eps enhanced color solid font 'Helvetica,25' size 9.0,6.0
#set terminal postscript eps enhanced color font 'Helvetica,10'
set output 'arxiv_batch_F20.eps'
set boxwidth 0.9 absolute
set style fill solid 1.00 border lt -1
# Set the legend (key) font
set key inside top right vertical font "Arial,60"  # Adjust the font size as needed
# Set the labels (no title)
set tmargin 2
set xlabel "Batch size" font "Arial,55 italic bold" offset 0,-3
set bmargin 8
set ylabel "Total time (Seconds)" font "Arial,55 italic bold" offset -3
set lmargin 12

# Define bar style and width
set style data histograms
set style histogram cluster gap 1
set style fill solid 1.0 border -1
set boxwidth 0.9

# Set y-axis grid line
set grid ytics

# Set xtics (x-axis labels) from the first column (no rotation)
set xtics nomirror
set xtics font "Arial,60 italic bold" offset 0,-0.5
set ytics font "Arial,60 italic bold"
set xtics rotate by -45
# Remove extra space between Y-axis and first bar by setting x range
#set xrange [-0.9:3.5]  # Adjust this range to remove extra space on the left side
#set xrange [-0.6:2.7]  # Adjust this range to remove extra space on the left side

# Plot the data from a text file
plot 'arxiv_batch_20.txt' using 2:xtic(1) title 'DGL' lt rgb "dark-magenta", \
     '' using 3 title 'CentGNN' lt rgb "light-blue"
     #'' using 4 title 'F:20' lt rgb "forest-green"
# Close the output file
set output

