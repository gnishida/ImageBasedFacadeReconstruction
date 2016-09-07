'''
This script combines all the images in a directory to a single big image.
The number of rows, the height of each row, and the margin between images can be specified.


Usage: python combine_img.py <directory> --output <output file name> --rows <number of rows> --height <height of row> --margin <margin in pixels>
Example:
       python combine_img.py results --output combined.png --rows 3 --height 256 --margin 30


Updates:
9/7/2016 - To show numbers, add "--number" option

Author: Gen Nishida
Date: 9/1/2016
'''

import sys
from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join
import argparse

def main(directory_path, output_file, rows, height_of_row, margin, number):
	# read all the image files in the directory
	images = []
	widths = []
	for f in listdir(directory_path):
		if not isfile(join(directory_path, f)): continue
		
		try:
			image = Image.open(join(directory_path, f)).convert('RGBA')
			width, height = image.size
			
			new_width = width * height_of_row / height
			new_img = image.resize((new_width, height_of_row), resample = Image.LANCZOS)
			images.append(new_img)
			widths.append(new_width + margin)
		except:
			pass

	# compute the size of the combined image
	total_width = sum(widths) + margin
	image_width = int(total_width / rows + 0.5)
	
	# check the image width
	while True:
		x_offset = margin
		row = 0
		for i in xrange(len(images)):
			if x_offset + images[i].size[0] > image_width:
				x_offset = margin
				row = row + 1
			x_offset += images[i].size[0] + margin
		if row < rows: break
		image_width = image_width + int(images[0].size[0] * 0.5)
	
	text_height = int(height_of_row * 0.15)

	# create the combined image
	if number:
		combined_img = Image.new('RGBA', (image_width, height_of_row * rows + margin * (rows + 1) + text_height * rows), "white")
	else:
		combined_img = Image.new('RGBA', (image_width, height_of_row * rows + margin * (rows + 1)), "white")

	# set font size
	fontsize = 1
	font = ImageFont.truetype("arial.ttf", fontsize)
	while font.getsize("1")[0] < text_height * 0.5:
	    # iterate until the text size is just larger than the criteria
	    fontsize += 1
	    font = ImageFont.truetype("arial.ttf", fontsize)
	fontsize -= 1
	font = ImageFont.truetype("arial.ttf", fontsize)

	# combine images
	draw = ImageDraw.Draw(combined_img)
	x_offset = margin
	y_offset = margin
	if number:
		y_offset += text_height
	for i in xrange(len(images)):
		if x_offset + images[i].size[0] > image_width:
			x_offset = margin
			y_offset += height_of_row + margin
			if number:
				y_offset += text_height
		
		combined_img.paste(images[i], (x_offset, y_offset), images[i])
		if number:
			draw.text((x_offset, y_offset - text_height), str(i) + ")", font=font, fill=(0,0,0,224))
		x_offset += images[i].size[0] + margin

	combined_img.save(output_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("directory_path", help="directory path")
	# Optional arguments.
	parser.add_argument(
		"--output",
		default="output.png",
		help="output file name",
	)
	parser.add_argument(
		"--rows",
		default=1,
		type=int,
		help="number of rows",
	)
	parser.add_argument(
		"--height",
		default=256,
		type=int,
		help="height of each row in pixel",
	)
	parser.add_argument(
		"--margin",
		default=30,
		type=int,
		help="margin size in pixel",
	)
	parser.add_argument(
		"--number",
		action="store_true",
		help="true if the number is to be displayed",
	)
	parser.set_defaults(number=False)
	args = parser.parse_args()

	main(args.directory_path, args.output, args.rows, args.height, args.margin, args.number)
	