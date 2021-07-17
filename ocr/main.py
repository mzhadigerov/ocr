import os

import click
import cv2
import pytesseract as tess
import utils
from pdf2image import convert_from_path
from utils import (
    check_verbose, get_text_from_image, post_process, pre_process,
    write_text_to_file
)

tess.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  #substitute with local tesseract.exe path


@click.group()
def cli():
    pass


@click.command()
@click.option('--input', help='Path to an input image')
@click.option('--output', help='Path to an output txt file')
@click.option('--verbose', is_flag=True, help='Verbose mode')
def image2text(input, output, verbose):
    """Retrieve text data from image `input` and store in `output` text file"""
    check_verbose(verbose)

    if input.endswith(('jpg', 'jpeg', 'png')):
        text = get_text_from_image(input)
        write_text_to_file(text, output, 'w')
    elif input.endswith('pdf'):
        pages = convert_from_path(input, 500)
        image_counter = 1
        file_names = []
        for page in pages:
            filename = "page_" + str(image_counter) + ".jpg"
            page.save(filename, "JPEG")
            file_names.append(filename)
            image_counter += 1
        filelimit = image_counter - 1
        for i in range(1, filelimit + 1):
            filename = "page_" + str(i) + ".jpg"
            text = get_text_from_image(filename)
            out_mode = 'w'
            if i > 1:
                out_mode = 'a'
            write_text_to_file(text, output, out_mode)
        for filename in file_names:
            if os.path.exists(filename):
                os.remove(filename)
    else:
        log.error('Wrong input format')


cli.add_command(image2text)

if __name__ == '__main__':
    cli()
