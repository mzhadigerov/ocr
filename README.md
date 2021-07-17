# Command line tool for OCR

1. Download Tesseract from [https://github.com/UB-Mannheim/tesseract/wiki](here)
2. Substitute `tess.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"` in `main.py` with `path\to\your\tesseract.exe` in your local machine.
3. Download all the libraries listed in `toml` file.
4. Run `preparation.sh`. Make sure you have `nltk` downloaded.


## Test:

`python .\main.py image2text --input=test_input.jpg --output=test_output.txt`

The program accepts `pdf`, `jpg` and `png` formats.

