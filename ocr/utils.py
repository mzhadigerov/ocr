import logging as log
import os
import re
from difflib import SequenceMatcher

import cv2
import nltk
import pytesseract as tess
import torch
import tqdm
from enchant.checker import SpellChecker
from pytesseract import Output
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer

DATA_FOLDER = './data/'
en_spch = SpellChecker("en_US")


def check_verbose(verbose):
    """On/Off verbose log mode"""
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose mode: ON.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")


def get_text_from_image(path):
    """Retrieve text data from image"""
    log.info("Processing an image...")
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(
        img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255, 255)
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = pre_process(img)
    log.info("Done pre-proccessing...")
    text = tess.image_to_string(img, lang='eng', config='--psm 6 --oem 1')
    log.info("Done converting image to text...")

    result_text = post_process(text)
    log.info("Done post-processing...")

    log.info("Done processing")
    return text


def write_text_to_file(text, file, out_mode='w'):
    """Writes given text to file"""
    if out_mode == 'w':
        log.info("Start writing text to file")
    elif out_mode == 'a':
        log.info("Appending text to file")
    else:
        log.error('Invalid out_mode. Accepted modes: "w", "a"')
        raise Exception('Invalid out_mode. Accepted modes: "w", "a"')
    with open(file, out_mode, encoding='utf-8') as f:
        f.write(text)
    log.info("Done writing text to file")


def format_text(text):
    """Perform data formatting using regex"""
    rep = {
        '\n': ' ',
        '\\': ' ',
        '\"': '"',
        '-': ' ',
        '"': ' " ',
        '"': ' " ',
        '"': ' " ',
        ',': ' , ',
        '.': ' . ',
        '!': ' ! ',
        '?': ' ? ',
        "n't": " not",
        "'ll": " will",
        '*': ' * ',
        '(': ' ( ',
        ')': ' ) ',
        "s'": "s '",
        ";": " ; ",
        "‘": " ‘ ",
        "’": " ’ "
    }
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def get_personslist(text):
    """Get list of named entities (person names)"""
    personslist = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0, (chunk.leaves()[0][0]))
    return list(set(personslist))


def predict_words(
    text_original, predictions, maskids, tokenizer, suggestedwords
):
    """Substitute masked words with most probable alternatives"""
    pred_words = []
    for i in range(len(maskids)):
        preds = torch.topk(predictions[0, maskids[i]], k=50)

        indices = preds.indices.tolist()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        list2 = suggestedwords[i]
        simmax = 0
        predicted_token = ''
        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, word1, word2).ratio()
                if s is not None and s > simmax:
                    simmax = s
                    predicted_token = word1
        pred_words.append(predicted_token)
        text_original = text_original.replace('[MASK]', predicted_token, 1)
    return text_original


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage,
        M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return newImage


def show_char_image_boxes(img):
    """Show word level bounding boxes in an image"""
    imcopy = img.copy()
    d = tess.image_to_data(imcopy, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if (d['text'][i] != ""):
            (x, y, w,
             h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(imcopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('result.png', imcopy)
    cv2.waitKey(0)


def pre_process(img):
    """Apply preprocessing to an image"""
    angle = getSkewAngle(img)
    img = rotateImage(img, -1.0 * angle)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    show_char_image_boxes(thresh)
    return thresh


def post_process(text):
    """Apply postprocessing to text"""
    text = text.replace("-\n", "")
    #TODO: apply truncation or other method to deal with > 511 token text size
    if len(text) > 511 or len(text) < 2:
        log.warning("Text size is 5< or > 511 tokens. Return text as is")
        return text
    text_original = str(text)
    text = format_text(text)
    personslist = get_personslist(text)
    ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
    words = text.split()
    incorrectwords = [
        w for w in words if not en_spch.check(w) and w not in ignorewords
    ]
    # using enchant.checker.SpellChecker, get suggested replacements
    suggestedwords = [en_spch.suggest(w) for w in incorrectwords]
    # replace incorrect words with [MASK]
    for w in incorrectwords:
        text = text.replace(' ' + w + ' ', ' [MASK] ')
        text_original = text_original.replace(' ' + w + ' ', ' [MASK] ')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']

    # prepare Torch inputs
    tokens_tensor = torch.tensor([indexed_tokens])
    # Load pre-trained model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor)
    return predict_words(
        text_original, predictions, MASKIDS, tokenizer, suggestedwords
    )


def load_images_from_folder(folder):
    """Load images from folder"""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_txt_from_folder(folder):
    """Load txt files from folder"""
    txts = []
    for file in os.listdir(folder):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{folder}\{file}"
            with open(file_path, 'r') as f:
                txts.append(f.read())
    return txts


#TODO: try other metrics
def accuracy(ocr_text, true_text):
    """Test accuracy of two texts using sequence matching"""
    ocr_text = "".join(ocr_text.split("\n"))
    true_text = "".join(true_text.split("\n"))
    sequence = SequenceMatcher(None, true_text, ocr_text)
    return sequence.ratio() * 100
