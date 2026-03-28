# import argparse
# import json
# from typing import Tuple, List

# import cv2
# import editdistance
# from path import Path

# from dataloader_iam import DataLoaderIAM, Batch
# from model import Model, DecoderType
# from preprocessor import Preprocessor


# class FilePaths:
#     """Filenames and paths to data."""
#     fn_char_list = '../model/charList.txt'
#     fn_summary = '../model/summary.json'
#     fn_corpus = '../data/corpus.txt'


# def get_img_height() -> int:
#     """Fixed height for NN."""
#     return 32


# def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
#     """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
#     if line_mode:
#         return 256, get_img_height()
#     return 128, get_img_height()


# def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
#     """Writes training summary file for NN."""
#     with open(FilePaths.fn_summary, 'w') as f:
#         json.dump({'averageTrainLoss': average_train_loss, 'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


# def char_list_from_file() -> List[str]:
#     with open(FilePaths.fn_char_list) as f:
#         return list(f.read())


# def train(model: Model,
#           loader: DataLoaderIAM,
#           line_mode: bool,
#           early_stopping: int = 25) -> None:
#     """Trains NN."""
#     epoch = 0  # number of training epochs since start
#     summary_char_error_rates = []
#     summary_word_accuracies = []

#     train_loss_in_epoch = []
#     average_train_loss = []

#     preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
#     best_char_error_rate = float('inf')  # best validation character error rate
#     no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
#     # stop training after this number of epochs without improvement
#     while True:
#         epoch += 1
#         print('Epoch:', epoch)

#         # train
#         print('Train NN')
#         loader.train_set()
#         while loader.has_next():
#             iter_info = loader.get_iterator_info()
#             batch = loader.get_next()
#             batch = preprocessor.process_batch(batch)
#             loss = model.train_batch(batch)
#             print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
#             train_loss_in_epoch.append(loss)

#         # validate
#         char_error_rate, word_accuracy = validate(model, loader, line_mode)

#         # write summary
#         summary_char_error_rates.append(char_error_rate)
#         summary_word_accuracies.append(word_accuracy)
#         average_train_loss.append((sum(train_loss_in_epoch)) / len(train_loss_in_epoch))
#         write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

#         # reset train loss list
#         train_loss_in_epoch = []

#         # if best validation accuracy so far, save model parameters
#         if char_error_rate < best_char_error_rate:
#             print('Character error rate improved, save model')
#             best_char_error_rate = char_error_rate
#             no_improvement_since = 0
#             model.save()
#         else:
#             print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
#             no_improvement_since += 1

#         # stop training if no more improvement in the last x epochs
#         if no_improvement_since >= early_stopping:
#             print(f'No more improvement for {early_stopping} epochs. Training stopped.')
#             break


# def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
#     """Validates NN."""
#     print('Validate NN')
#     loader.validation_set()
#     preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
#     num_char_err = 0
#     num_char_total = 0
#     num_word_ok = 0
#     num_word_total = 0
#     while loader.has_next():
#         iter_info = loader.get_iterator_info()
#         print(f'Batch: {iter_info[0]} / {iter_info[1]}')
#         batch = loader.get_next()
#         batch = preprocessor.process_batch(batch)
#         recognized, _ = model.infer_batch(batch)

#         print('Ground truth -> Recognized')
#         for i in range(len(recognized)):
#             num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
#             num_word_total += 1
#             dist = editdistance.eval(recognized[i], batch.gt_texts[i])
#             num_char_err += dist
#             num_char_total += len(batch.gt_texts[i])
#             print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
#                   '"' + recognized[i] + '"')

#     # print validation result
#     char_error_rate = num_char_err / num_char_total
#     word_accuracy = num_word_ok / num_word_total
#     print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
#     return char_error_rate, word_accuracy


# def infer(model: Model, fn_img: Path) -> None:
#     """Recognizes text in image provided by file path."""
#     img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
#     assert img is not None

#     preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
#     img = preprocessor.process_img(img)

#     batch = Batch([img], None, 1)
#     recognized, probability = model.infer_batch(batch, True)
#     print(f'Recognized: "{recognized[0]}"')
#     print(f'Probability: {probability[0]}')


# def parse_args() -> argparse.Namespace:
#     """Parses arguments from the command line."""
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
#     parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
#     parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
#     parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
#     parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
#     parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
#     parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
#     parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
#     parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

#     return parser.parse_args()


# def main():
#     """Main function."""

#     # parse arguments and set CTC decoder
#     args = parse_args()
#     decoder_mapping = {'bestpath': DecoderType.BestPath,
#                        'beamsearch': DecoderType.BeamSearch,
#                        'wordbeamsearch': DecoderType.WordBeamSearch}
#     decoder_type = decoder_mapping[args.decoder]

#     # train the model
#     if args.mode == 'train':
#         loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

#         # when in line mode, take care to have a whitespace in the char list
#         char_list = loader.char_list
#         if args.line_mode and ' ' not in char_list:
#             char_list = [' '] + char_list

#         # save characters and words
#         with open(FilePaths.fn_char_list, 'w') as f:
#             f.write(''.join(char_list))

#         with open(FilePaths.fn_corpus, 'w') as f:
#             f.write(' '.join(loader.train_words + loader.validation_words))

#         model = Model(char_list, decoder_type)
#         train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

#     # evaluate it on the validation set
#     elif args.mode == 'validate':
#         loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
#         model = Model(char_list_from_file(), decoder_type, must_restore=True)
#         validate(model, loader, args.line_mode)

#     # infer text on test image
#     elif args.mode == 'infer':
#         model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
#         infer(model, args.img_file)


# if __name__ == '__main__':
#     main()



# 


import os
# --- FIX: Force Legacy Keras Mode ---
import shutil
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# ------------------------------------

import argparse
import json
import cv2
import editdistance
import numpy as np
from path import Path
from typing import Tuple, List
from spellchecker import SpellChecker 

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

import os

def char_list_from_file():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    char_list_path = os.path.join(base_dir, 'model', 'charList.txt')
    
    print("Loading char list from:", char_list_path)  # debug line
    
    with open(char_list_path, 'r') as f:
        return f.read()


def detect_words(img):
    """
    SMART SEGMENTATION:
    Detects if the image is a single word or a sentence.
    It returns a list of cropped word images.
    """
    # 1. Convert to black and white (binary)
    #    Text becomes white, background becomes black
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # 2. DILATION (The Secret Sauce)
    #    We stretch the white pixels horizontally.
    #    - If letters are close (in one word), they merge into one blob.
    #    - If words are far apart, they remain separate blobs.
    #    Kernel (25, 5) means: Expand width by 25px, height by 5px.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)) 
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 3. Find Contours (The blobs)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Sort from Left to Right
    #    Computer finds blobs in random order. We must sort them by X coordinate.
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    word_imgs = []
    
    for ctr in contours:
        # Get position (x, y) and size (w, h)
        x, y, w, h = cv2.boundingRect(ctr)
        
        # Filter out tiny noise (dots/specks)
        if w < 10 or h < 10:
            continue
            
        # Crop the word from the original image
        # We add a little padding (2px) around it so letters aren't cut off
        curr_img = img[max(0, y-2):y+h+2, max(0, x-2):x+w+2]
        word_imgs.append(curr_img)

    return word_imgs


def infer(model: Model, fn_img: Path) -> None:
    # 1. Load Image
    full_img = cv2.imread(str(fn_img), cv2.IMREAD_GRAYSCALE)
    assert full_img is not None, f"Image not found at {fn_img}"

    print(f"\nAnalyzing image: {fn_img}...")

    # 2. Detect Words
    word_images = detect_words(full_img)
    
    if len(word_images) == 0:
        print("Error: No words detected.")
        return

    # --- DEBUG: Save cropped words to verify ---
    debug_dir = "../debug_words"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    for i, w_img in enumerate(word_images):
        cv2.imwrite(f"{debug_dir}/word_{i+1}.png", w_img)
    print(f"Detected {len(word_images)} segments (saved to {debug_dir}). Reading...\n")

    final_sentence = []
    spell = SpellChecker()
    # Add 'work' and 'level' to spell checker dictionary just in case
    spell.word_frequency.load_words(['work', 'level', 'line', 'online'])
    
    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)

    for i, img in enumerate(word_images):
        # --- NEW: INK THICKENER ---
        # This makes thin printed letters look like thick marker ink
        # which helps the AI recognize disjointed letters better.
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        # --------------------------

        processed_img = preprocessor.process_img(img)
        batch = Batch([processed_img], None, 1)
        
        recognized, probability = model.infer_batch(batch, True)
        raw_word = recognized[0]
        
        # Smart Correction
        corrected_word = spell.correction(raw_word)
        if corrected_word is None: corrected_word = raw_word
        
        print(f"  [Word {i+1}] AI: '{raw_word}' -> Auto-Correct: '{corrected_word}'")
        final_sentence.append(corrected_word)

    print("\n" + "=" * 40)
    print(f"FINAL RESULT: {' '.join(final_sentence)}")
    print("=" * 40 + "\n")


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/test.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    if args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file)
    else:
        print("For training, please revert to the original script.")

if __name__ == '__main__':
    main()




#new code

