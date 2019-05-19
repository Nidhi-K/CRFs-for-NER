# trainer.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from models import *


# Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
# are provded for convenience.
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, HMM, CRF)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class BadNerModel(object):
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence):
        pred_tags = []
        for tok in sentence.tokens:
            if tok.word in self.words_to_tag_counters:
                pred_tags.append(self.words_to_tag_counters[tok.word].argmax())
            else:
                pred_tags.append("O")
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def train_bad_ner_model(training_set):
    words_to_tag_counters = {}
    for sentence in training_set:
        tags = sentence.get_bio_tags()
        for idx in range(0, len(sentence)):
            word = sentence.tokens[idx].word
            if not word in words_to_tag_counters:
                words_to_tag_counters[word] = Counter()
            words_to_tag_counters[word].increment_count(tags[idx], 1.0)
    return BadNerModel(words_to_tag_counters)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train = read_data(args.train_path)
    dev = read_data(args.dev_path)
    # Here's a few sentences...
    #print("Examples of sentences:")
    #print(str(dev[1]))
    #print(str(dev[3]))
    #print(str(dev[5]))
    system_to_run = args.model
    # If set to True, runs your CRF on the test set to produce final output
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_ner_model(train)
        dev_decoded = [bad_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "HMM":
        hmm_model = train_hmm_model(train)
        dev_decoded = [hmm_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "CRF":
        crf_model = train_crf_model(train, dev)
        print("Data reading and training took %f seconds" % (time.time() - start_time))
        dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
        if args.run_on_test:
            print("Running on test")
            test = read_data(args.blind_test_path)
            test_decoded = [crf_model.decode(test_ex) for test_ex in test]
            print_output(test_decoded, args.test_output_path)
    else:
        raise Exception("Pass in either BAD, HMM, or CRF to run the appropriate system")
    # Print the evaluation statistics
    print_evaluation(dev, dev_decoded)
