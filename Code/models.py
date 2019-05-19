# models.py

from nerdata import *
from utils import *
from optimizers import *
import numpy as np
from math import *

# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        scores = np.zeros((len(sentence),len(self.tag_indexer))) #shape: (timesteps x tags)
        back_pointer = np.zeros((len(sentence),len(self.tag_indexer))) #shape: (timesteps x tags)
        pss = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)

        # Viterbi - Initialization 
        for cur_tag_idx in range(len(self.tag_indexer)):
            scores[0][cur_tag_idx] = pss.score_init(sentence, cur_tag_idx) + pss.score_emission(sentence, cur_tag_idx, 0)

        # Viterbi - Recurrence 
        for t in range(1, len(sentence)):
            for cur_tag_idx in range(len(self.tag_indexer)):
                s_em = pss.score_emission(sentence, cur_tag_idx, t)
                for prev_tag_idx in range(len(self.tag_indexer)):
                    s_trans = pss.score_transition(sentence, prev_tag_idx, cur_tag_idx)
                    cur_score = scores[t-1][prev_tag_idx] + s_trans + s_em
                    if prev_tag_idx == 0:
                        scores[t][cur_tag_idx] = cur_score
                        back_pointer[t][cur_tag_idx] = prev_tag_idx
                    elif cur_score > scores[t][cur_tag_idx]:
                        scores[t][cur_tag_idx] = cur_score 
                        back_pointer[t][cur_tag_idx] = prev_tag_idx

        # Viterbi - Final State 
        max_score = scores[len(sentence)-1][0]
        bp = 0
        for tag_idx in range(len(self.tag_indexer)):
            if scores[len(sentence)-1][tag_idx] > max_score:
                max_score = scores[len(sentence)-1][tag_idx]
                bp = tag_idx

        # Viterbi - backpointer 
        pred_tags = []
        for t in reversed(range(len(sentence))):
            pred_tags.append(self.tag_indexer.get_object(bp))
            bp = back_pointer[t][int(bp)]
                 
        pred_tags.reverse()
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i - 1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    #print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    #print("Tag indexer: %s" % tag_indexer)
    #print("Initial state log probabilities: %s" % init_counts)
    #print("Transition log probabilities: %s" % transition_counts)
    #print("Emission log probs too big to print...")
    #print("Emission log probs for India: %s" % emission_counts[:, word_indexer.get_index("India")])
    #print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.get_index("Phil")])
    #print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)

class FeatureBasedSequenceScorer(object):
    def __init__(self, weights, feature_cache, transition_log_probs):
        self.weights = weights
        self.feature_cache = feature_cache
        self.transition_log_probs = transition_log_probs

    def score_transition(self, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx][curr_tag_idx]

    def score_emission(self, word_idx, tag_idx):
        feats = self.feature_cache[word_idx][tag_idx]
        return score_indexed_features(feats, self.weights)

class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_log_probs):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_log_probs = transition_log_probs
    
    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        num_words = len(sentence)
        num_tags = len(self.tag_indexer)
        test_feature_cache = [[[] for k in range(num_tags)] for j in range(num_words)]
        
        for cur_tag_idx in range(num_tags):
            cur_tag = self.tag_indexer.get_object(cur_tag_idx)
            for prev_tag_idx in range(num_tags):
                prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                if prev_tag[0] == 'O' and cur_tag[0] == 'I':
                    self.transition_log_probs[prev_tag_idx][cur_tag_idx] = -np.inf
                elif cur_tag[0] == 'I':
                    if prev_tag[2:] != cur_tag[2:]:
                        self.transition_log_probs[prev_tag_idx][cur_tag_idx] = -np.inf
        
        # Filling feature cache
        for word_idx in range(num_words):
            for tag_idx in range(num_tags):
                tag = self.tag_indexer.get_object(tag_idx)
                test_feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, tag, self.feature_indexer, add_to_indexer=False)

        fss = FeatureBasedSequenceScorer(self.feature_weights, test_feature_cache, self.transition_log_probs)

        scores = np.zeros((num_words,num_tags)) #shape: (timesteps x tags)
        back_pointer = np.zeros((num_words, num_tags)) #shape: (timesteps x tags)

        # Viterbi - Initialization 
        for tag_idx in range(num_tags):
            scores[0][tag_idx] = fss.score_emission(0, tag_idx)
                
        # Viterbi - Recurrence 
        for word_idx in range(1, num_words):
            for cur_tag_idx in range(num_tags):
                s_em = fss.score_emission(word_idx, cur_tag_idx)
                for prev_tag_idx in range(num_tags):
                    s_trans = fss.score_transition(prev_tag_idx, cur_tag_idx)
                    cur_score = scores[word_idx-1][prev_tag_idx] + s_trans + s_em
                    if prev_tag_idx == 0:
                        scores[word_idx][cur_tag_idx] = cur_score
                        back_pointer[word_idx][cur_tag_idx] = prev_tag_idx
                    elif cur_score > scores[word_idx][cur_tag_idx]:
                        scores[word_idx][cur_tag_idx] = cur_score 
                        back_pointer[word_idx][cur_tag_idx] = prev_tag_idx

        # Viterbi - Final State 
        max_score = scores[num_words-1][0]
        bp = 0
        for tag_idx in range(num_tags):
            if scores[num_words-1][tag_idx] > max_score:
                max_score = scores[num_words-1][tag_idx]
                bp = tag_idx

        # Viterbi - backpointer 
        pred_tags = []
        for t in reversed(range(num_words)):
            pred_tags.append(self.tag_indexer.get_object(bp))
            bp = back_pointer[t][int(bp)]
                 
        pred_tags.reverse()
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences, dev):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True) 
    num_sentences = len(sentences)
    num_tags = len(tag_indexer)
    num_epochs = 20
    hmm_model = train_hmm_model(sentences);
    transition_log_probs = hmm_model.transition_log_probs
    
    weights = np.zeros(len(feature_indexer))
    grad_ascent = UnregularizedAdagradTrainer(weights)
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for sentence_idx in range(num_sentences):
            if sentence_idx % 100 == 0:
                print("Ex %i/%i" % (sentence_idx, len(sentences)))

            # Forward-backward algorithm for each sentence
            num_words = len(sentences[sentence_idx])
            alpha = np.zeros((num_words,num_tags)) # forward matrix
            beta = np.zeros((num_words,num_tags)) # backward matrix

            # Forward-backward initialization for alpha and beta
            for tag_idx in range(num_tags): 
                alpha[0][tag_idx] = grad_ascent.score(feature_cache[sentence_idx][0][tag_idx]) # Emission log probability
                beta[num_words - 1][tag_idx] = 0 # log(1) = 0

            # Forward recurrence for alpha
            for word_idx in range(1, num_words):
                for cur_tag_idx in range(num_tags):
                    emission_log_prob = grad_ascent.score(feature_cache[sentence_idx][word_idx][cur_tag_idx])
                    for prev_tag_idx in range(num_tags):
                        transition_log_prob = transition_log_probs[prev_tag_idx][cur_tag_idx]
                        cur_term = emission_log_prob + transition_log_prob + alpha[word_idx - 1][prev_tag_idx]
                        if prev_tag_idx == 0:
                            alpha[word_idx][cur_tag_idx] = cur_term 
                        else:
                            alpha[word_idx][cur_tag_idx] = np.logaddexp(alpha[word_idx][cur_tag_idx], cur_term) 

            # Backward recurrence for beta
            for word_idx in reversed(range(num_words-1)):
                for cur_tag_idx in range(num_tags):
                    for next_tag_idx in range(num_tags):
                        emission_log_prob = grad_ascent.score(feature_cache[sentence_idx][word_idx+1][next_tag_idx])
                        transition_log_prob = transition_log_probs[cur_tag_idx][next_tag_idx]
                        cur_term = emission_log_prob + transition_log_prob + beta[word_idx + 1][next_tag_idx]
                        if next_tag_idx == 0:
                            beta[word_idx][cur_tag_idx] = cur_term 
                        else:
                            beta[word_idx][cur_tag_idx] = np.logaddexp(beta[word_idx][cur_tag_idx], cur_term) 
                        
            # Computing marginal probabilites denominator
            log_marg_probs_denom = np.zeros(num_words)
            for word_idx in range(num_words):
                log_marg_probs_denom[word_idx] = alpha[word_idx][0] + beta[word_idx][0]
                for tag_idx in range(1,num_tags):
                    cur_term = alpha[word_idx][tag_idx] + beta[word_idx][tag_idx]
                    log_marg_probs_denom[word_idx] = np.logaddexp(log_marg_probs_denom[word_idx], cur_term)

            # Computing marginal probabilites 
            log_marg_probs = np.zeros((num_words, num_tags))
            for word_idx in range(num_words):
                for tag_idx in range(num_tags):
                    cur_term = alpha[word_idx][tag_idx] + beta[word_idx][tag_idx]
                    log_marg_probs[word_idx][tag_idx] = cur_term - log_marg_probs_denom[word_idx] 

            # Computing gradient
            gradient_counter = Counter()
            for word_idx in range(num_words):
                gold_tag = sentences[sentence_idx].get_bio_tags()[word_idx] 
                gold_tag_idx = tag_indexer.index_of(gold_tag)
                for feature_pos in feature_cache[sentence_idx][word_idx][gold_tag_idx]:
                    gradient_counter.increment_count(feature_pos, 1.0)
                
                for tag_idx in range(num_tags):
                    for feature_pos in feature_cache[sentence_idx][word_idx][tag_idx]:
                        gradient_counter.increment_count(feature_pos, -exp(log_marg_probs[word_idx][tag_idx]))
            grad_ascent.apply_gradient_update(gradient_counter,1)
                     
        crfmodel = CrfNerModel(tag_indexer, feature_indexer, grad_ascent.weights, transition_log_probs)
        dev_result = [crfmodel.decode(test_ex) for test_ex in dev]
        print("F1 Dev:")
        print_evaluation(dev,dev_result)
    return CrfNerModel(tag_indexer, feature_indexer, grad_ascent.weights, transition_log_probs)
                    
# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
