# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE
    nmt.py analysis [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --num-layers=<int>                      number of layers [default: 1]
    --bidirectional                         whether the encoder is bidirectional
    --cost-fcn=<str>                        ground cost function for computing ot loss [default: none]
    --attention-type=<str>                  type of attention, types are none/dot/general/concat [default: none]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --self-attention                        use self attention for the decoder
    --embedding_file=<file>                 location of pre-trained file
    --omodel=<file>                         location of original model
    --pmodel=<file>                         location of some other model
    --tau=<float>                           annealing parameter [default: 0.01]
    --gamma=<float>                         trade-off between mle and ot loss [default: 0.2]
    --dec-cri=<str>                         criterion in choosing the final translation [default: none]
"""

import math
import model
import numpy as np
import os
import pickle
import sys
import time
import torch
import heapq
import importlib
import sys

#importlib.reload(sys)
#sys.setdefaultencoding('utf8')

from pdb import set_trace as bp
from collections import namedtuple
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.autograd import Variable
from torch.nn import functional as F
from typing import Any, Dict, List, Set, Tuple, Union
from tqdm import tqdm

from utils import batch_iter, read_corpus
from vocab import Vocab, VocabEntry
from geomloss import SamplesLoss

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


#def init_weights(model):
#    for m in model.children():
#        #print(m, type(m))
#        if type(m) == torch.nn.Linear:
#            torch.nn.init.xavier_uniform_(m.weight)
#            m.bias.data.fill_(0.01)
#            #print(m.weight)
#            #print(m.bias)
#        if type(m) == torch.nn.LSTM:
#            for name, param in m.named_parameters():
#                if 'bias' in name:
#                    # special treatment for forget gate : order is b_ii|b_if|b_ig|b_io
#                    torch.nn.init.constant_(param[len(param)//4:len(param)//2], 1.0)
#                    # bias for rest of the gates, set to zero
#                    torch.nn.init.constant_(param[:len(param)//4], 0.0)
#                    torch.nn.init.constant_(param[len(param)//2:3*len(param)//2], 0.0)
#                    torch.nn.init.constant_(param[3*len(param)//2:], 0.0)
#                elif 'weight' in name:
#                    torch.nn.init.xavier_normal_(param)
#                #print(name, len(param), param)


class NMT(object):

    def __init__(self, 
                 embed_size, 
                 hidden_size, 
                 vocab, 
                 dropout_rate,
                 num_layers,
                 bidirectional,
                 attention_type,
                 self_attention,
                 tau,
                 gamma,
                 cost_fcn,
                 uniform_init, embedding_file=None):
                    
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.tau = tau
        self.gamma = gamma
        self.cost_fcn = cost_fcn
        src_vocab_size = len(self.vocab.src.word2id)
        tgt_vocab_size = len(self.vocab.tgt.word2id)

        if embedding_file is not None:

             Glove = {}
             f = open(embedding_file)
             print("Loading the vectors.")

             i = 0
             for line in f:
                  if i != 0:
                      word, vec = line.split(' ', 1)
                      Glove[word] = np.fromstring(vec, sep=' ')
                  i += 1
             f.close()

             print("Done.")
             X_train = np.zeros((len(self.vocab.src.id2word), self.embed_size))

             for i in range(len(self.vocab.src.id2word)):
                  if self.vocab.src.id2word[i] in Glove:
                      X_train[i] = Glove[self.vocab.src.id2word[i]]

             embeddings = np.asarray(X_train)
        else:
             embeddings = None

        self.encoder = model.EncoderRNN(vocab_size=src_vocab_size,
                                        embed_size=self.embed_size,
                                        hidden_size=hidden_size,
                                        dropout_rate=dropout_rate,
                                        num_layers=num_layers,
                                        bidirectional=bidirectional, embeddings=embeddings)
        self.decoder = model.DecoderRNN(embed_size=self.embed_size,
                                        hidden_size=self.hidden_size,
                                        output_size=tgt_vocab_size,
                                        dropout_rate=dropout_rate,
                                        num_layers=num_layers,
                                        attention_type=attention_type,
                                        self_attention=self_attention,
                                        bidirectional=bidirectional)
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda() 
       
        # Initialize all parameter weights uniformly
        for param in list(self.encoder.parameters()) + list(self.decoder.parameters()):
          torch.nn.init.uniform(param, a=-uniform_init, b=uniform_init)

        self.criterion = torch.nn.CrossEntropyLoss(reduce=0).cuda()
        
    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_encodings, decoder_init_state = self.encode(src_sents)
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)

        return scores

    def encode(self, src_sents: List[List[str]]) -> Tuple[torch.Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        # Numberize the source sentences
        numb_src_sents = self.vocab.src.numberize(src_sents)

        # Sort from longest to smallest
        sorted_indices = sorted(range(len(src_sents)), key=lambda i: len(numb_src_sents[i]), reverse=True)

        # Sort numberized sentences
        numb_src_sents = [numb_src_sents[i] for i in sorted_indices]

        # Pad each sentence to the maximum length
        max_len = len(numb_src_sents[0])
        padded_src_sent = [sent + [0]*(max_len - len(sent)) for sent in numb_src_sents]

        # Get the original sentence lengths
        input_lengths = [len(sent) for sent in numb_src_sents]

        # Construct a long tensor (seq_len * batch_size)
        input_tensor = Variable(torch.LongTensor(padded_src_sent).t()).cuda()
        # Call encoder
        src_encodings, decoder_init_state = self.encoder(input_tensor, input_lengths)

        # Unsort
        unsorted_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i])
        src_encodings = src_encodings[:,unsorted_indices]
        decoder_init_state = [e[:,unsorted_indices] for e in decoder_init_state]
        return src_encodings, decoder_init_state
           
   

    def euclidean_cost(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
    
        """
        :param: x: tensor of shape (Batch_size, n, d)
        :param: y: tensor of shape (Batch_size, m, d)
        :param cost: a string determined the underlying cost function
         where (1) n = max_len of sequences in the input batches
               (2) m = max_len of sequences in the output batches (in general m=n)
               (3) d = dimension of the target embedding space

        :return: c: the cost matrix as a tensor of shape (Batch_size, n, m) 

        """

        # d(u,v) = 2-norm(u - v) = sqrt(<u,u> + <v,v> - 2*<u,v>)
        xx =  (x.norm(dim=2))**2
        yy =  (y.norm(dim=2))**2
        xy = torch.bmm(x, y.permute(0,2,1).contiguous())
        c = (xx.unsqueeze(2) + yy.unsqueeze(1) - 2*xy)**0.5

        return c
    
    def cosine_cost(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:

        """
        :param: x: tensor of shape (Batch_size, n, d)
        :param: y: tensor of shape (Batch_size, m, d)
        :param cost: a string determined the underlying cost function
         where (1) n = max_len of sequences in the input batches
               (2) m = max_len of sequences in the output batches (in general m=n)
               (3) d = dimension of the target embedding space

        :return: c: the cost matrix as a tensor of shape (Batch_size, n, m) 

        """

        
        # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
        #                          = dot(u / norm(u), v / norm(v))
        # We fist normalize the rows, before computing their dot products via transposition:
        x_norm = x / x.norm(dim=2).unsqueeze(2)
        y_norm = y / y.norm(dim=2).unsqueeze(2)
        c = torch.bmm(x_norm, y_norm.permute(0,2,1).contiguous())
        
        return c

   
    def decode(self, src_encodings: torch.Tensor, decoder_init_state: Any, tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens
        
    Args:
        src_encodings: hidden states of tokens in source sentences
        decoder_init_state: decoder GRU/LSTM's initial state
        tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`
        
    Returns:
        scores: could be a variable of shape (batch_size, ) representing the 
            log-likelihood of generating the gold-standard target sentence for 
            each example in the input batch
        """

        # Numberize the target sentences
        numb_tgt_sents = self.vocab.tgt.numberize(tgt_sents)

        # Pad each sentence to the maximum length
        max_len = max([len(sent) for sent in numb_tgt_sents])
        padded_tgt_sent = [sent + [0]*(max_len - len(sent)) for sent in numb_tgt_sents]
        
        # Get the original sentence lengths
        input_lengths = torch.cuda.FloatTensor([len(sent) for sent in numb_tgt_sents])

        # Construct a long tensor (seq_len * batch_size) 
        input_tensor = torch.cuda.LongTensor(padded_tgt_sent).t()  # shape = (max_len, batch_size)
        mle_scores = torch.zeros(input_tensor[0].size()).cuda()
        last_hidden = decoder_init_state    

        if self.bidirectional == True:
            context = torch.ones(1, len(tgt_sents), self.hidden_size * 2).cuda()
        else:
            context = torch.ones(1, len(tgt_sents), self.hidden_size).cuda()

        inputs = []
        outputs = []

        for t in range(1, max_len):
            # Get output from the decoder
            output, last_hidden, context = self.decoder(src_encodings, last_hidden, input_tensor[t-1].unsqueeze(0), context)
            # Record vector representation of the input and output
            output_dist = F.softmax(output.squeeze(0)/self.tau, dim=1) # soft-argmax, w_t^SA in the paper 
            outputs.append(output_dist.mm(self.decoder.embedding.weight))
            inputs.append(self.decoder.embedding(input_tensor[t])) 
            # Compute mle scores and add them
            mle_scores += self.criterion(output.squeeze(0), input_tensor[t]) * (input_lengths > t).float()

        # outputs and inputs have shape = (max_len-1, batch_size, embed_size), convert their shape to (batch_size, max_len-1, embed_size)
        input_vec = torch.stack(inputs).permute(1,0,2).contiguous().cuda()
        output_vec = torch.stack(outputs).permute(1,0,2).contiguous().cuda()
        
        # the probability distributions for computing OT loss, taking into consideration the padded zeros
        weights = [[1./(l-1)]*(int(l)-1) + [0]*(max_len-int(l)) for l in input_lengths.tolist()] # shape = (batch_size, max_len-1)
        input_weight = torch.cuda.FloatTensor(weights)
        output_weight = torch.cuda.FloatTensor(weights)
	
        # compute OT loss
        if self.cost_fcn == 'cosine':
            ot_Loss =  SamplesLoss("sinkhorn", cost = self.cosine_cost, backend="tensorized" )
        if self.cost_fcn == 'l2':
            ot_Loss =  SamplesLoss("sinkhorn", cost = self.euclidean_cost, backend="tensorized")
        
        Wass_xy = ot_Loss(input_weight, input_vec, output_weight, output_vec)
        return (mle_scores/(input_lengths - 1) + self.gamma * Wass_xy).mean(), mle_scores.sum() 



    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70, dec_cri: str='MLE') -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: list[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src, dec_init_state = self.encode([src_sent])
        if self.bidirectional == True:
            context = torch.ones(1, 1, self.hidden_size * 2).cuda()
        else:
            context = torch.ones(1 , 1, self.hidden_size).cuda()
        # Greedy Decoding for testing
        
        # previous_word = '<sos>'

        # greedy_ouput = []
        
        # for _ in range(max_decoding_time_step):
        
        #     if previous_word == '</s>':             
        #            break

        #     word_indices = self.vocab.tgt.words2indices([[previous_word]])
        #     word_indices = torch.cuda.LongTensor(word_indices)
        #     scores, dec_init_state = self.decoder(dec_init_state, word_indices)
        #     top_scores, score_indices = torch.topk(scores, k=1, dim=2)
        #     top_scores = top_scores[0][0].data.cpu().numpy().tolist()
        #     score_indices = score_indices[0][0].data.cpu().numpy().tolist()

        #     # greedy decoding
        #     max_score_word = self.vocab.tgt.word2id[scores_indices.index(max(top_scores))]
        #     greedy_ouput.append(max_score_word)

        #     # update previous word
        #     previous_word = max_score_word 
        
        # return [Hypothesis(x, hypotheses[x]) for x in greedy_ouput]

        def to_cpu(h):
          return [e.cpu().detach() for e in h]

        def to_cuda(h):
          return [e.cuda().detach() for e in h]

        # Beam search decoding
        hypotheses = {str(self.vocab.tgt.word2id['<s>']): [0, to_cpu(dec_init_state), context.cpu().detach()]}
        start_time = time.time() 
        for t in range(max_decoding_time_step):
            new_hypotheses = {}
            for hyp,(score,hidden,context) in hypotheses.items():
                previous_word = int(hyp.split()[-1])
                if previous_word == self.vocab.tgt.word2id['</s>']:
                    new_hypotheses[hyp] = [score,None,None]
                    continue

                # Create a tensor for the last word
                last_word = torch.cuda.LongTensor([[previous_word]])

                # Pass through the decoder
                scores, new_hidden, new_context= self.decoder(src, to_cuda(hidden), last_word, context.cuda().detach())
                new_hidden = to_cpu(new_hidden)
                new_context = new_context.cpu().detach()
                scores = F.log_softmax(scores, dim=2)
                top_scores, score_indices = torch.topk(scores, k=beam_size+1, dim=2)

                # If we get UNK, do one more step. Otherwise skip the last step.
                seen_unk = False
                for i in range(beam_size+1):
                  if i == beam_size and not seen_unk:
                    continue

                  word_index = score_indices[0,0,i].item()
                  if word_index == self.vocab.tgt.unk_id:
                    seen_unk = True
                    continue

                  word = str(word_index)
                  new_score = score + top_scores[0,0,i].item()
                  new_hypotheses[hyp + " " + word] = [new_score, new_hidden, new_context]

            # Prune the hypotheses for the next step
            hypotheses = dict(sorted(new_hypotheses.items(), key=lambda t: t[1][0]/len(t[0].split()), reverse=True)[:beam_size])
        #print(" %s --- beam" %(time.time() - start_time))
        def _denumberize(s):
          nums = [int(e) for e in s.split()]
          return self.vocab.tgt.denumberize(nums)

        if dec_cri == 'OT':
            # incorporate OT distance as selection criterion
            numb_src_sent = self.vocab.src.numberize(src_sent)
            src_tensor = torch.cuda.LongTensor(numb_src_sent)
            n = len(src_tensor)
            src_weight = torch.cuda.FloatTensor([[1./n] * n]).squeeze(0)
            src_vec = self.encoder.embedding(src_tensor)
            for h in hypotheses:
                numb_h = [int(e) for e in h[0].split()]
                h_tensor = torch.cuda.LongTensor(numb_h)
                m = len(h_tensor)
                h_weight = torch.cuda.FloatTensor([[1./m] * m]).squeeze(0)
                h_vec = self.decoder.embedding(h_tensor)
                if self.cost_fcn == 'cosine':
                    ot_Loss =  SamplesLoss("sinkhorn", cost = self.cosine_cost, backend="tensorized" )
                if self.cost_fcn == 'l2':
                    ot_Loss =  SamplesLoss("sinkhorn", cost = self.euclidean_cost, backend="tensorized")
                Wass_xy = ot_Loss(src_weight, src_vec, h_weight, h_vec)
                hypotheses[h][0] = Wass_xy.item() 

        return [Hypothesis(_denumberize(x), hypotheses[x][0]) for x in hypotheses] # namedtuple('Hypothesis', hypotheses.keys())(**hypotheses) 
        

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """
        # Set model to eval
        self.encoder.eval()
        self.decoder.eval()

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            #loss = -self.model(src_sents, tgt_sents).sum()
            src_encodings, decoder_init_state = self.encode(src_sents)
            loss = self.decode(src_encodings, decoder_init_state, tgt_sents)[1]

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        # Set model back to train
        self.encoder.train()
        self.decoder.train()

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        model.encoder.LSTM.flatten_parameters()
        model.decoder.LSTM.flatten_parameters()
        return model

    def save(self, model_path: str):
        """
        Save current model to file
        """
        torch.save(self, model_path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    lr = float(args['--lr'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))


    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab,
                num_layers=int(args['--num-layers']),
                bidirectional=args['--bidirectional'],
                tau=float(args['--tau']),
                gamma=float(args['--gamma']),
                attention_type=args['--attention-type'],
                cost_fcn = args['--cost-fcn'],
                self_attention=args['--self-attention'],
                uniform_init=float(args['--uniform-init']),
                embedding_file=args['--embedding_file'])
    

    # Set training to true
    model.encoder.train()
    model.decoder.train()

    # model.cuda() or model = model.cuda() or model = NMT().cuda() # error: model has no attribute cuda

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # Define an Adam optimizer
    optim = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            # Zero out the gradients
            optim.zero_grad()

            train_iter += 1

            batch_size = len(src_sents)

            # (batch_size)
            start_time = time.time()
            loss, sum_loss = model(src_sents, tgt_sents)
            #print("forward", time.time() - start_time)
            #report_loss += loss.item()
            #for now using the sum_loss to calculate report_loss
            report_loss += sum_loss.item()
            cum_loss += sum_loss.item()

            # TODO: ensure that this can actually be called
            loss.backward()
            #print("backwards", time.time() - start_time)

            # Clip gradient norms
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) + list(model.decoder.parameters()), clip_grad)

            # Do a step of the optimizer
            optim.step()
            #print("step", time.time() - start_time)

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         np.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss/ cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=16)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # Save optimizer
                    torch.save(optim.state_dict(), 'optim.save')

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = model.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optim = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)
                        optim.load_state_dict(torch.load('optim.save'))
                        for state in optim.state.values():
                          for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                              state[k] = v.cuda()
                        for group in optim.param_groups:
                          group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, dec_cri: str) -> List[List[Hypothesis]]:

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, dec_cri=dec_cri)
        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    if os.path.exists(args['MODEL_PATH']):
        model = NMT.load(args['MODEL_PATH'])
    else:
        model = NMT(256, 256, pickle.load(open('data/vocab.bin', 'rb')))

    # Set models to eval (disables dropout)
    model.encoder.eval()
    model.decoder.eval()

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             dec_cri=args['--dec-cri'])

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value.split()[1:-1])
            f.write(hyp_sent + '\n')

    # Back to train (not really necessary for now)
    model.encoder.train()
    model.decoder.train()


def analyse(args: Dict[str, str]):
    """
    Creates files for the purpose of analysis
    """

    omodel = NMT.load(args['--omodel'])
    pmodel = NMT.load(args['--pmodel'])
    hehe = dict()

    hehe['o_embedding'] = omodel.encoder.embedding.weight.data.cpu().numpy()
    hehe['p_embedding'] = pmodel.encoder.embedding.weight.data.cpu().numpy()
    hehe['vocab'] = omodel.vocab.src.word2id
    dumper = open('results/en-az/dumped.bin', 'wb')
    pickle.dump(hehe, dumper)

def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    elif args['analysis']:
        analyse(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
