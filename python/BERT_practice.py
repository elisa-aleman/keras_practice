import os
import pathlib
import re
import json
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from ProjectPaths import *

# “BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.

# It is a form of transfer learning

# 

max_len = 384
configuration = BertConfig()


##############

# # Save the slow pretrained tokenizer
# slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = make_model_path('bert_base_uncased')
# slow_tokenizer.save_pretrained(save_path)

##############

tokenizer = BertWordPieceTokenizer(os.path.join(save_path,'vocab.txt'), lowercase=True)

##############

# SQuAD : Stanford Question Answering Dataset (SQuAD)

train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

##############

class SquadExample:
	def __init__(self, question, context, start_char_idx, answer_text, all_answers):
		self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
    	context = self.context
    	question = self.question
    	answer_text = self.answer_text
    	start_char_idx = self.start_char_idx

    	# Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

                






                