# 9장 자연어 처리에 의한 감정 분석(BERT)
# 구현에서 참조
# https://github.com/huggingface/pytorch-pretrained-BERT

# Copyright (c) 2018 Hugging Face
# Released under the Apache License 2.0
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/LICENSE



# 필요한 패키지 import
import copy
import math
import json
from attrdict import AttrDict
import collections

import torch
from torch import nn

from utils.tokenizer import BasicTokenizer, WordpieceTokenizer


def get_config(file_path):
    # 설정을 config.json에서 읽어들여, JSON의 사전 변수를 오브젝트 변수로 변환
    config_file = file_path  # "./weights/bert_config.json"

    # 파일을 열고, JSON으로 읽기
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)

    # 사전 변수를 오브젝트 변수로 변환
    config = AttrDict(json_object)

    return config


# BERT용으로 LayerNormalization 층을 정의합니다.
# 세부 구현을 TensorFlow에 맞추고 있습니다.
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization층입니다.
        학습된 모델을 그대로 로드하기 위해, 학습된 모델의 변수명으로 바꿉니다.
        원래의 GitHub 구현에서 변수명을 바꾸고 있습니다.
        weight→gamma, bias→beta
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weight에 대한 것
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # bias에 대한 것
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# BERT의 Embeddings 모듈입니다
class BertEmbeddings(nn.Module):
    """문장의 단어 ID열과, 첫번째인지 두번째 문장인지의 정보를, 내장 벡터로 변환한다
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # 3개의 벡터 표현 내장

        # Token Embedding: 단어 ID를 단어 벡터로 변환, 
        # vocab_size = 30522로, BERT의 학습된 모델에 사용된 vocabulary 양
        # hidden_size = 768로, 특징량 벡터의 길이는 768
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        # (주석)padding_idx=0은, idx=0의 단어 벡터는 0으로 한다. BERT의 vocabulary의 idx=0은 [PAD]임.

        # Transformer Positional Embedding: 위치 정보 텐서를 벡터로 변환
        # Transformer의 경우는 sin, cos로 이루어진 고정값이지만, BERT 학습시킴
        # max_position_embeddings = 512로, 문장의 길이는 512단어
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding: 첫번째, 두번째 문장을 벡터로 변환
        # type_vocab_size = 2
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # 작성한 LayerNormalization층
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids:  [batch_size, seq_len] 문장의 단어 ID를 나열
        token_type_ids: [batch_size, seq_len] 각 단어가 1번째 문장인지, 2번째 문장인지를 나타내는 id
        '''

        # 1. Token Embeddings
        # 단어 ID를 단어 벡터로 변환
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        # token_type_ids가 없는 경우는 문장의 모든 단어를 첫번째 문장으로 하여, 0으로 설정
        # input_ids와 같은 크기의 제로 텐서를 작성
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding: 
        # [0, 1, 2 ・・・]로 문장의 길이만큼 숫자가 하나씩 올라간다
        # [batch_size, seq_len]의 텐서 position_ids를 작성
        # position_ids를 입력해서, position_embeddings 층에서 768차원의 텐서를 꺼낸다
        seq_length = input_ids.size(1)  # 문장의 길이
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3개의 내장 텐서를 더한다 [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalization과 Dropout을 실행
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    '''BERT의 BertLayer 모듈입니다. Transformer가 됩니다'''

    def __init__(self, config):
        super(BertLayer, self).__init__()

        # Self-Attention 부분
        self.attention = BertAttention(config)

        # Self-Attention의 출력을 처리하는 전결합층
        self.intermediate = BertIntermediate(config)

        # Self-Attention에 의한 특징량과 BertLayer에 원래의 입력을 더하는 층
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states: Embedder 모듈의 출력 텐서 [batch_size, seq_len, hidden_size]
        attention_mask: Transformer의 마스크와 같은 기능의 마스킹
        attention_show_flg: Self-Attention의 가중치를 반환할지의 플래그
        '''
        if attention_show_flg == True:
            '''attention_show일 경우, attention_probs도 반환한다'''
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]


class BertAttention(nn.Module):
    '''BertLayer 모듈의 Self-Attention 부분입니다'''

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor: Embeddings 모듈 또는 앞단의 BertLayer에서의 출력
        attention_mask: Transformer의 마스크와 같은 기능의 마스킹입니다
        attention_show_flg: Self-Attention의 가중치를 반환할지의 플래그
        '''
        if attention_show_flg == True:
            '''attention_show일 경우, attention_probs도 반환한다'''
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class BertSelfAttention(nn.Module):
    '''BertAttention의 Self-Attention입니다'''

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads': 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size  # = 'hidden_size': 768

        # Self-Attention의 특징량을 작성하는 전결합층
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attention용으로 텐서의 형태를 변환한다
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states: Embeddings 모듈 또는 앞단의 BertLayer에서의 출력
        attention_mask: Transformer의 마스크와 같은 기능의 마스킹입니다
        attention_show_flg: Self-Attention의 가중치를 반환할지의 플래그
        '''
        # 입력을 전결합층에서 특징량 변환(주의, multi-head Attention 전부를 한꺼번에 변환하고 있습니다)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # multi-head Attention용으로 텐서 형태를 변환
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 특징량끼리를 곱해서 비슷한 정도를 Attention_scores로 구한다
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # 마스크가 있는 부분에 마스크를 적용합니다
        attention_scores = attention_scores + attention_mask
        # (비고)
        # 마스크는 곱셈이 아니라 덧셈이 직관적이지만, 그 후에 Softmax로 정규화하므로,
        # 마스크된 부분은 -inf로 합니다. attention_mask에는, 0이나-inf가
        # 원래 들어 있으므로 덧셈으로 합니다.

        # Attention을 정규화한다
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 드롭아웃합니다
        attention_probs = self.dropout(attention_probs)

        # Attention Map을 곱합니다
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attention의 텐서 형을 원래대로 되돌림
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_show일 경우, attention_probs도 반환한다
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    '''BertSelfAttention의 출력을 처리하는 전결합층입니다'''

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states: BertSelfAttention의 출력 텐서
        input_tensor: Embeddings 모듈 또는 앞단의 BertLayer에서의 출력
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    '''Gaussian Error Linear Unit라는 활성화 함수입니다.
    LeLU가 0으로 거칠고 불연속적이므로, 이를 연속적으로 매끄럽게 한 형태의 LeLU입니다.
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    '''BERT의 TransformerBlock 모듈의 FeedForward입니다'''

    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        # 전결합층: 'hidden_size': 768, 'intermediate_size': 3072
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # 활성화 함수 gelu
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states:  BertAttentionの출력テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # GELU에 의한 활성화
        return hidden_states


class BertOutput(nn.Module):
    '''BERT의 TransformerBlock 모듈의 FeedForward입니다'''

    def __init__(self, config):
        super(BertOutput, self).__init__()

        # 전결합층: 'intermediate_size': 3072, 'hidden_size': 768
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states:  BertIntermediate의 출력 텐서
        input_tensor: BertAttention의 출력 텐서
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# BertLayer 모듈의 반복 부분입니다
class BertEncoder(nn.Module):
    def __init__(self, config):
        '''BertLayer 모듈의 반복 부분입니다'''
        super(BertEncoder, self).__init__()

        # config.num_hidden_layers의 값. 즉, 12개의 BertLayer 모듈을 만듭니다
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states: Embeddings 모듈의 출력
        attention_mask: Transformer의 마스크와 동일한 기능의 마스킹입니다
        output_all_encoded_layers: 반환 값을 전체 TransformerBlock 모듈의 출력으로 할지, 
        최후 층만으로 한정할지의 플래그.
        attention_show_flg: Self-Attention의 가중치를 반환할지의 플래그
        '''

        # 반환 값으로 사용할 리스트
        all_encoder_layers = []

        # BertLayer 모듈의 처리를 반복
        for layer_module in self.layer:

            if attention_show_flg == True:
                '''attention_show의 경우, attention_probs도 반환한다'''
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            # 반환값으로 BertLayer에서 출력된 특징량을 12층 분, 모두 사용할 경우의 처리
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 반환값으로 최후의 BertLayer에서 출력된 특징량만을 사용할 경우의 처리
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        # attention_show의 경우, attention_probs(마지막 12단)도 반환한다
        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class BertPooler(nn.Module):
    '''입력 문장의 첫번째 단어[cls]의 특징량을 반환하고 유지하기 위한 모듈'''

    def __init__(self, config):
        super(BertPooler, self).__init__()

        # 전결합층, 'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 1번째 단어의 특징량을 취득
        first_token_tensor = hidden_states[:, 0]

        # 전결합층에서 특징량 변환
        pooled_output = self.dense(first_token_tensor)

        # 활성화 함수 Tanh를 계산
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):
    '''모듈을 전부 연결한 BERT 모델'''

    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3가지 모듈을 작성
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_ids:  [batch_size, sequence_length] 문장의 단어 ID를 나열
        token_type_ids:  [batch_size, sequence_length] 각 단어가 1번째 문장인지, 2번째 문장인지를 나타내는 id
        attention_mask: Transformer의 마스크와 같은 기능의 마스킹
        output_all_encoded_layers: 최후 출력에 12단의 Transformer의 전부를 리스트로 반환할지, 최후만인지를 지정
        attention_show_flg: Self-Attention의 가중치를 반환할지의 플래그
        '''

        # Attention의 마스크와 첫번째, 두번째 문장의 id가 없으면 작성한다
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 마스크 변형 [minibatch, 1, 1, seq_length]으로 한다
        # 나중에 multi-head Attention에서 사용할 수 있는 형태로 하고 싶으므로
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 마스크는 0, 1이지만 소프트 맥스를 계산할 때 마스크가 되도록, 0과 -inf으로 한다
        # -inf 대신 -10000으로 해 둡니다
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 순전파시킨다
        # BertEmbeddins 모듈
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayer 모듈(Transformer)을 반복하는 BertEncoder 모듈
        if attention_show_flg == True:
            '''attention_show의 경우, attention_probs도 반환한다'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPooler 모듈
        # encoder의 맨 마지막 BertLayer에서 출력된 특징량을 사용
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layers가 False인 경우는 리스트가 아니라, 텐서를 반환
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_show의 경우, attention_probs(가장 마지막)도 반환한다
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output


# 언어 모델 학습 모듈(추론시에는 사용하지 않음)
class BertPreTrainingHeads(nn.Module):
    '''BERT의 사전 학습 과제를 수행하는 어댑터 모듈'''

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        # 사전 학습 과제: Masked Language Model용 모델
        self.predictions = MaskedWordPredictions(config)

        # 사전 학습 과제: Next Sentence Prediction용 모델
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        '''입력 정보
        sequence_output:[batch_size, seq_len, hidden_size]
        pooled_output:[batch_size, hidden_size]
        '''
        # 입력의 마스크된 각 단어가 어떤 단어인지를 판정
        # 출력 [minibatch, seq_len, vocab_size]
        prediction_scores = self.predictions(sequence_output)

        # 선두 단어의 특징량에서 1번째, 2번째 문장이 연결되어 있는지 판정
        seq_relationship_score = self.seq_relationship(
            pooled_output)  # 출력 [minibatch, 2]

        return prediction_scores, seq_relationship_score


# 사전 학습 과제: Masked Language Model용 모듈
class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        '''사전 학습 과제: Masked Language Model용 모듈
        원래의 [2] 구현에서는, BertLMPredictionHead이라는 이름입니다.
        '''
        super(MaskedWordPredictions, self).__init__()

        # BERT에서 출력된 특징량을 변환하는 모듈(입출력 크기는 동일)
        self.transform = BertPredictionHeadTransform(config)

        # self.transform의 출력에서, 각 위치의 단어가 어떤 것인지를 알아맞히는 전결합층
        self.decoder = nn.Linear(in_features=config.hidden_size,  # 'hidden_size': 768
                                 out_features=config.vocab_size,  # 'vocab_size': 30522
                                 bias=False)
        # 바이어스 항
        self.bias = nn.Parameter(torch.zeros(
            config.vocab_size))  # 'vocab_size': 30522

    def forward(self, hidden_states):
        '''
        hidden_states: BERT에서의 출력[batch_size, seq_len, hidden_size]
        '''
        # BERT에서 출력된 특징량을 변환
        # 출력 크기: [batch_size, seq_len, hidden_size]
        hidden_states = self.transform(hidden_states)

        # 각 위치의 단어가 vocabulary의 어느 단어인지를 클래스 분류로 예측
        # 출력 크기: [batch_size, seq_len, vocab_size]
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    '''MaskedWordPredictions에서, BERT의 특징량을 변환하는 모듈(입출력 크기는 동일)'''

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        # 전결합층 'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # 활성화 함수 gelu
        self.transform_act_fn = gelu

        # LayerNormalization
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        '''hidden_statesはsequence_output:[minibatch, seq_len, hidden_size]'''
        # 전결합층에서 특징량 변환하여, 활성화 함수 gelu를 게산한 뒤, LayerNormalization을 수행
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# 사전 학습 과제: Next Sentence Prediction용 모듈
class SeqRelationship(nn.Module):
    def __init__(self, config, out_features):
        '''사전 학습 과제: Next Sentence Prediction용 모듈
        원래 인수[2] 구현에서는, 특별히 크래스로 제공하고 있지 않음.
        일반적인 전결합층에, 일부러 이름을 붙임.
        '''
        super(SeqRelationship, self).__init__()

        # 선두 단어의 특징량에서 1번째, 2번째 문장이 연결되어 있는지 판정하는 클래스 분류의 전결합층
        self.seq_relationship = nn.Linear(config.hidden_size, out_features)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)


class BertForMaskedLM(nn.Module):
    '''BERT 모델에 사전 학습 과제용의 어댑터 모듈
    BertPreTrainingHeads를 연결한 모델'''

    def __init__(self, config, net_bert):
        super(BertForMaskedLM, self).__init__()

        # BERT 모듈
        self.bert = net_bert

        # 사전 학습 과제용 어댑터 모듈
        self.cls = BertPreTrainingHeads(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        '''
        input_ids:  [batch_size, sequence_length] 문장의 단어 ID를 나열
        token_type_ids:  [batch_size, sequence_length] 각 단어가 1번째 문장인지, 2번째 문장인지를 나타내는 id
        attention_mask: Transformer의 마스크와 같은 기능의 마스킹
        '''

        # BERT의 기본 모델 부분의 순전파
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)

        # 사전 학습 과제의 추론을 실시
        prediction_scores, seq_relationship_score = self.cls(
            encoded_layers, pooled_output)

        return prediction_scores, seq_relationship_score


# 학습된 모델을 로드
def set_learned_params(net, weights_path = "./weights/pytorch_model.bin"):

    # 설정할 파라미터를 읽기
    loaded_state_dict = torch.load(weights_path)

    # 현재 네트워크 모델의 파라미터명
    net.eval()
    param_names = []  # 파라미터 이름을 저장해 나간다

    for name, param in net.named_parameters():
        param_names.append(name)

    # 현재 네트워크 정보를 복사하여 새로운 state_dict를 작성
    new_state_dict = net.state_dict().copy()

    # 새로운 state_dict에 학습된 값을 대입
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]  # 현재 네트워크의 파라미터명을 취득
        new_state_dict[name] = value  # 값을 넣는다
        print(str(key_name)+"→"+str(name))  # 어디로 들어갔는지 표시

        # 현재 네트워크의 파라미터를 전부 로드하면 끝낸다
        if (index+1 - len(param_names)) >= 0:
            break

    # 새로운 state_dict를 구축한 BERT 모델에 제공
    net.load_state_dict(new_state_dict)

    return net


# BERT용 Tokenizer
# vocab파일을 읽고, 
def load_vocab(vocab_file):
    """text 형식의 vocab 파일 내용을 사전에 저장합니다"""
    vocab = collections.OrderedDict()  # (단어, id) 순서의 사전 변수
    ids_to_tokens = collections.OrderedDict()  # (id, 단어) 순서의 사전 변수
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()

            # 저장
            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


# BasicTokenizer, WordpieceTokenizer는, 참고 문헌[2] 그대로입니다
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
# sub-word로 단어 분할을 실시하는 클래스들입니다.
class BertTokenizer(object):
    '''BERT용의 문장 단어 분할 클래스를 구현'''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        vocab_file: vocabulary에의 경로
        do_lower_case: 전처리에서 단어를 소문자로 바꾸는지 여부
        '''

        # vocabulary의 로드
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        # 분할 처리 함수를 "utils" 폴더에서 imoprt, sub-word로 단어 분할을 실시
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # (주석)위 단어는 도중에 분할하지 않는다. 이를 통해 하나의 단어로 간주함

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        '''문장의 단어를 분할하는 함수'''
        split_tokens = []  # 분할 후 단어들
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """분할된 단어 목록을 ID로 변환하는 함수"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """ID를 단어로 변환하는 함수"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
