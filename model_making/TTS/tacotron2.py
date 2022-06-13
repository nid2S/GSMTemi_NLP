import os
import logging
import numpy as np
from typing import Sequence, Optional, Tuple

import torch
import torch.nn.functional as F
from hparams import hparams as hps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.INFO)
info_logger.addHandler(logging.StreamHandler())

class LocationAwareAttention(torch.nn.Module):
    """
    LocationAware(== LocationSensitive)Attention.
    cloned from https://github.com/sooftware/attentions/blob/master/attentions.py

    Argments:
        - hidden_dim (int): dimesion of hidden state vector
        - smoothing (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = torch.nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = torch.nn.Linear(hidden_dim, 1, bias=True)
        self.bias = torch.nn.Parameter(torch.nn.init.uniform_(torch.rand(hidden_dim), -0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query: torch.Tensor, value: torch.Tensor, last_attn: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
            self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
            + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
            + conv_attn
            + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD

        return context, attn

class Tacotron2(torch.nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        # embeddings
        self.char_embedding_layer = torch.nn.Embedding(len(symbols), hps.embedding_size, padding_idx=hps.pad_token_idx, dtype=torch.float32)
        torch.nn.init.trunc_normal_(self.char_embedding_layer.weight, std=0.5)
        self.speaker_embedding_layer = torch.nn.Embedding(hps.num_speakers, hps.speaker_embedding_size, dtype=torch.float32)
        torch.nn.init.trunc_normal_(self.speaker_embedding_layer.weight, std=0.5)

        # encoder
        self.encoder_init_dense = torch.nn.Sequential(  # init dense for init state of bidirectional LSTM
            torch.nn.Linear(hps.speaker_embedding_size, hps.encoder_lstm_units*4), torch.nn.Softsign())
        self.decoder_init_denses = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(hps.speaker_embedding_size, hps.decoder_lstm_units*2), torch.nn.Softsign())
             for _ in range(2)]
        )
        self.encoder_conv_layers = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv1d(hps.embedding_size if i == 1 else hps.enc_conv_channels, hps.enc_conv_channels, kernel_size=hps.enc_conv_kernel_size, padding='same'),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hps.enc_conv_channels),
                torch.nn.Dropout(p=hps.dropout_prob)
            ) for i in range(3)]
        )
        self.encoder_BiLSTM = torch.nn.LSTM(hps.enc_conv_channels, hps.encoder_lstm_units,
                                            batch_first=True, bidirectional=True, dtype=torch.float32)

        # decoder
        self.attention = LocationAwareAttention(hps.attention_size)
        pass

        # postnet
        self.post_net_convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(hps.postnet_channels, hps.postnet_channels, kernel_size=hps.postnet_kernel_size, padding="same"),
            torch.nn.Tanh() if i != (hps.postnet_num_layers - 1) else None,
            torch.nn.Dropout(p=hps.dropout_prob)
        ) for i in range(hps.postnet_num_layers)])
        self.residual = torch.nn.Linear(hps.postnet_channels, hps.num_mels)
        self.postnet_output_layer = torch.nn.Linear(hps.num_mels, hps.num_freq)

    def forward(self, inputs: Sequence[int], speaker_id: Sequence[int] = None, mel_targets=None, is_training=False):
        # embedding
        char_embedded_inputs = self.char_embedding_layer(inputs)  # [batch, input_len, embedding_size]
        if hps.num_speakers > 1:
            speaker_embed = self.speaker_embedding_layer(speaker_id)  # [batch, input_len, speaker_embedding_size]
            encoder_rnn_init_state = self.encoder_init_dense(speaker_embed)
            decoder_rnn_init_states = [decoder_init_dense(speaker_embed) for decoder_init_dense in self.decoder_init_denses]
        else:  # hps.num_speakers == 1
            encoder_rnn_init_state = None
            decoder_rnn_init_states = None
            attention_rnn_init_state = None

        # encoder
        x = char_embedded_inputs
        for encoder_conv_layer in self.encoder_conv_layers:
            x = encoder_conv_layer(x)
        encoder_conv_output = x
        if encoder_rnn_init_state is not None:
            c_fw, h_fw, c_bw, h_bw = torch.split(encoder_rnn_init_state, hps.encoder_lstm_units, -1)
            h_0 = torch.concat((h_fw.unsqueeze(0), h_bw.unsqueeze(0)), dim=0)
            c_0 = torch.concat((h_fw.unsqueeze(0), h_bw.unsqueeze(0)), dim=0)
        else:  # single mode
            h_0 = torch.zeros(1, hps.batch_size, hps.encoder_lstm_units, requires_grad=False)
            c_0 = torch.zeros(1, hps.batch_size, hps.encoder_lstm_units, requires_grad=False)
        encoder_outputs, states = self.encoder_BiLSTM(x, (h_0, c_0))

        # decoder
        # TODO 어텐션에 입력을 넣고 zero_state(decoder_lstm)로 어텐션이 반영된 값을 빼오는 형식.
        attention_mechanism = LocationSensitiveAttention(hps.attention_size, encoder_outputs, hparams=hp,
                                                         is_training=is_training,
                                                         mask_encoder=hps.mask_encoder,
                                                         memory_sequence_length=input_lengths, smoothing=hps.smoothing,
                                                         cumulate_weights=hps.cumulative_weights)
        decoder_lstm = [ZoneoutLSTMCell(hps.decoder_lstm_units, is_training, zoneout_factor_cell=hps.tacotron_zoneout_rate,
                        zoneout_factor_output=hps.tacotron_zoneout_rate, name='decoder_LSTM_{}'.format(i + 1)) for i in range(hps.decoder_layers)]
        decoder_lstm = tf.contrib.rnn.MultiRNNCell(decoder_lstm, state_is_tuple=True)
        decoder_init_state = decoder_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)  # 여기서 zero_state를 부르면, 위의 AttentionWrapper에서 이미 넣은 준 값도 포함되어 있다.
        if hps.model_type == "multi-speaker":
            decoder_init_state = list(decoder_init_state)
            for idx, cell in enumerate(decoder_rnn_init_states):
                shape1 = decoder_init_state[idx][0].get_shape().as_list()
                shape2 = cell.get_shape().as_list()
                if shape1[1] * 2 != shape2[1]:
                    raise Exception(" [!] Shape {} and {} should be equal".format(shape1, shape2))
                c, h = tf.split(cell, 2, 1)
                decoder_init_state[idx] = LSTMStateTuple(c, h)

            decoder_init_state = tuple(decoder_init_state)
        # output_attention=False -> attention_layer_size에 값을 넣지 않아 attention == context vector가 됨.
        attention_cell = AttentionWrapper(decoder_lstm, attention_mechanism,
                                           initial_cell_state=decoder_init_state, alignment_history=True, output_attention=False)
        # Decoder input -> prenet -> decoder_lstm -> concat[output, attention]
        dec_prenet_outputs = DecoderWrapper(attention_cell, is_training, hps.dec_prenet_sizes, hps.dropout_prob, hps.inference_prenet_dropout)
        dec_outputs_cell = OutputProjectionWrapper(dec_prenet_outputs, (hps.num_mels + 1) * hps.reduction_factor)
        if is_training:
            helper = TacoTrainingHelper(mel_targets, hps.num_mels, hps.reduction_factor)  # inputs은 batch_size 계산에만 사용
        else:
            helper = TacoTestHelper(batch_size, hps.num_mels, hps.reduction_factor)
        decoder_init_state = dec_outputs_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        (decoder_outputs, _), final_decoder_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(BasicDecoder(dec_outputs_cell, helper, decoder_init_state),
                                              maximum_iterations=int(hps.max_n_frame / hps.reduction_factor))
        decoder_mel_outputs = tf.reshape(decoder_outputs[:, :, :hps.num_mels * hps.reduction_factor], [batch_size, -1, hps.num_mels])  # [N,iters,400] -> [N,5*iters,80]
        stop_token_outputs = tf.reshape(decoder_outputs[:, :, hps.num_mels * hps.reduction_factor:], [batch_size, -1])  # [N,iters]

        # Postnet
        x = decoder_mel_outputs
        for post_net_conv in self.post_net_convs:
            x = post_net_conv(x)
        residual = self.residual(x)
        mel_outputs = decoder_mel_outputs + residual

        # Add post-processing CBHG:
        # mel_outputs: (N,T,num_mels)
        post_outputs = cbhg(mel_outputs, None, is_training, hps.post_bank_size, hps.post_bank_channel_size,
                            hps.post_maxpool_width, hps.post_highway_depth, hps.post_rnn_size,
                            hps.post_proj_sizes, hps.post_proj_width, scope='post_cbhg')
        postnet_outputs = self.postnet_output_layer(post_outputs)
        # Grab alignments from the final decoder state: batch_size, text length(encoder), target length(decoder)
        alignments = torch.permute(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        def log_self():
            info_logger.info('=' * 40)
            info_logger.info(' model_type: %s' % hps.model_type)
            info_logger.info('=' * 40)

            info_logger.info('Initialized Tacotron model. Dimensions: ')
            info_logger.info('\tembedding:                        %d' % char_embedded_inputs.shape[-1])
            info_logger.info('\tencoder conv out:                 %d' % encoder_conv_output.shape[-1])
            info_logger.info('\tencoder out:                      %d' % encoder_outputs.shape[-1])
            info_logger.info('\tattention out:                    %d' % attention_cell.output_size)
            info_logger.info('\tdecoder prenet lstm concat out :  %d' % dec_prenet_outputs.output_size)
            info_logger.info('\tdecoder cell out:                 %d' % dec_outputs_cell.output_size)
            info_logger.info('\tdecoder out:                      %d' % decoder_outputs.shape[-1])
            info_logger.info('\tdecoder mel out:                  %d' % decoder_mel_outputs.shape[-1])
            info_logger.info('\tmel out:                          %d' % mel_outputs.shape[-1])
            info_logger.info('\tpostnet out:                      %d' % post_outputs.shape[-1])
            info_logger.info('\tlinear out:                       %d' % postnet_outputs.shape[-1])
            info_logger.info('\tTacotron Parameters               {:.3f} Million.'.format(
                np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))

        return decoder_mel_outputs, stop_token_outputs, postnet_outputs, alignments
