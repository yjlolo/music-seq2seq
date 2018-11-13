import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseRNN, BaseModel
from model.attention import Attention
from util import utils


class EncoderRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, n_layers=1, rnn_cell='gru',
                 input_dropout_p=0, dropout_p=0, variable_lengths=False):
        super(EncoderRNN, self).__init__(input_size, hidden_size, n_layers, rnn_cell, input_dropout_p, dropout_p)

        self.variable_lengths = variable_lengths
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        input_var = self.input_dropout(input_var)
        if self.variable_lengths:
            input_var = nn.utils.rnn.pack_padded_sequence(input_var, input_lengths, batch_first=True)
        output, hidden = self.rnn(input_var)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden


class DecoderRNN(BaseRNN):

    KEY_ATTN_SCORE = 'attention_score'

    def __init__(self, input_size, hidden_size, n_layers=1, rnn_cell='gru',
                 input_dropout_p=0, dropout_p=0, teacher_forcing_threshold=1, use_attention=False):
        super(DecoderRNN, self).__init__(input_size, hidden_size, n_layers, rnn_cell, input_dropout_p, dropout_p)

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.teacher_forcing_threshold = teacher_forcing_threshold

        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.input_size)

    def forward_step(self, input_var, hidden, encoder_outputs):
        input_var = self.input_dropout(input_var)
        output, hidden = self.rnn(input_var, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out(output)

        #return output, hidden
        return torch.tanh(output), hidden, attn

    def forward(self, inputs, encoder_hidden, is_train=True, encoder_outputs=None):
        # if self.rnn_cell is nn.LSTM:
        #     batch_size = encoder_hidden[0].size(1)
        # elif self.rnn_cell is nn.GRU:
        #     batch_size = encoder_hidden.size(1)

        # max_length = inputs.size(1) - 1
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        decoder_hidden = encoder_hidden

        ################################################################
        # feedprevious defines the decoder forward time-step by time-step
        # only use when use_teacher_forcing=False

        def feedprevious(inputs, decoder_hidden, encoder_outputs):
            decoder_input = inputs[:, 0].unsqueeze(1)  # the start-of-sentence dummy
            decoder_output = torch.zeros(inputs[:, 1:].size()).to(inputs.device)

            for di in range(decoder_output.size(1)):
                decoder_input, decoder_hidden, step_attn = self.forward_step(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
                decoder_output[:, di, :] = decoder_input.squeeze(1)
                #ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            return decoder_output, decoder_hidden
        ################################################################

        use_teacher_forcing = True if random.random() > self.teacher_forcing_threshold and is_train else False

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]  # the expected decoder output
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            for di in range(decoder_output.size(1)):
                step_attn = attn[:, di, :] if attn is not None else None
                #ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
        else:
            decoder_output, decoder_hidden = feedprevious(inputs, decoder_hidden, encoder_outputs)

        return decoder_output, ret_dict


class Seq2seq(BaseModel):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable, is_train=True, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        """
        #print(encoder_hidden.size())
        encoder_hidden = encoder_hidden.view(encoder_hidden.size(1), -1)
        #print(encoder_hidden.size())
        encoder_hidden = torch.tanh(self.lin(encoder_hidden))
        #print(encoder_hidden.size())
        encoder_hidden = encoder_hidden.view(1, encoder_hidden.size(0), -1)
        #print(encoder_hidden.size())
        """
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              is_train=is_train)

        return result, encoder_hidden


if __name__ == '__main__':
    from data_loader import PMEmoDataLoader

    if torch.cuda.is_available():
        USE_CUDA = True
        device = torch.device('cuda')
    else:
        USE_CUDA = False
        device = torch.device('cpu')

    print(device)
    n_mel = 64
    hidden_size = 128
    n_layer = 2
    n_workers = 0
    drop = 0
    use_attn = True
    tf_thres = 1

    encoder = EncoderRNN(n_mel, hidden_size, n_layers=n_layer, rnn_cell='gru',
                         input_dropout_p=0, dropout_p=drop, variable_lengths=True)
    decoder = DecoderRNN(n_mel, hidden_size, n_layers=n_layer, rnn_cell='gru',
                         input_dropout_p=0, dropout_p=drop, use_attention=use_attn, teacher_forcing_threshold=tf_thres)
    model = Seq2seq(encoder, decoder).to(device)

    model.summary()

    dl = PMEmoDataLoader(batch_size=2, shuffle=False,
                         validation_split=0.0, num_workers=0, load_transformed='Spec_minmax_fixlen')
    batch = next(iter(dl))

    x, y, seqlen, songid, mask = batch[0], batch[1], batch[2], batch[3], batch[4]
    print(seqlen)
    batch_size = x.size(0)
    input_size = x.size(-1)
    input_var = x.to(device)
    sos = torch.zeros(batch_size, 1, input_size).to(device)  # begin of sentence
    target_var = torch.cat((sos, input_var), dim=1)
    (outputs, attn_score), enc_outputs = model(input_var, target_var, input_lengths=seqlen, is_train=False)

    #print(outputs.size(), enc_outputs.size())
