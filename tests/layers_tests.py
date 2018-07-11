import unittest
import torch as T

from TTS.layers.tacotron import Prenet, CBHG, Decoder, Encoder
from TTS.layers.losses import L1LossMasked, sequence_mask
from TTS.layers.attention import AttentionRNNCell


class PrenetTests(unittest.TestCase):

    def test_in_out(self):
        layer = Prenet(128, out_features=[256, 128])
        dummy_input = T.rand(4, 128)

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 128


class CBHGTests(unittest.TestCase):

    def test_in_out(self):
        layer = CBHG(128, K=6, projections=[128, 128], num_highways=2)
        dummy_input = T.rand(4, 8, 128)

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256


class DecoderTests(unittest.TestCase):

    def test_in_out(self):
        layer = Decoder(in_features=256, memory_dim=80, r=2)
        dummy_input = T.rand(4, 8, 256)
        dummy_memory = T.rand(4, 2, 80)

        output, alignment, stop_tokens = layer(dummy_input, dummy_memory)

        assert output.shape[0] == 4
        assert output.shape[1] == 1, "size not {}".format(output.shape[1])
        assert output.shape[2] == 80 * 2, "size not {}".format(output.shape[2])
        assert stop_tokens.shape[0] == 4
        assert stop_tokens.max() <= 1.0
        assert stop_tokens.min() >= 0

        # check with input lengths
        layer = Decoder(in_features=256, memory_dim=80, r=2)
        dummy_input = T.rand(4, 8, 256)
        dummy_memory = T.rand(4, 2, 80)
        input_lens = T.randint(8, (4,)).long()
        input_lens[0] = 8

        output, alignment, stop_tokens = layer(dummy_input, dummy_memory, input_lens)

        assert output.shape[0] == 4
        assert output.shape[1] == 1, "size not {}".format(output.shape[1])
        assert output.shape[2] == 80 * 2, "size not {}".format(output.shape[2])
        assert stop_tokens.shape[0] == 4
        assert stop_tokens.max() <= 1.0
        assert stop_tokens.min() >= 0


class EncoderTests(unittest.TestCase):

    def test_in_out(self):
        layer = Encoder(128)
        dummy_input = T.rand(4, 8, 128)

        print(layer)
        output = layer(dummy_input)
        print(output.shape)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256  # 128 * 2 BiRNN


class L1LossMaskedTests(unittest.TestCase):

    def test_in_out(self):
        layer = L1LossMasked()
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.ones(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.data[0])
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0)
                * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.data[0])


class AttentionRNNCellTests(unittest.TestCase):

    def test_in_out(self):
        layer = AttentionRNNCell(64, 64 , 64)
        annot = T.ones(4, 8, 64)
        memory = T.ones(4, 64)
        context = T.ones(4, 64)
        rnn_state = T.zeros(4, 64)
        # input -> output test
        rnn_state, context, alignment = layer.forward(memory, context,
                                                      rnn_state, annot)
        # masking test
        annot_lens = T.randint(8, [4]).long()
        annot_lens[0] = 8
        rnn_state, context, alignment = layer.forward(memory, context,
                                                      rnn_state, annot,
                                                      annot_lens)
        annot_lens = T.ones(4).long()
        annot_lens[0] = 8
        rnn_state, context, alignment = layer.forward(memory, context,
                                                      rnn_state, annot,
                                                      annot_lens)
