import torch
from torch import nn

from model.transformer_sublayers import PositionalEncoding, TransformerEncoder, TransformerDecoder, TransformerOutput

torch.random.manual_seed(0)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.input_embeddings = nn.Embedding(config['input_vocab_size'], config['emb_size'], padding_idx=0)
        self.output_embeddings = nn.Embedding(config['output_vocab_size'], config['emb_size'], padding_idx=0)
        self.pos_embeddings = PositionalEncoding(config['emb_size'], self.config['dropout'])
        self.norm = lambda x: x * torch.sqrt(1. / config['emb_size'])

        encoder_decoder_cfg = {
            'dropout': config['dropout'],
            'emb_size': config['emb_size'],
            'hidden_size': config['hidden_size'],
            'f_hidden_size': config['f_hidden_size'],
            'nrof_heads': config['nrof_heads']
        }
        self.encoders = nn.ModuleList([TransformerEncoder(**encoder_decoder_cfg) for _ in range(config['nrof_layers'])])
        self.decoders = nn.ModuleList([TransformerDecoder(**encoder_decoder_cfg) for _ in range(config['nrof_layers'])])
        self.output = TransformerOutput(emb_size=config['emb_size'], vocab_size=config['output_vocab_size'])

    def encode(self, input, input_mask):
        embed_input = self.pos_embeddings(self.input_embeddings(input))
        encoded_output = embed_input
        for encoder in self.encoders:
            encoded_output = encoder(encoded_output, input_mask)

        return encoded_output

    def decode(self, encoded_input, target, target_mask):
        embed_target = self.pos_embeddings(self.output_embeddings(target))
        decoded_output = embed_target
        for decoder in self.decoders:
            decoded_output = decoder(decoded_output, encoded_input, target_mask)

        return self.output(decoded_output)

    def forward(self, input_encoder, input_decoder, encoder_mask, decoder_mask):
        encoded_input = self.encode(input_encoder, encoder_mask)
        result = self.decode(encoded_input, input_decoder, decoder_mask)

        return result
