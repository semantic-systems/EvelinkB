# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

from torch import nn


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(
            self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled

        self.bert_model = bert_model
        if add_linear:
            bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):

        # print("Sizes")
        # print(token_ids.size(), segment_ids.size(), attention_mask.size())
        # print(token_ids, segment_ids, attention_mask)

        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # print(output_bert.size())
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # output_bert = output_bert[0]
        # print(embeddings.size())
        # in case of dimensionality reduction
        # time.sleep(1000)
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result
