import torch
from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        if batch[0][self.data_names.index('image')] is not None:
            max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
            image_none = False
        else:
            image_none = True
        
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        max_question_length = max([len(data[self.data_names.index('question')]) for data in batch])

        ## Hie
        max_row_word_length = 0
        max_col_word_length = 0
        max_row_phrase_length = 0
        max_col_phrase_length = 0
        for data in batch:
            w = data[self.data_names.index('word')]
            ph = data[self.data_names.index('phrase')]
            if len(w) > max_row_word_length:
                max_row_word_length = len(w)
            if len(ph) > max_row_phrase_length:
                max_row_phrase_length = len(ph)
            for partial_w in w:
                if len(partial_w) > max_col_word_length:
                    max_col_word_length = len(partial_w)
            for partial_ph in ph:
                if len(partial_ph) > max_col_phrase_length:
                    max_col_phrase_length = len(partial_ph)
        # Hie

        for i, ibatch in enumerate(batch):
            out = {}

            if image_none:
                out['image'] = None
            else:
                image = ibatch[self.data_names.index('image')]
                out['image'] = clip_pad_images(image, max_shape, pad=0)

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            question = ibatch[self.data_names.index('question')]
            out['question'] = clip_pad_1d(question, max_question_length, pad=0)

            ## Hie
            word = ibatch[self.data_names.index('word')]
            out['word'] = irregular_pad(word, (max_row_word_length, max_col_word_length), pad = 0)

            word_weight = ibatch[self.data_names.index('word_weight')]
            out['word_weight'] = irregular_pad(word_weight, (max_row_word_length, max_col_word_length), pad = 0)
            
            phrase = ibatch[self.data_names.index('phrase')]
            out['phrase'] = irregular_pad(phrase, (max_row_phrase_length, max_col_phrase_length), pad = 0)

            phrase_weight = ibatch[self.data_names.index('phrase_weight')]
            out['phrase_weight'] = irregular_pad(phrase_weight, (max_row_phrase_length, max_col_phrase_length), pad = 0)
            ## Hie

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)
        
        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple

