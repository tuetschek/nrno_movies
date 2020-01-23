#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import codecs
import re
import sys
from unidecode import unidecode
import ner


class Filter(object):

    def __init__(self, cfg=''):
        min_len, max_len, ne_types, _, _ = (cfg + '::::').split(':', 4)
        self.max_length = int(max_len) if max_len else -1
        self.min_length = int(min_len) if min_len else -1
        self.ban_punct_only = (min_len >= 0)
        if ne_types:
            self.ne_types = re.compile('^([' + ne_types + '].+)$')
            self.ner = ner.SocketNER(host='localhost', port=8080)
        else:
            self.ne_types = None

    def filter_sentence(self, sent):
        # normalize sentence
        sent = unidecode(sent)
        # remove URLs, HTML tags and entities, weird characters
        sent = re.sub(r'https? ?: ?/ ?/[^ ]*', '', sent)
        sent = re.sub(r'&(amp|lt|gt);', '', sent)
        sent = re.sub(r'< ?/? ?(strong|b|span|u|i|em|h[1-7]|li|ul|ol|div)(?: [^>]*)?>', '', sent)
        sent = re.sub(r'\[[^)]*\]', '', sent)  # delete all stuff in brackets
        sent = re.sub(r'\([^)]*\)', '', sent)  # delete all stuff in brackets
        sent = re.sub(r'[a-z.]*@[a-z.]*', '', sent)  # delete email adresses
        sent = re.sub(r'[^A-Za-z0-9\',;:!?.-]', ' ', sent)  # delete all but listed characters
        sent = re.sub(r' +', r' ', sent).strip()
        # sentence too long
        if self.max_length >= 0 and sent.count(' ') > self.max_length:  # TODO approximation
            return None
        # sentence too short
        if self.min_length >= 0 and sent.count(' ') < self.min_length - 1:
            return None
        # sentence only contains punctuation characters
        if self.ban_punct_only and re.match(r'^[ \',;:!?.-]*$', sent):
            return None
        # sentence contains NEs
        if self.ne_types:
            ents = self.ner.get_entities(sent)
            if self.ne_types.match(' '.join(list(ents.keys()))):
                return None
        return sent


if __name__ == '__main__':

    filter = Filter({'max_length': 20,
                     'ban_punct_only': True})

    stdin = codecs.getreader('UTF-8')(sys.stdin)
    stdout = codecs.getreader('UTF-8')(sys.stdout)

    for line in stdin:
        res = filter.filter([line])
        if res:
            print(res[0], "\n", file=stdout)
        else:
            print('<<REMOVED>>', "\n", file=stdout)
