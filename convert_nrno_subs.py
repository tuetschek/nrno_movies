#!/usr/bin/env python3
# -"- coding: utf-8 -"-


import re
from argparse import ArgumentParser
from filtering.tokenize import Tokenizer
from filtering.filter import Filter
from futil.split import get_data_parts
from futil.io import write_lines
from futil.dialogue import create_turns
import numpy as np
import xml.etree.ElementTree as ET
import os.path
import datetime
import json
import logging
import sys


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter('%(asctime)-15s %(message)s')
logger.addHandler(handler)


class Movie(object):

    def __init__(self, subtitle_file):
        self.filename = subtitle_file
        self.xml_tree = ET.parse(subtitle_file)
        self.scene_bound_time = datetime.timedelta(seconds=4)
        self.dialogues = None
        self.num_turns = 0

    def _extract_dialogues(self):
        """Extract speakers and dialogue lines from the movie. Return the dialogues
        and the total number of turns."""

        dialogues = []
        cur_dialogue = []
        cur_statement = ''
        cur_speaker = ''
        prev_start_time = - self.scene_bound_time - datetime.timedelta(seconds=1)
        prev_end_time = - self.scene_bound_time - datetime.timedelta(seconds=1)
        turns = 0

        for seg in self.xml_tree.getroot():
            # extract information from the segment
            text = ' '.join([tok.text for tok in seg if tok.tag == 'w'])
            start_time = prev_start_time
            end_time = prev_end_time
            for time_spec in seg.findall('time'):
                time_val = self._parse_time(time_spec.attrib['value'])
                if not time_val:  # invalid expression, ignore it
                    continue
                if time_spec.attrib['id'].endswith('S'):  # start time
                    start_time = time_val
                elif time_spec.attrib['id'].endswith('E'):  # end time
                    end_time = time_val

            # ignore '-' at beginning of segment
            if text.startswith('- '):
                text = text[2:]

            # detect turn boundary in manual/automatic annotation
            turn_bound = (False if (seg.attrib.get('continued') or
                                    seg.attrib.get('turn', 1) < 0.5) else True)

            # starting a new scene
            if (turn_bound and (start_time - prev_end_time) > self.scene_bound_time):
                if cur_statement:
                    cur_dialogue.append((cur_speaker, cur_statement))
                if len(cur_dialogue) > 1:
                    dialogues.append(cur_dialogue)
                    turns += len(cur_dialogue)
                cur_dialogue = []
                cur_statement = ''

            # starting a new statement
            elif turn_bound and cur_statement:
                cur_dialogue.append((cur_speaker, cur_statement))
                cur_statement = ''

            # buffering the current values
            cur_statement += (' ' if cur_statement else '') + text
            cur_speaker = seg.attrib.get('speaker', '')
            prev_start_time = start_time
            prev_end_time = end_time

        # clearing the buffers at the end
        if cur_statement:
            cur_dialogue.append((cur_speaker, cur_statement))
        if len(cur_dialogue) > 1:
            dialogues.append(cur_dialogue)
            turns += len(cur_dialogue)

        self.dialogues = dialogues
        self.num_turns = turns

    def _parse_time(self, text):
        time_expr = re.match(r'([0-9]+):([0-9]+):([0-9]+)[,.]([0-9]+)', text)
        if not time_expr:  # invalid expression
            logger.warn('Invalid time indicator in %s: %s' % (self.filename, text))
            return None
        hrs, mins, secs, millis = [int(val) for val in time_expr.groups()]
        return datetime.timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=millis)

    def _remove_bracketed(self, text):
        """Remove bracketed stuff -- remarks, emotions etc."""
        text = re.sub(r'\([^\)]*\)', r'', text)
        text = re.sub(r'\[[^\]]*\]', r'', text)
        return text

    def _postprocess(self, do_tokenize=False, do_lowercase=False, do_filter=False):
        """Postprocess dialogues (remove bracketed stuff, tokenize, lowercase)."""

        tok = Tokenizer()
        filt = Filter('')
        for dialogue in self.dialogues:
            # we're changing the list so we need to use indexes here
            for turn_no in range(len(dialogue)):
                speaker, statement = dialogue[turn_no]

                speaker = self._remove_bracketed(speaker)
                statement = self._remove_bracketed(statement)
                if do_filter:
                    statement = filt.filter_sentence(statement)
                if do_tokenize:
                    statement = tok.tokenize(statement)
                if do_lowercase:
                    statement = statement.lower()
                dialogue[turn_no] = (speaker, statement)  # assign new values

            # remove all turns that have been rendered empty by the postprocessing
            dialogue[:] = [(speaker_, statement_) for speaker_, statement_ in dialogue
                           if statement is not None and statement.strip()]

    def get_dialogues(self, do_tokenize=False, do_lowercase=False, do_filter=False):
        """Load and process one movie file, return the dialogues."""
        # actually extract the dialogues
        self._extract_dialogues()

        # remove brackets, tokenize, lowercase
        self._postprocess(do_tokenize, do_lowercase, do_filter)


def process_all(args):

    dialogues = []
    year_from, year_to = 0, 100000
    if args.years_only:
        year_from, year_to = [int(year) for year in args.years_only.split('-')]

    # read and process all movies
    for main_dir, subdirs, files in os.walk(args.movie_dir):

        if not files:
            continue

        # filter years (assume directories structured by year)
        year = re.search(r'\b[0-9]{4}\b', main_dir)
        if year:
            year = int(year.group(0))
            if year < year_from or year > year_to:
                continue

        movie_file = os.path.join(main_dir, files[0])  # just use the 1st file
        # load and try to identify the movie
        movie = Movie(movie_file)
        # extract dialogues from the movie
        movie.get_dialogues(args.tokenize, args.lowercase, args.filter is not None)
        if not movie.dialogues:
            logger.warn("Empty movie: %s." % movie_file)
            continue
        # lose speaker names, add the stuff to the whole file
        logger.info("Processed %s: %d dialogues / %d turns" %
                    (movie_file, len(movie.dialogues), movie.num_turns))
        for dialogue in movie.dialogues:
            dialogues.append([utt for _, utt in dialogue])

    if args.whole_dialogues:
        data = dialogues
    else:
        # create the actual output data
        data = create_turns(dialogues, args.history_length, args.exact_len_only)

    # filter the responses, if needed
    if args.filter:
        filtered = []
        filt = Filter(args.filter)  # use the actual filter settings
        for context, resp in data:
            resp = filt.filter_sentence(resp)
            if resp:
                filtered.append((context, resp))
        data = filtered
        logger.info('%d turns remain after filtering.' % len(data))

    if args.shuffle:
        logger.info('Shuffling...')
        np.random.seed(1234)
        np.random.shuffle(data)

    # split data into parts
    data_labels = args.split.split(':')
    data_sizes = [float(part) for part in args.split_ratio.split(':')]
    data_parts = get_data_parts(data, data_sizes)

    if not os.path.isdir(args.directory):
        logger.info("Directory %s not found, creating..." % args.directory)
        os.mkdir(args.directory)

    # write the parts out
    for data_part, label in zip(data_parts, data_labels):
        prefix = os.path.join(args.directory, label + '.' if label else '')
        if args.whole_dialogues:
            logger.info('Writing %s (size %d)' % (prefix + args.context_file, len(data_part)))
            write_lines(prefix + args.context_file, [json.dumps(d) for d in data_part])
        else:
            logger.info('Writing %s+%s (size: %d)' %
                        (prefix + args.context_file, prefix + args.response_file, len(data_part)))
            write_lines(prefix + args.context_file, [c for c, _ in data_part])
            write_lines(prefix + args.response_file, [r for _, r in data_part])


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-H', '--history-length', type=int, default=0,
                    help='History length to prepend before each utterance')
    ap.add_argument('-e', '--exact-len-only', action='store_true',
                    help='Only use beginnings of dialogues in the length of history + 1')
    ap.add_argument('-l', '--lowercase', action='store_true', help='Lowercase all outputs?')
    ap.add_argument('-t', '--tokenize', action='store_true', help='Tokenize all outputs?')
    ap.add_argument('-f', '--filter', type=str, default=None,
                    help='Filter parameters e.g., 3:20:P[OI] (no filter if unset,' +
                    'just filter non-English characters if set to empty).')
    ap.add_argument('-s', '--shuffle', action='store_true', help='Shuffle the lines?')
    ap.add_argument('-S', '--split', type=str, default='',
                    help='File name prefixes if splitting the set, e.g., "train:devel:test"')
    ap.add_argument('-y', '--years-only', type=str, help='Use only movies within these years ' +
                    '(YYYY-YYYY)')
    ap.add_argument('-r', '--split-ratio', type=str, default='1',
                    help='Numeric ratio for splitting the set')
    ap.add_argument('-d', '--directory', type=str, default='.',
                    help='Output directory (default: current)')
    ap.add_argument('-D', '--whole-dialogues', action='store_true',
                    help='Save the whole dialogues as JSON, not context-response (filtering won\'t work!)')
    ap.add_argument('movie_dir', type=str,
                    help='Input movie directory (will take 1st file out of each subdirectory)')
    ap.add_argument('context_file', type=str, help='Output context file (or whole dialogue file)')
    ap.add_argument('response_file', type=str, help='Output response file', nargs='?')

    args = ap.parse_args()
    process_all(args)
