import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        pianist_id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or pianist_id < min_id:
            min_id = pianist_id
        if max_id is None or pianist_id > max_id:
            max_id = pianist_id

    return min_id, max_id


def find_files(directory, pattern='*.wav'):
    """Recursively finds all files matching the pattern."""
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    """Generator that yields audio waveforms from the directory."""
    def randomize_files(fns):
        for _ in fns:
            file_index = random.randint(0, len(fns) - 1)
            yield fns[file_index]

    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        # normalize audio
        audio = librosa.util.normalize(audio)
        # trim the last 5 seconds to account for music rollout
        audio = audio[:-5*sample_rate]
        audio = np.reshape(audio, (-1, 1))
        yield audio, filename, category_id


def not_all_have_id(files):
    """ Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id."""
    id_reg_exp = re.compile(FILE_PATTERN)
    for f in files:
        ids = id_reg_exp.findall(f)
        if not ids:
            return True
    return False


# TODO: rewrite queue and enqueue with tf.train.
# TODO: and tf.train.input_producer
# https://www.tensorflow.org/api_guides/python/io_ops
class AudioReader(object):
    """Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue."""
    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=10240,
                 queue_size=32,
                 gc_enabled=None,
                 lc_enabled=None):
        """
        :param audio_dir: The directory containing WAV files
        :param coord: tf.train.Coordinator 
        :param sample_rate: Sample rate of the audio files
        :param sample_size: Number of timesteps of a cropped sample
        :param queue_size: Size of input pipeline
        :param gc_enabled: Is global conditioning enabled
        :param lc_enabled: Is local conditioning enabled
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.gc_enabled = gc_enabled
        self.lc_enabled = lc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        if self.lc_enabled:
            self.lc_placeholder = tf.placeholder(dtype=tf.float32,
                                                 shape=(None, 88))
            self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                                shapes=[(None, 88)])
            self.lc_enqueue = self.lc_queue.enqueue([self.lc_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def dequeue_lc(self, num_elements):
        return self.lc_queue.dequeue_many(num_elements)


    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, filename, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                crop = tf.random_crop(audio, [self.sample_size, audio.shape[1]])

                sess.run(self.enqueue, {self.sample_placeholder: crop})

                if self.gc_enabled:
                    sess.run(self.gc_enqueue, {self.id_placeholder: category_id})

                if self.lc_enabled:
                    # reshape piece into 1-D audio signal
                    audio = np.reshape(audio, (audio.shape[0],))
                    lc = self._midi_notes_encoding(audio)
                    sess.run(self.lc_enqueue, {self.lc_placeholder: lc})


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    @staticmethod
    def _midi_notes_encoding(audio):
        """
        compute frame-based midi encoding of audio
        :param audio: 1-D array of audio time series 
        """
        pitches, magnitudes = librosa.piptrack(audio)
        pitches = np.transpose(pitches)
        magnitudes = np.transpose(magnitudes)
        lc = np.zeros((pitches.shape[0], 88), dtype=np.float32)
        for i in range(pitches.shape[0]):
            # count non-zero entries of pitches
            nz_count = len(np.nonzero(pitches[i])[0])
            # keep a maximum of 6 detected pitches
            num_ind_to_keep = min(nz_count, 6)
            ind_of_largest_pitches = np.argpartition(
                magnitudes[i], -num_ind_to_keep)[-num_ind_to_keep:] \
                if num_ind_to_keep != 0 else []
            # convert the largest pitches to midi notes
            midi_notes = librosa.hz_to_midi(pitches[i, ind_of_largest_pitches]).round()
            # normalize magnitudes of pitches
            midi_mags = magnitudes[i, ind_of_largest_pitches] / \
                        np.linalg.norm(magnitudes[i, ind_of_largest_pitches], 1)
            np.put(lc[i], midi_notes.astype(np.int64) - [9], midi_mags)
        return lc
