from math import ceil
import numpy as np
from scipy.signal import decimate
from scipy.io import wavfile
from software_model.constants import DATA_FILES_LOCATION, DOWNSAMPLE_FACTOR, NUM_SAMPS_IN_CHUNK, SAVE_DOWNSAMPLED_FILES
from software_model.chunk import Chunk
import os


class NetworkDataPreprocessor:

    def __init__(self, file_list):
        self._file_list = file_list
        self._audio = self.read_wav_files()
        self.num_files = len(file_list)

    def read_wav_files(self):
        """This method reads in .wav sound files."""
        files_to_return = []

        try:
            files_to_return = list(
                np.array(wavfile.read(DATA_FILES_LOCATION + file_name + '_downsampled.wav')[1]) for file_name in self._file_list)

        except FileNotFoundError:
            # If the file isn't found, attempt to see if the file simply hasn't been downsampled yet and do so.
            for file_name in self._file_list:
                self.downsample(DATA_FILES_LOCATION + file_name + '.wav', DATA_FILES_LOCATION + file_name + '_downsampled.wav', DOWNSAMPLE_FACTOR)
            # Retry Reading the files again
            files_to_return = list(np.array(wavfile.read(DATA_FILES_LOCATION + file_name + '_downsampled.wav')[1]) for file_name in self._file_list)

        if SAVE_DOWNSAMPLED_FILES is False:
            for file_name in self._file_list:
                os.remove(DATA_FILES_LOCATION + file_name + '_downsampled.wav')
        return files_to_return

    def get_chunk(self, file_index, chunk_index):
        """Retrives a specified chunk given a file index and chunk index"""
        audio_file = self._audio[file_index]

        start = chunk_index * NUM_SAMPS_IN_CHUNK
        end = start + NUM_SAMPS_IN_CHUNK

        chunk_data = audio_file[start:end, :]

        if end >= audio_file.shape[0]:
            chunk_data = np.lib.pad(chunk_data, ((0, end - audio_file.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))

        return Chunk(chunk_data)

    def get_all_chunks_in_file(self, file_index=0):
        audio_file = self._audio[file_index]
        num_chunks_in_file = ceil(audio_file.shape[0] / NUM_SAMPS_IN_CHUNK)
        chunks = []
        for i in range(num_chunks_in_file):
            chunks.append(self.get_chunk(file_index, i))

        return chunks

    def downsample(self, infile, outfile, downsample_factor=DOWNSAMPLE_FACTOR):
        """Downsample an audio file"""
        rate, aud = wavfile.read(infile)
        daud = decimate(aud, int(downsample_factor), axis=0)
        int16_max = np.iinfo(np.dtype('int16')).max
        daudint = np.int16(daud / np.max(np.abs(daud)) * int16_max)
        wavfile.write(outfile, rate // downsample_factor, daudint)
