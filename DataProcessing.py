import sys
import os
import numpy as np
from tqdm.auto import tqdm

class DataProcessing:

    def __init__(self, filepath_in, filepath_out, batch_size=50):
        """
        Constructor for DataProcessing class
        :param filepath_in: str, path to the initial data file of predefined format
        :param filepath_out:  str, path to the result file
        :param batch_size:  int, size of the batch to use while processing the data file
        """
        # Variables that are used for files processing
        self.filepath_in = filepath_in
        self.filepath_out = filepath_out
        self.batch_size = batch_size

        # additional variables used for z-normalization
        self._features_code = None
        self._features_dim = None
        self._batch_mean_x = []
        self._batch_mean_x2 = []
        self._batch_sizes = []
        self.means = None
        self.stds = None

    def process_file(self, ):
        """
        Main method of the class. Used to run the full preprocess of the file
        """
        self._preprocess_file()
        self._get_statistics()
        self._postprocess_file()

    @staticmethod
    def _process_line(line):
        """
        Method used to parse the line from initial line and return id_job, features_code, and list of features
        :param line: str, line from the file
        :return: id_job: int - job_id, features_code: int - features code, features: list - list of features
        """
        id_job, tmp = line.split('\t')
        id_job = int(id_job)

        tmp = tmp[:-1].split(',')
        features_code = int(tmp[0])
        features = tmp[1:]

        return id_job, features_code, features

    def _preprocess_batch(self, features):
        """
        Function used to calculate the batch statistics, which will be used for
        calculating general statistics of the whole file
        :param features: list - list of features
        """
        features = np.array(features, dtype=np.int)
        self._batch_mean_x.append(features.mean(axis=0))
        self._batch_mean_x2.append(np.power(features, 2).mean(axis=0))
        self._batch_sizes.append(features.shape[0])

    def _preprocess_file(self, ):
        """
        First path through the file to calculate required statistics used in normalization
        :return:
        """
        with open(self.filepath_in) as f:
            f.readline()  # to pass the header
            batch = []
            print('Starting Preprocessing...')
            for line in tqdm(f):
                _, features_code, features = self._process_line(line)
                batch.append(features)
                if len(batch) == self.batch_size:
                    self._preprocess_batch(batch)
                    batch = []
            self._preprocess_batch(batch)
            self._features_code = features_code
            self._features_dim = len(features)

    def _postprocess_batch(self, job_ids, features):
        """
        Function used to process the batch on the second run through the file.
        Uses results of preprocess for normalization and feature engineering
        :param job_ids: list, job_ids
        :param features: np.array, features to be processed
        :return: list of str - lines to be written to the output file.
        """
        features = np.array(features, dtype=np.int)
        max_feature_index = self._get_max_feature_index(features)
        max_feature_abs_mean_diff = self._get_max_feature_abs_mean_diff(features, max_feature_index)
        features = self._z_normalize(features)
        lines = self._create_lines_out(job_ids, features, max_feature_index, max_feature_abs_mean_diff)
        return lines

    def _postprocess_file(self, ):
        """
        Second path through the file to normalize features using precalculated statistics and do feature engineering
        The result is a new file.
        :return:
        """
        outF = open(self.filepath_out, "w")
        outF.writelines([self._create_header(self._features_code, self._features_dim)])  # writing the header
        with open(self.filepath_in) as f:
            f.readline()  # to pass the header
            job_ids, features = [], []
            print('Starting Postprocessing...')
            for line in tqdm(f):
                idd, _, feat = self._process_line(line)
                job_ids.append(idd)
                features.append(feat)
                if len(features) == self.batch_size:
                    lines = self._postprocess_batch(job_ids, features)
                    job_ids, features = [], []
                    outF.writelines(lines)
            lines = self._postprocess_batch(job_ids, features)
            outF.writelines(lines)
        outF.close()

    @staticmethod
    def _create_lines_out(job_ids, features, maxx, diff):
        """
        Function that convert result of the preprocess to the appropriate format for output.
        :param job_ids: list, job_ids
        :param features: np.array, processed features
        :param maxx: list, index of maximal item
        :param diff: list,
        :return:
        """
        lines = []
        for i, f, m, d in zip(job_ids, features.astype(str), maxx, diff):
            tmp = [str(i)] + list(f) + [str(m)] + [str(d)]
            lines.append('\t'.join(tmp) + '\n')
        return lines

    def _get_statistics(self, ):
        """
        Calculating general statistics of the whole file using intermediate batch results
        :return:
        """
        self.means = np.average(self._batch_mean_x, weights=self._batch_sizes, axis=0)
        means_x2 = np.average(self._batch_mean_x2, weights=self._batch_sizes, axis=0)
        self.stds = np.sqrt(means_x2 - np.power(self.means, 2))

    def _z_normalize(self, features):
        """
        Normalizing features using Z-normalization
        :param features: np.array of type int
        :return: np.array
        """
        return (features - self.means) / self.stds

    @staticmethod
    def _create_header(feature_code, feature_dim):
        """
        Function to create header for the output file
        :param feature_code: int, code of the feature
        :param feature_dim: number of features
        :return: str, line that is used as a header in output file
        """
        features_names = [f"feature_{feature_code}_stand_{i}" for i in range(feature_dim)]
        header = ['job_id'] + features_names \
                 + [f'max_feature_{feature_code}_index'] \
                 + [f'max_feature_{feature_code}_abs_mean_diff']
        header_line = '\t'.join(header) + '\n'
        return header_line

    def _get_max_feature_index(self, features):
        """
        Function to get max_feature_index given original features
        :param features:
        :return: list of int, indexes of max features
        """
        return features.argmax(axis=1)

    def _get_max_feature_abs_mean_diff(self, features, max_feature_indexes):
        """
        Function to get max_feature_abs_mean_diff given original features and max_feature_indexes
        :param features:
        :param max_feature_indexes:
        :return: list of double, indexes of max features
        """
        return np.abs(features.max(axis=1) - self.means[max_feature_indexes])


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or len(args) > 3:
        print(
            'Wrong number of arguments passed. Please pass input and output files (mandatory), and batch_size (optional)')
    else:
        path_in = args[0]
        path_out = args[1]
        batch_size = 50
        if len(args) == 3:
            try:
                batch_size = int(args[2])
            except:
                print('Error with batch_size parsing, using default value 50.')

        # check if parameters are correct
        if not os.path.exists(path_in):
            print('Input file does not exist.')
        else:
            pr = DataProcessing(path_in, path_out, batch_size)
            pr.process_file()



