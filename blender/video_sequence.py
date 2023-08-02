import os
import shutil


class VideoSequence:

    def __init__(self,
                 base_dir,
                 beg_frame,
                 end_frame,
                 interval,
                 input_subdir='videos',
                 key_subdir='keys0',
                 tmp_subdir='tmp',
                 input_format='frame%04d.jpg',
                 key_format='%04d.jpg',
                 out_subdir_format='out_%d',
                 blending_out_subdir='blend',
                 output_format='%04d.jpg'):
        if (end_frame - beg_frame) % interval != 0:
            end_frame -= (end_frame - beg_frame) % interval

        self.__base_dir = base_dir
        self.__input_dir = os.path.join(base_dir, input_subdir)
        self.__key_dir = os.path.join(base_dir, key_subdir)
        self.__tmp_dir = os.path.join(base_dir, tmp_subdir)
        self.__input_format = input_format
        self.__blending_out_dir = os.path.join(base_dir, blending_out_subdir)
        self.__key_format = key_format
        self.__out_subdir_format = out_subdir_format
        self.__output_format = output_format
        self.__beg_frame = beg_frame
        self.__end_frame = end_frame
        self.__interval = interval
        self.__n_seq = (end_frame - beg_frame) // interval
        self.__make_out_dirs()
        os.makedirs(self.__tmp_dir, exist_ok=True)

    @property
    def beg_frame(self):
        return self.__beg_frame

    @property
    def end_frame(self):
        return self.__end_frame

    @property
    def n_seq(self):
        return self.__n_seq

    @property
    def interval(self):
        return self.__interval

    @property
    def blending_dir(self):
        return os.path.abspath(self.__blending_out_dir)

    def remove_out_and_tmp(self):
        for i in range(self.n_seq + 1):
            out_dir = self.__get_out_subdir(i)
            shutil.rmtree(out_dir)
        shutil.rmtree(self.__tmp_dir)

    def get_input_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            id_list = list(range(end_id, beg_id, -1))
        path_dir = [
            os.path.join(self.__input_dir, self.__input_format % id)
            for id in id_list
        ]
        return path_dir

    def get_output_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        out_subdir = self.__get_out_subdir(i)
        path_dir = [
            os.path.join(out_subdir, self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_temporal_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'temporal_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_edge_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'edge_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_pos_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'pos_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_flow_sequence(self, i, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id - 1))
            path_dir = [
                os.path.join(self.__tmp_dir, 'flow_f_%04d.npy' % id)
                for id in id_list
            ]
        else:
            id_list = list(range(end_id, beg_id + 1, -1))
            path_dir = [
                os.path.join(self.__tmp_dir, 'flow_b_%04d.npy' % id)
                for id in id_list
            ]

        return path_dir

    def get_input_img(self, i):
        return os.path.join(self.__input_dir, self.__input_format % i)

    def get_key_img(self, i):
        sequence_beg_id = self.get_sequence_beg_id(i)
        return os.path.join(self.__key_dir,
                            self.__key_format % sequence_beg_id)

    def get_blending_img(self, i):
        return os.path.join(self.__blending_out_dir, self.__output_format % i)

    def get_sequence_beg_id(self, i):
        return i * self.__interval + self.__beg_frame

    def __get_out_subdir(self, i):
        dir_id = self.get_sequence_beg_id(i)
        out_subdir = os.path.join(self.__base_dir,
                                  self.__out_subdir_format % dir_id)
        return out_subdir

    def __get_tmp_out_subdir(self, i):
        dir_id = self.get_sequence_beg_id(i)
        tmp_out_subdir = os.path.join(self.__tmp_dir,
                                      self.__out_subdir_format % dir_id)
        return tmp_out_subdir

    def __make_out_dirs(self):
        os.makedirs(self.__base_dir, exist_ok=True)
        os.makedirs(self.__blending_out_dir, exist_ok=True)
        for i in range(self.__n_seq + 1):
            out_subdir = self.__get_out_subdir(i)
            tmp_subdir = self.__get_tmp_out_subdir(i)
            os.makedirs(out_subdir, exist_ok=True)
            os.makedirs(tmp_subdir, exist_ok=True)
