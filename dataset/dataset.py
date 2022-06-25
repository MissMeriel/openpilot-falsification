import os, cv2
import numpy as np
import h5py, math
# Author: Meriel von Stein

class Dataset:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir

    # used by formatInput
    def _parse_image(self, frame):
        H = (frame.shape[0] * 2) // 3
        W = frame.shape[1]
        parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)
        parsed[0] = frame[0:H:2, 0::2]
        parsed[1] = frame[1:H:2, 0::2]
        parsed[2] = frame[0:H:2, 1::2]
        parsed[3] = frame[1:H:2, 1::2]
        parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
        parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
        return parsed

    def _deparse_image(self, frame):
        shape = (384, 512); H = 256; W = 512
        deparsed = np.zeros(shape)
        deparsed[0:H:2, 0::2] = frame[0]
        deparsed[1:H:2, 0::2] = frame[1]
        deparsed[0:H:2, 1::2] = frame[2]
        deparsed[1:H:2, 1::2] = frame[3]
        deparsed[H:H + H // 4] = frame[4].reshape((64, W))
        deparsed[H + H // 4:H + H // 2] = frame[5].reshape((64, W))
        # deparsed = np.array([frame[0, ] for i in range(H)]
        # a = frame[0][], frame[2][]
        return deparsed

    def formatInput(self, frame):
        img = cv2.resize(frame, (512, 256))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        parsed = self._parse_image(img_yuv)
        return parsed

    def permuteInput(self, img1, img2):
        parsed_images = np.array([img1, img2]).astype('float32')
        input_imgs = np.array(parsed_images).astype('float32')
        input_imgs.resize((1, 12, 128, 256))
        return input_imgs

    def load_dataset(self):
        dataset, img_pairs = [], []
        campath = os.path.join(self.parent_dir, "camera")
        h5files = [f for f in os.listdir(campath) if ".h5" in f]
        for f in h5files[1:2]:
            fullpath = os.path.join(campath, f)
            file = h5py.File(fullpath, 'r')
            dset = file['X']
            for i in range(3000, 20000, 20): #dset.shape[0]):
                if i % 2 == 0:
                    pass
                img1, img2 = dset[i], dset[i+1]
                img1 = np.transpose(img1, (1, 2, 0))
                img2 = np.transpose(img2, (1, 2, 0))
                img_pairs.append(tuple([img1, img2]))
                img1f = self.formatInput(img1)
                img2f = self.formatInput(img2)
                input_imgs = self.permuteInput(img1f, img2f)
                dataset.append(input_imgs)
        return dataset, img_pairs

    def deparseInput(self, imgs):
        imgs = np.resize(imgs, (2, 6, 128, 256))
        img1, img2 = np.split(imgs, 2, axis=0)
        img1 = self._deparse_image(img1[0])
        img2 = self._deparse_image(img2[0])
        img1_bgr = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_YUV2BGR_I420)
        img2_bgr = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_YUV2BGR_I420)
        return img1_bgr, img2_bgr

    # https://github.com/littlemountainman/modeld/blob/master/lane_visulaizer_dynamic.py
    def parseOutput(self, result):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        plans = {"plan1": result[0:991], "plan2": result[991:1982], "plan3": result[1982:2973],
                 "plan4": result[2973:3964], "plan5": result[3964:4955]}
        lanelines = {"y_pos": result[4955:5219], "z_pos": result[5219:5483]}
        laneline_prob = result[5483:5491]
        roadedges = {"y_pos": result[5491:5623], "z_pos": result[5623:5755]}
        leads = result[5755:5857]
        lead_prob = result[5857:5860]
        desire = result[5860:5868]
        meta = result[5868:5948]
        pose = result[5948:5960]
        recurrent = result[5960:6472]

        parsed_output = {'plans': {}, 'lanelines': {}, 'laneline_prob': [], 'road_edges': {}, 'leads': {},
                         'lead_prob': [], 'desire': {}, 'pose': {}, 'recurrent_state': {}}

        parsed_output['desire'] = desire
        parsed_output['meta'] = meta
        parsed_output['recurrent'] = recurrent

        parsed_output['plans'] = {}

        # get plan predictions
        d = {'x_pos': (0, 66), 'y_pos': (66, 132), 'z_pos': (132, 198)}
        for key in list(plans.keys()):
            parsed_output['plans'][key] = {}
            parsed_output['plans'][key] = {'means': {'x_pos': [], 'y_pos': [], 'z_pos': []},
                                           'stdevs': {'x_pos': [], 'y_pos': [], 'z_pos': []}}
        '''
        for plan in plans:
            for var in d:
                for i in range(d[var][0], d[var][1]):
                    val = plans[plan][i]
                    if i % 2 == 0:
                        parsed_output['plans'][plan]['means'][var].append(val)
                    else:
                        parsed_output['plans'][plan]['stdevs'][var].append(val)

            parsed_output['plans'][plan]['prob'] = sigmoid(plans[plan][-1])
        '''

        d = {'x_pos': (0, 2), 'y_pos': (2, 4), 'z_pos': (4, 6)}
        for plan in plans:
            for j in range(0, len(plans[plan])-1, 15*2):
                for var in d:
                    for i in range(d[var][0], d[var][1]):
                        val = plans[plan][i+j]
                        if (i) % 2 == 0:
                            parsed_output['plans'][plan]['means'][var].append(val)
                        else:
                            parsed_output['plans'][plan]['stdevs'][var].append(val)

            parsed_output['plans'][plan]['prob'] = sigmoid(plans[plan][-1])

        '''
        plan = result[0:4955]  # (N, 4955)
        plans = plan.reshape((5, 991))  # (N, 5, 991)
        plan_probs = plans[:, -1]  # (N, 5)
        plans = plans[:, :-1].reshape(5, 2, 33, 15)  # (N, 5, 2, 33, 15)
        best_plan_idx = np.argmax(plan_probs, axis=0)  # (N,)
        best_plan = plans[best_plan_idx, ...]  # (N, 2, 33, 15)
        '''

        # get laneline predictions
        d = {'far_left': (0, 66), 'near_left': (66, 132), 'near_right': (132, 198), 'far_right': (198, 264)}
        for v in d:
            parsed_output['lanelines'][v] = {}
            parsed_output['lanelines'][v] = {'means': {'y_pos': [], 'z_pos': []}, 'stdevs': {'y_pos': [], 'z_pos': []}}

        for dim in lanelines:
            for edge in d:
                for i in range(d[edge][0], d[edge][1]):
                    val = lanelines[dim][i]
                    if i % 2 == 0:
                        parsed_output['lanelines'][edge]['means'][dim].append(val)
                    else:
                        parsed_output['lanelines'][edge]['stdevs'][dim].append(val)

        # get laneline probabilities
        for i in range(len(laneline_prob)):
            if i % 2 == 1:
                parsed_output['laneline_prob'].append(sigmoid(laneline_prob[i]))

        for i in range(3):
            parsed_output['lead_prob'].append(sigmoid(lead_prob[i]))

        # get road edges
        d = {'left': (0, 66), 'right': (66, 132)}
        for v in d:
            parsed_output['road_edges'][v] = {}
            parsed_output['road_edges'][v] = {'means': {'y_pos': [], 'z_pos': []}, 'stdevs': {'y_pos': [], 'z_pos': []}}

        for dim in roadedges:
            for edge in d:
                for i in range(d[edge][0], d[edge][1]):
                    val = roadedges[dim][i]
                    if i % 2 == 0:
                        parsed_output['road_edges'][edge]['means'][dim].append(val)
                    else:
                        parsed_output['road_edges'][edge]['stdevs'][dim].append(val)

        # get leads
        d = {0:(0,24), 2:(24,48)}
        for v in d:
            parsed_output["leads"][v] = {"means":{}, "stdevs":{}}
        leads = np.split(leads, 2)
        for lead in leads:
            seconds = np.split(lead[:48], 6)
            sec_labels = [0, 2, 4, 6, 8, 10]
            for sl, s in zip(sec_labels, seconds):
                pass

        return parsed_output