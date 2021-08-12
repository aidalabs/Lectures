"""
2장 SSD 예측 결과를 그림으로 그리는 클래스
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCV 라이브러리
import torch

from utils.ssd_model import DataTransform


class SSDPredictShow():
    """SSD의 예측과 화상의 표시를 한 번에 수행하는 클래스"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # 클래스명
        self.net = net  # SSD 네트워크

        color_mean = (104, 117, 123)  # (BGR) 색의 평균값
        input_size = 300  # 화상의 input 크기를 300×300으로 한다
        self.transform = DataTransform(input_size, color_mean)  # 전처리 클래스

    def show(self, image_file_path, data_confidence_level):
        """
        물체 감지의 예측 결과를 표시하는 함수.

        Parameters
        ----------
        image_file_path:  str
            화상의 파일 경로
        data_confidence_level: float
            예측에서 발견했다고 여기는 신뢰도의 임계치

        Returns
        -------
        없음. rgb_img에 물체 검출 결과가 더해진 화상이 표시된다.
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSD로 예측하는 함수

        Parameters
        ----------
        image_file_path:  str
            화상의 파일 경로

        dataconfidence_level: float
            예측에서 발견했다고 여기는 신뢰도의 임계치

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgb의 화상 데이터를 취득
        img = cv2.imread(image_file_path)  # [높이][폭][색BGR]
        height, width, channels = img.shape  # 화상 크기 취득
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 화상의 전처리
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # 어노테이션이 존재하지 않으므로 ""으로 한다.
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSD로 예측
        self.net.eval()  # 네트워크를 추론 모드로
        x = img.unsqueeze(0)  # 미니 배치화: torch.Size([1, 3, 300, 300])

        detections = self.net(x)
        # detections의 형은, torch.Size([1, 21, 200, 5])  ※200은 top_k의 값

        # confidence_level이 기준 이상인 것을 꺼낸다
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 조건 이상의 값을 추출
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 추출한 물체수만큼 루프를 돈다
            if (find_index[1][i]) > 0:  # 배경 클래스가 아닌 것
                sc = detections[i][0]  # 신뢰도
                bbox = detections[i][1:] * [width, height, width, height]
                # find_index는 미니 배치 수, 클래스, top의 tuple
                lable_ind = find_index[1][i]-1
                # (주석)
                # 배경 클래스는 0이므로 1을 뺀다

                # 반환값 리스트에 추가
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        물체 감지의 예측 결과를 화상으로 표시하는 함수.

        Parameters
        ----------
        rgb_img:rgb의 화상
            대상 화상 데이터
        bbox: list
            물체의 BBox 리스트
        label_index: list
            물체의 라벨 인덱스
        scores: list
            물체의 신뢰도
        label_names: list
            라벨명의 배열

        Returns
        -------
        없음. rgb_img에 물체 검출 결과가 더해진 화상이 표시된다.
        """

        # 테두리 색상 설정
        num_classes = len(label_names)  # 클래스 수(배경 제외)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 화상 표시
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox만큼 루프
        for i, bb in enumerate(bbox):

            # 라벨명
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # 클래스마다 다른 색깔의 테두리를 부여

            # 테두리에 붙이는 라벨 (예: person: 0.72)
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 테두리의 좌표
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 직사각형 그리기
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 직사각형의 테두리의 좌측 상단에 라벨을 그린다
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})
