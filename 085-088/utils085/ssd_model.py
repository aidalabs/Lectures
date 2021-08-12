"""
2장 SSD에서 구현한 내용을 정리한 파일
"""

# 패키지 import
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt

# XML 파일이나 텍스트를 읽고 가공 및 저장하는 라이브러리
import xml.etree.ElementTree as ET

# "utils" 폴더의 data_augumentation.py를 import. 입력 화상의 전처리를 수행하는 클래스
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

# "utils"의 match 함수를 기술한 match.py를 import
from utils.match import match


# 학습 및 검증용 화상 데이터, 어노테이션 데이터의 파일 경로 리스트를 작성
def make_datapath_list(rootpath):
    """
    데이터의 경로를 저장한 리스트를 작성한다.

    Parameters
    ----------
    rootpath : str
        데이터 폴더의 경로

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        데이터의 경로를 저장한 리스트
    """

    # 화상 파일과 어노테이션 파일의 경로 템플릿을 작성
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 훈련 및 검증 각각의 파일 ID(파일 이름)를 취득
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    # 훈련 데이터의 화상 파일과 어노테이션 파일의 경로 리스트를 작성
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 공백과 줄바꿈 제거
        img_path = (imgpath_template % file_id)  # 화상의 경로
        anno_path = (annopath_template % file_id)  # 어노테이션의 경로
        train_img_list.append(img_path)  # 리스트에 추가
        train_anno_list.append(anno_path)  # 리스트에 추가

    # 검증 데이터의 화상 파일과 어노테이션 파일의 경로 리스트 작성
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 공백과 줄바꿈 제거
        img_path = (imgpath_template % file_id)  # 화상의 경로
        anno_path = (annopath_template % file_id)  # 어노테이션의 경로
        val_img_list.append(img_path)  # 리스트에 추가
        val_anno_list.append(anno_path)  # 리스트에 추가

    return train_img_list, train_anno_list, val_img_list, val_anno_list


#　"XML 형식의 어노테이션"을 리스트 형식으로 변환하는 클래스
class Anno_xml2list(object):
    """
    한 장의 화상에 대한 "XML 형식의 어노테이션 데이터"를 화상 크기로 규격화해 리스트 형식으로 변환한다.

    Attributes
    ----------
    classes : 리스트
        VOC의 클래스명을 저장한 리스트
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        한 장의 화상에 대한 "XML 형식의 어노테이션 데이터"를 화상 크기로 규격화해 리스트 형식으로 변환한다.

        Parameters
        ----------
        xml_path : str
            xml 파일의 경로.
        width : int
            대상 화상의 폭.
        height : int
            대상 화상의 높이.

        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            물체의 어노테이션 데이터를 저장한 리스트. 화상에 존재하는 물체수만큼의 요소를 가진다.
        """

        # 화상 내 모든 물체의 어노테이션을 이 리스트에 저장합니다
        ret = []

        # xml 파일을 로드
        xml = ET.parse(xml_path).getroot()

        # 화상 내에 있는 물체(object)의 수만큼 반복
        for obj in xml.iter('object'):

            # 어노테이션에서 감지가 difficult로 설정된 것은 제외
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 한 물체의 어노테이션을 저장하는 리스트
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 물체 이름
            bbox = obj.find('bndbox')  # 바운딩 박스 정보

            # 어노테이션의 xmin, ymin, xmax, ymax를 취득하고, 0~1으로 규격화
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOC는 원점이 (1,1)이므로 1을 빼서 (0, 0)으로 한다
                cur_pixel = int(bbox.find(pt).text) - 1

                # 폭, 높이로 규격화
                if pt == 'xmin' or pt == 'xmax':  # x 방향의 경우 폭으로 나눈다
                    cur_pixel /= width
                else:  # y 방향의 경우 높이로 나눈다
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # 어노테이션의 클래스명 index를 취득하여 추가
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # res에 [xmin, ymin, xmax, ymax, label_ind]을 더한다
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# 입력 화상의 전처리를 수행하는 클래스
class DataTransform():
    """
    화상과 어노테이션의 전처리 클래스. 훈련과 추론에서 다르게 작동한다.
    화상의 크기를 300x300으로 한다.
    학습시 데이터 확장을 수행한다.

    Attributes
    ----------
    input_size : int
        리사이즈 대상 화상의 크기.
    color_mean : (B, G, R)
        각 색상 채널의 평균값.
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # int를 float32로 변환
                ToAbsoluteCoords(),  # 어노테이션 데이터의 규격화를 반환
                PhotometricDistort(),  # 화상의 색조 등을 임의로 변화시킴
                Expand(color_mean),  # 화상의 캔버스를 확대
                RandomSampleCrop(),  # 화상 내의 특정 부분을 무작위로 추출
                RandomMirror(),  # 화상을 반전시킨다
                ToPercentCoords(),  # 어노테이션 데이터를 0~1로 규격화
                Resize(input_size),  # 화상 크기를 input_size × input_size로 변형
                SubtractMeans(color_mean)  # BGR 색상의 평균값을 뺀다
            ]),
            'val': Compose([
                ConvertFromInts(),  # intをfloatに変換
                Resize(input_size),  # 화상 크기를 input_size × input_size로 변형
                SubtractMeans(color_mean)  # BGR 색상의 평균값을 뺀다
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드를 지정.
        """
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
    """
    VOC2012의 Dataset을 만드는 클래스. PyTorch의 Dataset 클래스를 상속받는다.

    Attributes
    ----------
    img_list : 리스트
        화상의 경로를 저장한 리스트
    anno_list : 리스트
        어노테이션의 경로를 저장한 리스트
    phase : 'train' or 'test'
        학습 또는 훈련을 설정한다.
    transform : object
        전처리 클래스의 인스턴스
    transform_anno : object
        xml 어노테이션을 리스트로 변환하는 인스턴스
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train 또는 val을 지정
        self.transform = transform  # 화상의 변형
        self.transform_anno = transform_anno  # 어노테이션 데이터를 xml에서 리스트로 변경

    def __len__(self):
        '''화상의 매수를 반환'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        전처리한 화상의 텐서 형식 데이터와 어노테이션을 취득
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        '''전처리한 화상의 텐서 형식 데이터, 어노테이션, 화상의 높이, 폭을 취득한다'''

        # 1. 화상 읽기
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [높이][폭][색BGR]
        height, width, channels = img.shape  # 화상의 크기 취득

        # 2. xml 형식의 어노테이션 정보를 리스트에 저장
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 전처리 실시
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        # 색상 채널의 순서가 BGR이므로, RGB로 순서를 변경
        # 또한 (높이, 폭, 색상 채널)의 순서를 (색상 채널, 높이, 폭)으로 변경
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # BBox와 라벨을 세트로 한 np.array를 작성, 변수 이름 "gt"는 ground truth의 약칭
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def od_collate_fn(batch):
    """
    Dataset에서 꺼내는 어노테이션 데이터의 크기가 화상마다 다릅니다.
    화상 내의 물체 수가 2개이면 (2, 5) 사이즈이지만, 3개이면 (3, 5) 등으로 변화합니다.
    이러한 변화에 대응하는 DataLoader을 작성하기 위한 collate_fn을 만듭니다.
    collate_fn은 PyTorch 리스트로 mini-batch를 작성하는 함수입니다.
    미니 배치 분량의 화상이 나열된 리스트 변수 batch에 미니 배치 번호를 지정하는 차원을 선두에 하나 추가하여 리스트의 형태를 변형합니다.
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0] 은 화상img입니다
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] 은 어노테이션 gt입니다

    # imgs는 미니 배치 크기의 리스트입니다
    # 리스트의 요소는 torch.Size([3, 300, 300]) 입니다
    # 이 리스트를 torch.Size([batch_num, 3, 300, 300])의 텐서로 변환합니다
    imgs = torch.stack(imgs, dim=0)

    # targets은 어노테이션의 정답인 gt의 리스트입니다
    # 리스트의 크기는 미니 배치의 크기가 됩니다
    # targets 리스트의 요소는 [n, 5] 로 되어 있습니다
    # n은 화상마다 다르며, 화상 속 물체의 수입니다
    # 5는 [xmin, ymin, xmax, ymax, class_index] 입니다

    return imgs, targets


# 34층에 걸쳐, vgg모듈을 작성
def make_vgg():
    layers = []
    in_channels = 3  # # 34층에 걸쳐, vgg모듈을 작성

    # vgg 모듈에서 사용하는 합성곱 층이나 맥스 풀링의 채널 수
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceil은 출력 크기를 계산 결과(float)에 대해, 소수점을 올려 정수로 하는 모드
            # 디폴트는 출력 크기를 계산 결과(float)에 대해, 소수점을 버려 정수로 하는 floor 모드
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


# 8층에 걸친 extras모듈 작성
def make_extras():
    layers = []
    in_channels = 1024  # vgg모듈에서 출력된, extra에 입력되는 화상 채널 수

    # vgg모듈에서 출력된, extra에 입력되는 화상 채널 수
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)


# 디폴트 박스의 오프셋을 출력하는 loc_layers,
# 디폴트 박스에 대한 각 클래스의 신뢰도 confidence를 출력하는 conf_layers를 작성
def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    # VGG의 22층, conv4_3(source1)에 대한 합성곱층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # VGG의 최종층(source2)에 대한 합성곱층
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source3)에 대한 합성곱층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source4)에 대한 합성곱층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source5)에 대한 합성곱층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source6)에 대한 합성곱층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# convC4_3로부터의 출력을 scale=20의 L2Norm으로 정규화하는 층
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  # 부모 클래스의 생성자 실행
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # 계수 weight의 초기값으로 설정할 값
        self.reset_parameters()  # 파라미터의 초기화
        self.eps = 1e-10

    def reset_parameters(self):
        '''結合パラメータを大きさscaleの値にする初期化を実行'''
        init.constant_(self.weight, self.scale)  # weight의 값이 모두 scale(=20)이 된다

    def forward(self, x):
        '''38×38의 특징량에 대해 512 채널에 걸쳐 제곱합의 루트를 구했다
        38×38개의 값을 사용하여 각 특징량을 정규화한 후 계수를 곱하여 계산하는 층'''

        # 각 채널에서의 38×38개의 특징량의 채널 방향의 제곱합을 계산하고,
        # 또한 루트를 구해 나누어 정규화한다
        # norm의 텐서 사이즈는 torch.Size([batch_num, 1, 38, 38])입니다
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        # 계수를 곱한다. 계수는 채널마다 하나로, 512개의 계수를 갖는다
        # self.weight의 텐서 사이즈는 torch.Size([512])이므로
        # torch.Size([batch_num, 512, 38, 38])까지 변형합니다
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


# 디폴트 박스를 출력하는 클래스
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 초기설정
        self.image_size = cfg['input_size']  # 화상 크기 300
        # [38, 19, …] 각 source의 특징량 맵의 크기
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])  # source의 개수 6
        self.steps = cfg['steps']  # [8, 16, …] DBox의 픽셀 크기

        self.min_sizes = cfg['min_sizes']
        # [30, 60, …] 작은 정사각형의 DBox 픽셀 크기(정확히는 면적)

        self.max_sizes = cfg['max_sizes']
        # [60, 111, …] 큰 정사각형의 DBox 픽셀 크기(정확히는 면적)

        self.aspect_ratios = cfg['aspect_ratios']  # 정사각형의 DBox의 화면비(종횡비)

    def make_dbox_list(self):
        '''DBox를 작성한다'''
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # f까지의 수로 두 쌍의 조합을 작성한다 f_P_2개
                # 특징량의 화상 크기
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # DBox의 중심 좌표 x,y. 0~1로 정규화되어 있음
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 화면비 1의 작은 DBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 화면비 1의 큰 DBox [cx,cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 그 외 화면비의 defBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # DBox를 텐서로 변환 torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # DBox가 화상 밖으로 돌출되는 것을 막기 위해, 크기를 최소 0, 최대 1로 한다
        output.clamp_(max=1, min=0)

        return output


# 오프셋 정보를 이용하여 DBox를 BBox로 변환하는 함수
def decode(loc, dbox_list):
    """
    오프셋 정보를 이용하여 DBox를 BBox로 변환한다.

    Parameters
    ----------
    loc:  [8732,4]
        SSD 모델로 추론하는 오프셋 정보.
    dbox_list: [8732,4]
        DBox 정보

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBox 정보
    """

    # DBox는 [cx, cy, width, height]로 저장되어 있음
    # loc도 [Δcx, Δcy, Δwidth, Δheight]로 저장되어 있음

    # 오프셋 정보로 BBox를 구한다
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    # boxes의 크기는 torch.Size([8732, 4])가 됩니다

    # BBox의 좌표정보를 [cx, cy, width, height]에서 [xmin, ymin, xmax, ymax]으로 변경
    boxes[:, :2] -= boxes[:, 2:] / 2  # 좌표 (xmin,ymin)로 변환
    boxes[:, 2:] += boxes[:, :2]  # 좌표 (xmax,ymax)로 변환

    return boxes

# Non-Maximum Suppression을 실시하는 함수
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppression을 실시하는 함수.
    boxes 중에서 겹치는(overlap 이상)의 BBox를 삭제한다.

    Parameters
    ----------
    boxes : [신뢰도 임계값(0.01)을 넘은 BBox 수,4]
        BBox 정보
    scores :[신뢰도 임계값(0.01)을 넘은 BBox 수]
        conf 정보

    Returns
    -------
    keep : 리스트
        conf의 내림차순으로 nms를 통과한 index가 저장됨
    count: int
        nms를 통과한 BBox 수
    """

    # return의 모형을 작성
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep: torch.Size([신뢰도 임계값을 넘은 BBox 수]), 요소는 전부 0

    # 각 BBox의 면적 area를 계산
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxes를 복사한다. 나중에 BBox 중복도(IOU) 계산시의 모형으로 준비
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socre를 오름차순으로 나열한다
    v, idx = scores.sort(0)

    # 상위 top_k개(200개)의 BBox의 index를 꺼낸다(200개 존재하지 않는 경우도 있음)
    idx = idx[-top_k:]

    # idx의 요소수가 0가 아닌 한 루프한다
    while idx.numel() > 0:
        i = idx[-1]  # conf의 최대 index를 i로

        # keep의 끝에 conf 최대 index를 저장
        # 이 index의 BBox와 크게 겹치는 BBox를 삭제
        keep[count] = i
        count += 1

        # 마지막 BBox인 경우 루프를 빠져나옴
        if idx.size(0) == 1:
            break

        # 현재 conf 최대의 index를 keep에 저장했으므로, idx를 하나 감소시킴
        idx = idx[:-1]

        # -------------------
        # 이제부터 keep에 저장한 BBox과 크게 겹치는 BBox를 추출하여 삭제한다
        # -------------------
        # 하나 감소시킨 idx까지의 BBox를, out으로 지정한 변수로 작성한다
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # 모든 BBox에 대해, 현재 BBox=index가 i로 겹치는 값까지로 설정(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # w와 h의 텐서 크기를 index를 하나 줄인 것으로 한다
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clamp한 상태에서 BBox의 폭과 높이를 구한다
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 폭이나 높이가 음수인 것은 0으로 한다
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clamp된 상태의 면적을 구한다
        inter = tmp_w*tmp_h

        # IoU = intersect부분 / (area(a) + area(b) - intersect부분)의 계산
        rem_areas = torch.index_select(area, 0, idx)  # 각 BBox의 원래 면적
        union = (rem_areas - inter) + area[i]  # 두 구역의 합(OR)의 면적
        IoU = inter/union

        # IoU가 overlap보다 작은 idx만 남긴다
        idx = idx[IoU.le(overlap)]  # le은 Less than or Equal to 처리를 하는 연산입니다
        # IoU가 overlap보다 큰 idx는 처음 선택한 keep에 저장한 idx과 동일한 물체에 대해 BBox를 둘러싸고 있으므로 삭제

    # while 루프에서 빠져나오면 종료

    return keep, count


# SSD 추론시에 conf와 loc의 출력에서 겹침(중복)을 제거한 BBox를 출력한다
class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # conf를 소프트맥스 함수로 정규화하기 위해 준비
        self.conf_thresh = conf_thresh  # conf가 conf_thresh=0.01보다 높은 DBox만을 취급
        self.top_k = top_k  # nm_supression으로 conf가 높은 top_k개의 계산에 사용하는, top_k = 200
        self.nms_thresh = nms_thresh  # nm_supression으로 IOU가 nms_thresh=0.45보다 크면 동일한 물체의 BBox로 간주

    def forward(self, loc_data, conf_data, dbox_list):
        """
        순전파 계산을 수행한다.

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            오프셋 정보
        conf_data: [batch_num, 8732,num_classes]
            감지 신뢰도
        dbox_list: [8732,4]
            DBox의 정보

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num, 클래스, conf의 top200, BBox 정보)
        """

        # 각 크기를 취득
        num_batch = loc_data.size(0)  # 미니 배치 크기
        num_dbox = loc_data.size(1)  # DBox 수 = 8732
        num_classes = conf_data.size(2)  # 클래스 수 = 21

        # conf는 소프트맥스를 적용하여 정규화한다
        conf_data = self.softmax(conf_data)

        # 출력 형식을 작성한다. 텐서 크기는 [minibatch수, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_data를 [batch_num,8732,num_classes]에서 [batch_num, num_classes,8732]에 순서 변경
        conf_preds = conf_data.transpose(2, 1)

        # 미니 배치마다 루프
        for i in range(num_batch):

            # 1. loc와 DBox로 수정한 BBox [xmin, ymin, xmax, ymax] 를 구한다
            decoded_boxes = decode(loc_data[i], dbox_list)

            # conf의 복사본을 작성
            conf_scores = conf_preds[i].clone()

            # 화상 클래스별 루프(배경 클래스의 index인 0은 계산하지 않고, index=1부터)
            for cl in range(1, num_classes):

                # 2.conf의 임계값을 넘은 BBox를 꺼낸다
                # conf의 임계값을 넘고 있는지에 대한 마스크를 작성하여,
                # 임계값을 넘은 conf의 인덱스를 c_mask로 취득
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gt는 Greater than을 의미. gt에 의해 임계값을 넘은 것이 1, 이하는 0이 된다.
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scores는 torch.Size([임계값을 넘은 BBox 수])
                scores = conf_scores[cl][c_mask]

                # 임계값을 넘은 conf가 없는 경우, 즉 scores=[]의 경우에는 아무것도 하지 않는다
                if scores.nelement() == 0:  # nelement로 요소수의 함계를 구한다
                    continue

                # c_mask를 decoded_boxes에 적용할 수 있도록 크기를 변경합니다
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_mask를 decoded_boxes로 적용합니다
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]로 1차원이 되어버리기 때문에,
                # view에서 (임계값을 넘은 BBox 수, 4) 크기로 변형한다

                # 3. Non-Maximum Suppression를 실시하여, 겹치는 BBox를 제거
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                # ids: conf의 내림차순으로 Non-Maximum Suppression를 통과한 index가 저장됨
                # count: Non-Maximum Suppression를 통과한 BBox 수

                # output에 Non-Maximum Suppression를 뺀 결과를 저장
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])

# SSD 클래스를 작성한다
class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference를 지정
        self.num_classes = cfg["num_classes"]  # 클래스 수=21

        # SD 네트워크를 만든다
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox 작성
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 추론시에는 "Detect" 클래스를 준비합니다
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = list()  # loc와 conf로의 입력 source1-6을 저장
        loc = list()  # loc의 출력을 저장
        conf = list()  # conf의 출력을 저장

        # vgg의 conv4_3까지 계산한다
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3의 출력을 L2Norm에 입력하고, source1을 작성하여, sources에 추가
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vgg를 끝까지 계산하여, source2를 작성하고, sources에 추가
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extras의 conv와 ReLU를 계산
        # source3~6을 sources에 추가
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # conv→ReLU→cov→ReLU를 하여 source에 넣는다
                sources.append(x)

        # source1~6에 각각 대응하는 합성곱을 1회씩 적용한다
        # zip으로 for 루프의 여러 리스트 요소를 취득
        # source1~6까지 있으므로 루프가 6회 실시됨
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permute으로 요소의 순서를 교체
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # l(x)와 c(x)으로 합성곱을 실행
            # l(x)와 c(x)의 출력 크기는 [batch_num, 4*화면비의 종류 수, featuremap 높이, featuremap 폭]
            # source에 따라 화면비의 종류 수가 다르며, 번거로우므로 순서 교체로 조정한다
            # permute로 요소 순서를 교체하여,
            # [minibatch 수, featuremap 수, featuremap 수, 4*화면비의 종류 수]으로
            # (주석)
            # torch.contiguous()은 메모리 상에 요소를 연속적으로 배치하는 명령입니다
            # 다음으로 view 함수를 사용합니다.
            # 이 view를 수행하기 때문에, 대상의 변수가 메모리 상에 연속적으로 배치되어 있어야 합니다.

        # 또한 loc와 conf의 형을 변형
        # loc의 크기는 torch.Size([batch_num, 34928])
        # conf의 크기는 torch.Size([batch_num, 183372])가 된다
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # 그리고 loc와 conf의 형을 변형
        # loc의 크기는 torch.Size([batch_num, 8732, 4])
        # conf의 크기는 torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 마지막으로 출력한다
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  # 추론시
            # "Detect" 클래스의 forward를 실행
            # 반환값의 크기는 torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])

        else:  # 학습시
            return output
            # 반환값은 (loc, conf, dbox_list)의 튜플


class MultiBoxLoss(nn.Module):
    """SSD의 손실함수 클래스입니다"""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 match 함수의 jaccard 계수의 임계치
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining의 음과 양의 비율
        self.device = device  # CPU와 GPU 중 어느 것으로 계산하는가

    def forward(self, predictions, targets):
        """
        손실 함수 계산

        Parameters
        ----------
        predictions : SSD net의 훈련시의 출력(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])

        targets : [num_batch, num_objs, 5]
            5는 정답의 어노테이션 정보[xmin, ymin, xmax, ymax, label_ind]를 나타낸다

        Returns
        -------
        loss_l : 텐서
            loc의 손실값
        loss_c : 텐서
            conf의 손실값
        """

        # SSD 모델의 출력이 튜플로 되어 있으므로 개별적으로 해체한다
        loc_data, conf_data, dbox_list = predictions

        # 요소 수 파악
        num_batch = loc_data.size(0)  # 미니배치 크기
        num_dbox = loc_data.size(1)  # DBox 수 = 8732
        num_classes = conf_data.size(2)  # 클래스 수 = 21

        # 손실 계산에 사용할 것을 저장하는 변수를 작성
        # conf_t_label: 각 DBox에 가장 가까운 정답 BBox의 라벨을 저장
        # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보를 저장
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # loc_t와 conf_t_label에,
        # DBox와 정답 어노테이션 targets를 match 시킨 결과를 덮어쓰기
        for idx in range(num_batch):  # 미니배치 루프

            # 현재 미니 뱃지의 정답 어노테이션의 BBox와 라벨을 취득
            truths = targets[idx][:, :-1].to(self.device)  # BBox
            # 라벨 [물체1의 라벨, 물체2의 라벨, …]
            labels = targets[idx][:, -1].to(self.device)

            # 디폴트 박스를 새로운 변수로 준비
            dbox = dbox_list.to(self.device)

            # match 함수를 실행하여 loc_t와 conf_t_label의 내용을 갱신한다
            # (상세)
            # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보가 덮어써진다
            # conf_t_label: 각 DBox에 가장 가까운 BBox의 라벨이 덮어써진다
            # 단, 가장 가까운 BBox와의 jaccard overlap이 0.5보다 작은 경우
            # 정답 BBox의 라벨 conf_t_label은 배경 클래스 0으로 한다
            variance = [0.1, 0.2]
            # 이 variance는 DBox에서 BBox로 보정 계산할 때 사용하는 식의 계수입니다
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)

        # ----------
        # 위치의 손실: loss_l을 계산
        # Smooth L1 함수로 손실을 계산한다. 단, 물체를 발견한 DBox의 오프셋만 계산한다
        # ----------
        # 물체를 감지한 BBox를 꺼내는 마스크를 작성
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # pos_mask를 loc_data 크기로 변형
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBox의 loc_data와 지도 데이터 loc_t를 취득
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 물체를 발견한 Positive DBox의 오프셋 정보 loc_t의 손실(오차)을 계산
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----------
        # 클래스 예측의 손실: loss_c를 계산
        # 교차 엔트로피 오차 함수로 손실을 계산한다. 단, 배경 클래스가 정답인 DBox가 압도적으로 많으므로,
        # Hard Negative Mining을 실시하여 물체 발견 DBox 및 배경 클래스 DBox의 비율이 1:3이 되도록 한다.
        # 배경 클래스 DBox로 예상한 것 중에서 손실이 적은 것은, 클래스 예측 손실에서 제외한다
        # ----------
        batch_conf = conf_data.view(-1, num_classes)

        # 클래스 예측의 손실 함수를 계산(reduction = 'none'으로 하여, 합을 취하지 않고, 차원을 보존한다)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

        # -----------------
        # 이제 Negative DBox 중에서 Hard Negative Mining으로 추출하는 것을 구하는 마스크를 작성합니다
        # -----------------

        # 물체 발견한 Positive DBox의 손실을 0으로 한다
        # (주의)물체는 label이 1 이상으로 되어 있다. 라벨 0은 배경을 의미.
        num_pos = pos_mask.long().sum(1, keepdim=True)  # 미니배치별 물체 클래스 예측의 수
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # 물체를 발견한 DBox는 손실 0으로 한다

        # Hard Negative Mining을 실시
        # 각 DBox 손실의 크기 loss_c의 순위 idx_rank을 구한다
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # (주석)
        # 구현된 코드는 특수하며 직관적이지 않습니다.
        # 위 두 줄의 요점은 각 DBox에 대해, 손실 크기가 몇 번째인지의 정보를
        # 변수 idx_rank에 고속으로 취득하는 코드입니다.
        
        # DBox의 손실 값이 큰 쪽에서 내림차순으로 정렬하여, DBox의 내림차순의 index를 loss_idx에 저장한다.
        # 손실 크기 loss_c의 순위 idx_rank를 구한다.
        # 여기서 내림차순이 된 배열 index인 loss_idx를 0부터 8732까지 오름차순으로 다시 정렬하기 위해서는,
        # 몇 번째 loss_idx의 인덱스를 취할지를 나타내는 것이 idx_rank이다.
        # 예를 들면, idx_rank 요소의 0번째 = idx_rank[0]를 구하려면 loss_idx의 값이 0인 요소,
        # 즉 loss_idx[?]=0의, ?는 몇 번째를 구할 것인지가 된다. 여기서 ? = idx_rank[0] 이다.
        # 지금 loss_idx[?]=0의 0은, 원래 loss_cの의 요소 0번째라는 의미이다.
        # 즉 ?은 원래 loss_c의 요소 0번째는, 내림차순으로 정렬된 loss_idx의 몇 번째입니까?
        # 를 구하는 것이 되어, 결과적으로
        # ? = idx_rank[0] 은 loss_c의 요소 0번째가 내림차순으로 몇 번째인지 나타낸다.

        # 배경 DBox의 수 num_neg를 구한다. HardNegative Mining에 의해,
        # 물체 발견 DBox의 수 num_pos의 3배(self.negpos_ratio 배)로 한다
        # 만에 하나, DBox의 수를 초과한 경우에는 DBox의 수를 상한으로 한다
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        # idx_rank는 각 DBox의 손실의 크기가 위에서 부터 몇 번째인지가 저장되어 있다
        # 배경 DBox의 수 num_neg보다 순위가 낮은(손실이 큰) DBox를 취하는 마스크를 작성
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # (종료) 이제부터 Negative DBox 중에서, Hard Negative Mining로 추출할 것을 구하는 마스크를 작성합니다
        # -----------------

        # 마스크의 모양을 성형하여 conf_data에 맞춘다
        # pos_idx_mask는 Positive DBox의 conf를 꺼내는 마스크입니다
        # neg_idx_mask는 Hard Negative Mining으로 추출한 Negative DBox의 conf를 꺼내는 마스크입니다
        # pos_mask: torch.Size([num_batch, 8732])→pos_idx_mask: torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_data에서 pos와 neg만 꺼내서 conf_hnm으로 한다. 형태는 torch.Size([num_pos+num_neg, 21])
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        # (주석) gt는 greater than(>)의 약칭. mask가 1인 index를 꺼낸다
        # pos_idx_mask+neg_idx_mask는 덧셈이지만, index로 mask를 정리할 뿐이다
        # 즉, pos이든 neg이든, 마스크가 1인 것을 더해 하나의 리스트로 만들어, 이를 gt로 취득

        # 마찬가지로 지도 데이터인 conf_t_label에서 pos와 neg만 꺼내어 conf_t_label_hnm으로
        # torch.Size([pos+neg]) 형태가 된다
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # confidence의 손실함수를 계산(요소의 합계=sum을 구한다)
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 물체를 발견한 BBox의 수 N(전체 미니 배치의 합계)으로 손실을 나눔
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
