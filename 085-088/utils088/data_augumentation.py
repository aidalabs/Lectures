# 3장 시맨틱 분할의 데이터 확장
# 주의: 어노테이션 이미지는 색상 팔레트 형식(인덱스 컬러 이미지)로 되어 있음.

# 패키지 import
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class Compose(object):
    """transform 인수에 저장된 변형을 순차적으로 실행하는 클래스
       대상 화상과 어노테이션 화상을 동시에 변환합니다. 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[폭][높이]
        height = img.size[1]  # img.size=[폭][높이]

        # 확대 비율을 임의로 설정
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[폭][높이]
        scaled_h = int(height * scale)  # img.size=[폭][높이]

        # 화상 리사이즈
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # 어노테이션 리사이즈
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 화상을 원래 크기로 잘라
        # 위치를 구한다
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # input_size보다 짧으면 padding을 수행한다
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(
                anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        # 회전 각도 결정
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 회전
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """50% 확률로 좌우 반전시키는 클래스"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """input_size 인수의 크기를 변형하는 클래스"""

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):

        # width = img.size[0]  # img.size=[폭][높이]
        # height = img.size[1]  # img.size=[폭][높이]

        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL 이미지를 Tensor로 변환. 크기는 최대 1로 규격화된다
        img = transforms.functional.to_tensor(img)

        # 색상 정보의 표준화
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # 어노테이션 화상을 Numpy로 변환
        anno_class_img = np.array(anno_class_img)  # [높이][폭]

        # 'ambigious'에는 255가 저장되어 있으므로, 배경(0)으로 설정해 놓는다
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # 어노테이션 화상을 Tensor로 변환
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img
