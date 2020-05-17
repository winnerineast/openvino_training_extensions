# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

""" This module contains evaluation procedure. """

import Polygon as plg
import cv2
import numpy as np
import pycocotools.mask as mask_utils

IOU_CONSTRAINT = 0.5
AREA_PRECISION_CONSTRAINT = 0.5


def masks_to_rects(masks, is_rle):
    """ Creates masks from rects. """

    rects = []
    for mask in masks:
        decoded_mask = mask_utils.decode(mask) if is_rle else mask
        decoded_mask = np.ascontiguousarray(decoded_mask)
        contours, _ = cv2.findContours(decoded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        areas = []
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)

        if areas:
            i = np.argmax(areas)
            rects.append(boxes[i])

    return rects


def polygon_from_points(points):
    """ Returns a Polygon object to use with the Polygon2 class from a list of 8 points:
        x1,y1,x2,y2,x3,y3,x4,y4
    """

    point_mat = np.array(points[:]).astype(np.int32).reshape(-1, 2)
    return plg.Polygon(point_mat)


def draw_gt_polygons(image, gt_polygons, gt_dont_care_nums):
    """ Draws groundtruth polygons on image. """

    for point_idx, polygon in enumerate(gt_polygons):
        color = (128, 128, 128) if point_idx in gt_dont_care_nums else (255, 0, 0)
        for i in range(len(polygon[0])):
            pt1 = int(polygon[0][i][0]), int(polygon[0][i][1])
            pt2 = int(polygon[0][(i + 1) % len(polygon[0])][0]), int(polygon[0][(i + 1) % len(polygon[0])][1])
            cv2.line(image, pt1, pt2, color, 2)
    return image


def draw_pr_polygons(image, pr_polygons, pr_dont_care_nums, pr_matched_nums, pr_confidences_list,
                     pr_transcriptions=None):
    """ Draws predicted polygons on image. """

    for point_idx, _ in enumerate(pr_polygons):
        if pr_confidences_list[point_idx] > 0.25:
            polygon = pr_polygons[point_idx]
            color = (0, 0, 255)
            if point_idx in pr_dont_care_nums:
                color = (255, 255, 255)
            if point_idx in pr_matched_nums:
                color = (0, 255, 0)

                if pr_transcriptions:
                    pt1 = int(polygon[0][0][0]), int(polygon[0][0][1])
                    cv2.putText(image, pr_transcriptions[point_idx], pt1, cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)

            for i in range(4):
                pt1 = int(polygon[0][i][0]), int(polygon[0][i][1])
                pt2 = int(polygon[0][(i + 1) % 4][0]), int(polygon[0][(i + 1) % 4][1])
                cv2.line(image, pt1, pt2, color, 2)
    return image


def get_union(polygon1, polygon2):
    """ Returns area of union of two polygons. """

    return polygon1.area() + polygon2.area() - get_intersection(polygon1, polygon2)


def get_intersection_over_union(polygon1, polygon2):
    """ Returns intersection over union of two polygons. """

    union = get_union(polygon1, polygon2)
    return get_intersection(polygon1, polygon2) / union if union else 0.0


def get_intersection(polygon1, polygon2):
    """ Returns are of intersection of two polygons. """

    intersection = polygon1 & polygon2
    if len(intersection) == 0:
        return 0
    return intersection.area()


def compute_ap(conf_list, match_list, num_gt_care):
    """ Returns average precision metrics. """

    correct = 0
    average_precision = 0
    if conf_list:
        conf_list = np.array(conf_list)
        match_list = np.array(match_list)
        sorted_ind = np.argsort(-conf_list)
        match_list = match_list[sorted_ind]

        for idx, matched in enumerate(match_list):
            if matched:
                correct += 1
                average_precision += float(correct) / (idx + 1)

        if num_gt_care > 0:
            average_precision /= num_gt_care

    return average_precision


def strip(text):
    """ Removes special characters. """

    if text.lower().endswith("'s"):
        text = text[:-2]
    text = text.strip('-')
    for character in "'!?.:,*\"()·[]/":
        text = text.replace(character, ' ')
    text = text.strip()

    return text


def is_word(text):
    """ Checks whether input txt is a word. """

    text = strip(text)

    if ' ' in text:
        return False

    if len(text) < 3:
        return False

    forbidden_symbols = "×÷·"

    range1 = [ord(u'a'), ord(u'z')]
    range2 = [ord(u'A'), ord(u'Z')]
    range3 = [ord(u'À'), ord(u'ƿ')]
    range4 = [ord(u'Ǆ'), ord(u'ɿ')]
    range5 = [ord(u'Ά'), ord(u'Ͽ')]
    range6 = [ord(u'-'), ord(u'-')]

    for char in text:
        char_code = ord(char)
        if char in forbidden_symbols:
            return False

        if not (range1[0] <= char_code <= range1[1] or \
                range2[0] <= char_code <= range2[1] or \
                range3[0] <= char_code <= range3[1] or \
                range4[0] <= char_code <= range4[1] or \
                range5[0] <= char_code <= range5[1] or \
                range6[0] <= char_code <= range6[1]):
            return False

    return True


def parse_gt_objects(gt_annotation, use_transcription):
    """ Parses groundtruth objects from annotation. """

    gt_polygons_list = []
    gt_transcriptions = [] if use_transcription else None
    gt_dont_care_polygon_nums = []
    for gt_object in gt_annotation:
        polygon = polygon_from_points(gt_object['points'])
        gt_polygons_list.append(polygon)

        transcription = gt_object['transcription']

        if transcription == '###':
            gt_dont_care_polygon_nums.append(len(gt_polygons_list) - 1)
        elif use_transcription:
            if is_word(transcription):
                transcription = strip(gt_object['transcription'])
            else:
                gt_dont_care_polygon_nums.append(len(gt_polygons_list) - 1)

        if use_transcription:
            gt_transcriptions.append(transcription)

    return gt_polygons_list, gt_dont_care_polygon_nums, gt_transcriptions


def parse_pr_objects(pr_annotation, use_transcription):
    """ Parses predicted objects from annotation. """

    pr_polygons_list = []
    pr_confidences_list = []
    pr_transcriptions = [] if use_transcription else None
    for pr_object in pr_annotation:
        polygon = polygon_from_points(pr_object['points'])
        pr_polygons_list.append(polygon)
        pr_confidences_list.append(pr_object['confidence'])
        if use_transcription:
            pr_transcriptions.append(pr_object['transcription'])

    return pr_polygons_list, pr_confidences_list, pr_transcriptions


def match_dont_care_objects(gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list):
    """ Matches ignored objects. """

    pr_dont_care_polygon_nums = []

    if gt_dont_care_polygon_nums:
        for pr_polygon_idx, pr_polygon in enumerate(pr_polygons_list):
            for dont_care_polygon_num in gt_dont_care_polygon_nums:
                intersected_area = get_intersection(gt_polygons_list[dont_care_polygon_num],
                                                    pr_polygon)
                pd_dimensions = pr_polygon.area()
                precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                if precision > AREA_PRECISION_CONSTRAINT:
                    pr_dont_care_polygon_nums.append(pr_polygon_idx)
                    break

    return pr_dont_care_polygon_nums


def match(gt_polygons_list, gt_transcriptions, gt_dont_care_polygon_nums, pr_polygons_list,
          pr_transcriptions, pr_dont_care_polygon_nums):
    """ Matches all objects. """

    pr_matched_nums = []
    pr_matched_but_not_recognized = []

    output_shape = [len(gt_polygons_list), len(pr_polygons_list)]
    iou_mat = np.empty(output_shape)
    gt_rect_mat = np.zeros(len(gt_polygons_list), np.int8)
    pr_rect_mat = np.zeros(len(pr_polygons_list), np.int8)
    for gt_idx, gt_polygon in enumerate(gt_polygons_list):
        for pr_idx, pr_polygon in enumerate(pr_polygons_list):
            iou_mat[gt_idx, pr_idx] = get_intersection_over_union(gt_polygon, pr_polygon)

    for gt_idx, _ in enumerate(gt_polygons_list):
        for pr_idx, _ in enumerate(pr_polygons_list):
            if gt_rect_mat[gt_idx] == 0 and pr_rect_mat[pr_idx] == 0 \
                    and gt_idx not in gt_dont_care_polygon_nums \
                    and pr_idx not in pr_dont_care_polygon_nums:
                if iou_mat[gt_idx, pr_idx] > IOU_CONSTRAINT:
                    gt_rect_mat[gt_idx] = 1
                    pr_rect_mat[pr_idx] = 1
                    if gt_transcriptions is not None and pr_transcriptions is not None:
                        if gt_transcriptions[gt_idx].lower() == pr_transcriptions[pr_idx].lower():
                            pr_matched_nums.append(pr_idx)
                        else:
                            print(gt_transcriptions[gt_idx], pr_transcriptions[pr_idx])
                            pr_matched_but_not_recognized.append(pr_idx)
                    else:
                        pr_matched_nums.append(pr_idx)

    return pr_matched_nums, pr_matched_but_not_recognized


def text_eval(pr_annotations, gt_annotations, images=None, imshow_delay=1,
              use_transcriptions=False):
    """ Annotation format:
        {"image_path": [
                            {"points": [x1,y1,x2,y2,x3,y3,x4,y4],
                             "confidence": float,
                             "transcription", str}
                        ],
         "image_path": [points],

         ### - is a transcription of non-valid word.

    """

    assert len(pr_annotations) == len(gt_annotations), str(len(pr_annotations)) + ' ' + str(
        len(gt_annotations))

    matched_sum = 0
    num_global_care_gt = 0
    num_global_care_pr = 0

    arr_global_confidences = []
    arr_global_matches = []

    for frame_id, _ in enumerate(pr_annotations):
        gt_polygons_list, gt_dont_care_polygon_nums, gt_transcriptions = parse_gt_objects(
            gt_annotations[frame_id], use_transcriptions)
        pr_polygons_list, pr_confidences_list, pr_transcriptions = parse_pr_objects(
            pr_annotations[frame_id], use_transcriptions)

        pr_dont_care_polygon_nums = match_dont_care_objects(
            gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list)

        pr_matched_nums = []
        pr_matched_but_not_recognized = []
        if gt_polygons_list and pr_polygons_list:
            pr_matched_nums, pr_matched_but_not_recognized = match(gt_polygons_list,
                                                                   gt_transcriptions,
                                                                   gt_dont_care_polygon_nums,
                                                                   pr_polygons_list,
                                                                   pr_transcriptions,
                                                                   pr_dont_care_polygon_nums)

            matched_sum += len(pr_matched_nums)

            for pr_num in range(len(pr_polygons_list)):
                if pr_num not in pr_dont_care_polygon_nums:
                    # we exclude the don't care detections
                    matched = pr_num in pr_matched_nums
                    arr_global_confidences.append(pr_confidences_list[pr_num])
                    arr_global_matches.append(matched)

        num_global_care_gt += len(gt_polygons_list) - len(gt_dont_care_polygon_nums)
        num_global_care_pr += len(pr_polygons_list) - len(pr_dont_care_polygon_nums)

        if images:
            image = images[frame_id]
            if isinstance(image, str):
                image = cv2.imread(image)
            draw_gt_polygons(image, gt_polygons_list, gt_dont_care_polygon_nums)
            draw_pr_polygons(image, pr_polygons_list, pr_dont_care_polygon_nums,
                             pr_matched_nums, pr_confidences_list)
            if use_transcriptions:
                draw_pr_polygons(image, pr_polygons_list, pr_dont_care_polygon_nums,
                                 pr_matched_nums + pr_matched_but_not_recognized, pr_confidences_list,
                                 pr_transcriptions)
            image = cv2.resize(image, (640, 480))
            cv2.imshow('result', image)
            k = cv2.waitKey(imshow_delay)
            if k == 27:
                return -1, -1, -1, -1

    method_recall = 0 if num_global_care_gt == 0 else float(matched_sum) / num_global_care_gt
    method_precision = 0 if num_global_care_pr == 0 else float(matched_sum) / num_global_care_pr
    denominator = method_precision + method_recall
    method_hmean = 0 if denominator == 0 else 2.0 * method_precision * method_recall / denominator

    average_precision = compute_ap(arr_global_confidences, arr_global_matches, num_global_care_gt)

    return method_recall, method_precision, method_hmean, average_precision
