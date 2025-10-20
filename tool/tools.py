import cv2
import xlwt


def update_center_points(data, dic_center_points):
    '''
    更新坐标
    '''
    for row in data:
        x1, y1, x2, y2, cls_name, conf, obj_id = row[:7]

        # 计算中心点坐标
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 更新字典
        if obj_id in dic_center_points:
            # 判断列表长度是否超过30
            if len(dic_center_points[obj_id]) >= 30:
                dic_center_points[obj_id].pop(0)
            dic_center_points[obj_id].append((center_x, center_y))
        else:
            dic_center_points[obj_id] = [(center_x, center_y)]

    return dic_center_points


def res2OCres(results):
    lst_res = []
    if results is None:
        return lst_res
    for res in results.tolist():
        box = res[:4]
        conf = res[-2]
        cls = res[-1]
        lst_res.append([cls, conf, box])

    return list(lst_res)


def result_info_format(result_info, box, score, cls_name):
    '''
        格式组合
    '''
    # 类别
    result_info['cls_name'] = cls_name
    # 置信度
    result_info['score'] = round(score, 2)
    result_info['label_xmin_v'] = int(box[0])
    result_info['label_ymin_v'] = int(box[1])
    result_info['label_xmax_v'] = int(box[2])
    result_info['label_ymax_v'] = int(box[3])

    return result_info


def format_data(results):
    '''
    整理模型的识别结果
    '''
    lst_results = []
    for i, r in enumerate(results):
        # r.show()
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        conf = r.boxes.conf.cpu().numpy().tolist()
        cls = r.boxes.cls.cpu().numpy().tolist()
        names = r.names
        # speed = r.speed
        # # 转成 s
        # consum_time = round((speed['preprocess'] + speed['inference'] + speed['postprocess']) / 1000, 3)
        for box, con, c in zip(boxes, conf, cls):
            lst_results.append([names[c], round(con, 2), box])
    return lst_results


def writexls(DATA, path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Data')
    for i, Data in enumerate(DATA):
        for j, data in enumerate(Data):
            ws.write(i, j, str(data))
    wb.save(path)


def writecsv(DATA, path):
    try:
        f = open(path, 'w', encoding='utf8')
        for data in DATA:
            f.write(','.join('%s' % dat for dat in data) + '\n')
        f.close()
    except Exception as e:
        print(e)


def resize_with_padding(image, target_width, target_height, padding_value):
    """
    填充原图片的四周
    """
    # 原始图像大小
    original_height, original_width = image.shape[:2]

    # 计算宽高比例
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 确定调整后的图像大小和填充大小
    if width_ratio < height_ratio:
        new_width = target_width
        new_height = int(original_height * width_ratio)
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left, right = 0, 0
    else:
        new_width = int(original_width * height_ratio)
        new_height = target_height
        left = (target_width - new_width) // 2
        right = target_width - new_width - left
        top, bottom = 0, 0

    # 调整图像大小并进行固定值填充
    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=padding_value)

    return padded_image


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class.
    This version improves color distinctiveness by using the golden ratio.
    """
    # 黄金比例
    golden_ratio_conjugate = 0.618033988749895
    # 初始色调（可以选择任意数值）
    hue = (label * golden_ratio_conjugate) % 1

    # 使用色调转换为RGB值
    def hsv_to_rgb(h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
        return int(r * 255), int(g * 255), int(b * 255)

    # 调整饱和度和亮度
    saturation = 0.7
    value = 0.95

    color = hsv_to_rgb(hue, saturation, value)
    return color


def draw_text_with_red_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2,
                                  color=(0, 0, 255)):
    # 获取文本的大小
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算矩形背景的大小和位置
    background_width = text_width + 10
    background_height = text_height + 10
    background_position = (position[0], position[1] - text_height + 5)

    # 在图像上绘制红色背景矩形
    cv2.rectangle(image, background_position,
                  (background_position[0] + background_width, background_position[1] + background_height),
                  color, cv2.FILLED)

    # 计算文本的居中位置
    text_x = background_position[0] + int((background_width - text_width) / 2)
    text_y = background_position[1] + int((background_height + text_height) / 2)

    # 在图像上绘制文本
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image


def draw_info(frame, results):
    for i, bbox in enumerate(results):
        cls_name = bbox[0]
        conf = bbox[1]
        box = bbox[2]

        color = compute_color_for_labels(i)
        # 彩框 yolov5
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        # cls_name
        frame = draw_text_with_red_background(frame, str(cls_name) + ' ' + str(conf), (int(box[0]), int(box[1])),
                                              font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1, color=color)
        # cv2.putText(frame, str(cls_name), (int(box[0]), int(box[1])), 0, 1, color, 2)

    return frame
