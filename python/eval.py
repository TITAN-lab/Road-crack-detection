from __future__ import division
from ctypes import *
import math, cv2, os
import random
import xml.etree.ElementTree as ET
import pandas as pd 
import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def getGT(annpath):
    fullname = os.path.join(annpath)
    tree = ET.parse(fullname)
    bbox = [] 
    object = []; classes = []   
    for obj in tree.findall('object'):
        bndbox_tree = obj.find('bndbox')
        bbox.append([int(bndbox_tree.find(tag).text) 
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        
        classes.append(obj.find('name').text)
    return [bbox,classes]

def bb_intersection_over_union(cbox, gbox):
    xA = max(cbox[0], gbox[0])
    yA = max(cbox[1], gbox[1])
    xB = min(cbox[2], gbox[2])
    yB = min(cbox[3], gbox[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    cboxArea = (cbox[2] - cbox[0] + 1) * (cbox[3] - cbox[1] + 1)
    gboxArea = (gbox[2] - gbox[0] + 1) * (gbox[3] - gbox[1] + 1)
    iou = interArea / float(cboxArea + gboxArea - interArea)
    return iou


if __name__ == "__main__":
    all_cols = ['filename','class', 'xmin', 'ymin', 'xmax', 'ymax']
    df_ini = pd.DataFrame(columns=all_cols)

    net = load_net("../cfg/yolo-dpm.cfg", "../dpm-cfg/backup/yolo-dpm_final.weights", 0)
    meta = load_meta("../cfg/dpm.data")
    test_data_path = "../scripts/devkit/test_darknet.txt"
    ann_path = "../scripts/devkit/Annotations/"
    ann_files = os.listdir(ann_path)
    tp = 0;fp=0;fn=0
    tclasses = ['D00','D01','D10','D11','D20','D40','D43','D44']
    gclasses = ['D00','D01','D10','D11','D20','D40','D43','D44']
    outclasses = [1,2,3,4,5,6,7,8]
    tpc = [0,0,0,0,0,0,0,0]
    fpc = [0,0,0,0,0,0,0,0]
    fnc = [0,0,0,0,0,0,0,0]

    tp_d = 0
    fp_d = 0
    fn_d = 0
    
    with open(test_data_path) as f: 
        contents = f.readlines()
        for content in contents:
            cimg = content.strip()
            cann = os.path.basename(cimg).replace('.jpg','.xml')
            cann_path = ann_path + cann
            if os.path.isfile(cimg) and os.path.isfile(cann_path):
                df = pd.DataFrame(columns=all_cols)
                [gboxes,classes] = getGT(cann_path)
                r = detect(net, meta, cimg)
                frame = cv2.imread(cimg)

                all_det = []
                all_det.append(os.path.basename(cimg))
                for det in r:
                    [x,y,w,h] = det[2]
                    [xmin,ymin,xmax,ymax] = [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]
                    cnt2 = 0
                    cnt3 = 0
                    idx = tclasses.index(det[0])
                    class_pred = gclasses[idx]
                    outclas_pred = outclasses[idx]

                    df.loc[len(df)] = [os.path.basename(cimg), outclas_pred, xmin,ymin,xmax,ymax]
                    
                    for gbox,gclas in zip(gboxes, classes):
                        [gxmin,gymin,gxmax,gymax] = [gbox[1],gbox[0],gbox[3],gbox[2]]
                        frame = cv2.rectangle(frame, (gxmin,gymin),(gxmax,gymax), (0,255,0),5)
                        cv2.putText(frame,class_pred, (int((gxmin+gxmax)/2),int((gymin+gymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

                        iou=bb_intersection_over_union([gxmin,gymin,gxmax,gymax], [xmin,ymin,xmax,ymax])
                        if (iou >=0.5):
                            cnt3+=1
                            tp_d+=1

                        if (iou >=0.5) and (class_pred == gclas):
                            tp+=1
                            cnt2+=1
                            tpc[idx]+=1
                            tp_d+=1
                            # print (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2), det[0])
                            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
                            frame = cv2.circle(frame, (xmax,ymax),7, (0,0,255),-1)
                            cv2.putText(frame,class_pred, (int((xmin+xmax)/2),int((ymin+ymax+50)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                            # cv2.imwrite('results/images/tp/' + os.path.basename(cimg), frame)
                            
                            # print (gclasses[idx], gclas, outclas_pred)
                    if cnt2==0:
                        fp+=1
                        fpc[idx]+=1
                        # frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
                        # frame = cv2.circle(frame, (xmax,ymax),7, (0,0,255),-1)
                        # cv2.putText(frame,class_pred, (int((xmin+xmax)/2),int((ymin+ymax+50)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                        # cv2.imwrite('results/images/fp/' + os.path.basename(cimg), frame)
                        
                    if cnt3==0:
                        fp_d+=1
                        

                # df.to_csv("titan_submission.csv", mode='a+', header=False, index=False)
                for gbox in gboxes:
                    cnt=0;
                    [gxmin,gymin,gxmax,gymax] = [gbox[1],gbox[0],gbox[3],gbox[2]]
                    frame = cv2.rectangle(frame, (gxmin,gymin),(gxmax,gymax), (0,255,0),5)
                    cv2.putText(frame,class_pred, (int((gxmin+gxmax)/2),int((gymin+gymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                    for det in r:
                        [x,y,w,h] = det[2]
                        [xmin,ymin,xmax,ymax] = [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]
                        iou=bb_intersection_over_union([gxmin,gymin,gxmax,gymax], [xmin,ymin,xmax,ymax])
                        if (iou>=0.5):
                            cnt+=1
                    if (cnt==0):
                        fn+=1
                        fnc[idx]+=1
                        fn_d+=1
                        cv2.imwrite('results/images/fn/' + os.path.basename(cimg), frame)
            
            all = [tpc,fpc,fnc]
            dfx = pd.DataFrame( np.transpose(np.array(all)))
            all_d = [[tp_d],[fp_d],[fn_d]]
            dfy = pd.DataFrame(np.transpose(np.array(all_d)))
            if (fp+tp>0) and (fn+tp>0):
                p = tp/(tp+fp) 
                r = tp/(tp+fn)
                if (p+r)>0:
                    F1 = 2*((p*r)/(p+r))
                    print (F1)
                    print ('.................')

                cv2.imshow('image',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        dfx.to_csv('results/accuracy_class.csv',index=False, header=['tp','fp','fn'])
        dfy.to_csv('results/accuracy_detection.csv',index=False, header=['tp','fp','fn'])
        cv2.destroyAllWindows()  



    
