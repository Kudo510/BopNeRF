import os
import cv2, json
import numpy as np
import re
import torch


def extractRT(path, occid):
    head, tail = os.path.split(path)  ## head = NerfEmb\bop\tless\train\000001\rgb
    mainfolder, _ = os.path.split(head)   ## NerfEmb\bop\tless\train\000001\
    a = re.findall(r'\d+', tail) ##return 000001
    id = str(int(a[0])) ## 1
    transformdata = json.load(open(mainfolder + "/scene_gt.json"))
    R = transformdata[id][occid]["cam_R_m2c"]
    R = np.asarray(R).reshape(3, 3)
    T = np.asarray(transformdata[id][occid]["cam_t_m2c"])
    return R, T

def generate_bop_realsamples(datasetPath = "data/BOP/tless", objid="22", maskStr="mask", maxB=200, offset=10,
                                           synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
                                           fewids=[0], background=False, lmDir="train"):
    scene_id = 22
    frame_path = datasetPath + "/train_pbr" + "/0000" + str(scene_id).zfill(2) + "/"
    cam_params = json.load(open(frame_path + "scene_camera.json"))

    bboxDets = json.load(open(datasetPath + "/train_pbr/" + str(objid).zfill(6) + "/scene_gt_info.json"))

    frame_ids = torch.arange(len(bboxDets))
    frame_num = len(cam_params)  # number of all framee in the scene- for frame zB 22 in icbin/train_pbr - scene_num = 1000
    #import pdb;pdb.set_trace()   
    

    # lines=fewids
    # imCT=len(fewids)


    target_images = torch.zeros((frame_num, maxB, maxB, 3))
    target_silhouettes = torch.zeros((frame_num, maxB, maxB))

    RObj = np.zeros((frame_num, 3, 3))
    TObj = np.zeros((frame_num, 3))
    KObj = np.zeros((frame_num, 4, 4))

    for idx in range(frame_num):
        frame_id = int(frame_ids[idx])
        rgb = cv2.imread(frame_path + "rgb/" + str(frame_id).zfill(6) + ".png")
        # Do Nerf for first object - _000000.png
        mask = cv2.imread(frame_path + "mask/" + str(frame_id).zfill(6) + "_000000.png")

        rgb[np.where(mask == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(mask[:, :, 0]) # bbox position of the mask

        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw = int(w2 / 2)
        hh = int(h2 / 2)
        hd1 = int(np.max((w2, h2)) / 2)
        maxd = int(np.max((w2, h2)))

        squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8)
        if background:
            squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

        squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8)
        hs1 = int(squareRGB1.shape[0] / 2)
        squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2]
        squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
        if background:
            squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        # if justScaleImages:
        #         target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
        #         target_silhouettes[a11] = torch.from_numpy(
        #             cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        # else:
        #     try:
        target_images[a11] = torch.from_numpy(
            cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
        target_silhouettes[a11] = torch.from_numpy(
            cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
        if background:
            bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
            if bgScale % 2 == 0:
                bgScale = bgScale - 1
            target_backgrounds = target_backgrounds + [
                torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
        # except:
        #     print("a")
        # import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)
        camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh)
        # (centerY - offset-hd1)
        camparams = camparams * maxB / squareRGB1.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1

    if fewSamps:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines

    else:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj