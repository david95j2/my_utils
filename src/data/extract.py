import torch
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import os
from scipy.sparse.csgraph import connected_components
import argparse
import time


def run(
        video="joo_3.mp4",
        weights = "best.pt"
):

    #############################################
    if not os.path.exists(video): print("[Error] : Please check video file path ("+video+") !"); return 
    elif not os.path.exists(weights): print("[Error] : Please check weights file ("+weights+") !"); return   

    model = torch.hub.load('../yolov5', 'custom', path=weights, source='local')
    model.conf = 0.5
    save_folder = ["results/images/","results/bboxes/"]
    video_path = video
    video_file_name = os.path.basename(video_path)

    for i in range(len(save_folder)):
        if not os.path.exists(save_folder[i]+video_file_name.split(".")[0]):
            os.makedirs(save_folder[i]+video_file_name.split(".")[0])
        else:
            video_file_name = video_file_name.split(".")[0]+"_"+str(len(glob.glob(save_folder[i]+video_file_name.split(".")[0]+"*"))+1).zfill(2)
            os.makedirs(save_folder[i]+video_file_name.split(".")[0])

    ##############################################


    def box_area(corners: np.array) -> float:

        """
        Calculate the area of a box given the
        corners:

        Args:
        corners: float array of shape (N, 4)
            with the values [x1, y1, x2, y2] for
            each batch element.

        Returns:
        area: (N, 1) tensor of box areas for
            all boxes in the batch
        """

        x1 = corners[..., 0]
        y1 = corners[..., 1]
        x2 = corners[..., 2]
        y2 = corners[..., 3]

        return (x2 - x1) * (y2 - y1)


    def box_iou(box1: np.array, box2: np.array) -> np.array:

        """
        Calculate the intersection over union for two
        tensors of bounding boxes.

        Args:

        box1, box2: arrays of shape (N, 4)
            with the values [x1, y1, x2, y2] for
            each batch element.

        Returns:
        iou: array of shape (N, 1) giving
            the intersection over union of boxes between
            box1 and box2.   
        """

        x1 = np.max([box1[0], box2[0]])
        y1 = np.max([box1[1], box2[1]])
        x2 = np.min([box1[2], box2[2]])
        y2 = np.min([box1[3], box2[3]])

        intersection_box = np.stack([x1, y1, x2, y2], axis=-1)
        intersection_area = box_area(intersection_box)

        box1_area = box_area(box1)
        box2_area = box_area(box2)
        union_area = (box1_area + box2_area) - intersection_area

        # If x1 is greater than x2 or y1 is greater than y2
        # then there is no overlap in the bounding boxes.
        # Find the indices where there is a valid overlap.
        valid = np.logical_and(x1 <= x2, y1 <= y2)

        # For the valid overlapping boxes, calculate the intersection
        # over union. For the invalid overlaps, set the value to 0.  
        iou = np.where(valid, (intersection_area / union_area), 0)    
        check = 1 if iou > 0 else 0
        return check

    def extract_adj_matrix(box_list: np.array) -> np.array:

        total_size = box_list.shape[0]
        box_coord = box_list[:,1:5]
        adj_matrix = np.zeros((total_size, total_size))

        for i in range(total_size):
            for j in range(i+1, total_size):
                adj_matrix[i, j] = box_iou(box_coord[i], box_coord[j])
        adj_matrix += adj_matrix.T

        return adj_matrix

    def box_extract(box_list: np.array) -> np.array:

        """
        Args:
        box_list: arrays of shape (N, 6)
            with the values [class ,x1, y1, x2, y2, confidence_score] 
            for each batch element.

        Returns:
        total_mixed_box : arrays of shape (M, 6) where (M <= N) is components_N
            with the overlapped area of the components
        """

        total_mixed_box = []
        classes = np.unique(box_list[:,0])

        for cls in classes:
            adj_matrix = extract_adj_matrix(box_list)
            components_N, components_indices = connected_components(adj_matrix)
            mixed_box = np.zeros((components_N, box_list.shape[1])) # (N, 6)
            for index in range(components_N): # (M, 6)

                slices = np.where(components_indices==index, True, False)

                mixed_box[index, 0] = cls
                mixed_box[index, 1] = np.min(box_list[slices, 1])
                mixed_box[index, 2] = np.min(box_list[slices, 2])
                mixed_box[index, 3] = np.max(box_list[slices, 3])
                mixed_box[index, 4] = np.max(box_list[slices, 4])
                mixed_box[index, 5] = np.mean(box_list[slices, 5])

            total_mixed_box.append(mixed_box)
        total_mixed_box = np.concatenate(total_mixed_box)

        return total_mixed_box

    def get_final_label(img,output_folder,filename,grid_size=(4,6)):
        

        """
        Args:
        img : img detection
        foldername : make folder
        grid_size : how to divide the image as grid pathces

        Returns:
        text file : result/bboxes/exp/..txt
        """

        output_folder = "results/bboxes/" + output_folder + "/"

        # load image info             
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_copy = img.copy()
        imgheight=img.shape[0] 
        imgwidth=img.shape[1] 
        

        if imgheight % grid_size[0] !=0 or imgwidth % grid_size[1] != 0:
            print("please check original resolutions (ex : 3840,2160)")
            
        H = imgheight // grid_size[0] # height
        W = imgwidth // grid_size[1] # width

        x1 = 0
        y1 = 0
        img_list = []

        df = pd.DataFrame([], columns=["i","j","class","xmin","ymin","xmax","ymax","confidence"])

        # grid
        for i, y in enumerate(range(0, imgheight, H)):
            for j, x in enumerate(range(0, imgwidth, W)):
                if (imgheight - y) < H or (imgwidth - x) < W:
                    break
                    
                y1 = y + H
                x1 = x + W

                # check whether the patch width or height exceeds the image width or height
                if x1 >= imgwidth and y1 >= imgheight:
                    x1 = imgwidth - 1
                    y1 = imgheight - 1
                    tiles = image_copy[y:y+H+50, x:x+W]
                    img_list.append(tiles)

                elif y1 >= imgheight: # when patch height exceeds the image height
                    y1 = imgheight - 1
                    tiles = image_copy[y:y+H, x:x+W+50]
                    img_list.append(tiles)

                elif x1 >= imgwidth: # when patch width exceeds the image width
                    x1 = imgwidth - 1
                    tiles = image_copy[y:y+H+50, x:x+W]
                    img_list.append(tiles)

                else:
                    tiles = image_copy[y:y+H+50, x:x+W+50]
                    img_list.append(tiles)

        # inference      
        result = model(img_list)

        # result processing
        for k in range(0,grid_size[0]*grid_size[1]):
            
            if len(result.pandas().xyxy[k]):
                i = k // grid_size[1]
                j = k % grid_size[1]
                pred = result.pandas().xyxy[k][['class', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence']]
                pred['i'] = i
                pred['j'] = j

                df = pd.concat([df, pred])


        df['xmin'] = (df['xmin'] + df['j'] * W).astype(int)
        df['xmax'] = (df['xmax'] + df['j'] * W).astype(int)
        df['ymin'] = (df['ymin'] + df['i'] * H).astype(int)
        df['ymax'] = (df['ymax'] + df['i'] * H).astype(int)
        df['confidence'] = df['confidence'].astype(np.float64).round(2)

        np.savetxt(fname=output_folder+"/"+filename+".txt", 
        X=box_extract(df[['class','xmin','ymin','xmax','ymax','confidence']].values),
        fmt='%d %d %d %d %d %.2f')

    class framecut:
        def __init__(self, video_path = ""):
            self.video_path = video_path
            self.video_file_name = os.path.basename(self.video_path).split(".")[0]
            self.vidcap = cv2.VideoCapture(self.video_path)
            self.save_folder = ["results/images/","results/bboxes/"]
            
            if not self.vidcap.isOpened():
                print("Could not Open : "+self.video_path)
                exit(0)

            self.length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
            
        def __call__(self,frame=2):
            cap = cv2.VideoCapture(self.video_path) # 동영상 캡쳐 객체 생성

            if cap.isOpened():                 # 캡쳐 객체 초기화 확인
                cnt = 0
                
                while True:
                    start = time.time()  # 시작 시간 저장
                    ret, img = cap.read()      # 다음 프레임 읽기      
                    
                    if ret:
                        cnt += 1
                        if cnt % (self.fps * frame) != 0: continue

                        # txt file 만들기
                        get_final_label(img,output_folder=self.video_file_name,filename = str(cnt).zfill(4))

                        # 이미지 저장            
                        cv2.imwrite(self.save_folder[0]+self.video_file_name+"/"+str(cnt).zfill(4)+".jpg",img) 
                        
                        print('[Message] : Extract %s frame Success! (%.2f)' % (str(cnt).zfill(4), time.time() - start))

                    else: break # 재생 완료
                print("[Message] : "+self.video_file_name+" video file extraction done !")
                
            else: print("can't open video.")      # 캡쳐 객체 초기화 실패
            cap.release()                       # 캡쳐 자원 반납

    a = framecut(video_path)
    a()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default="joo_3.mp4", help='video file path')
    parser.add_argument('--weights', type=str, default="best.pt", help='weight file path')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
