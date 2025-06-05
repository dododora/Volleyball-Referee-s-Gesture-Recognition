import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

all_visibilitys = []
def flat_landmarl(results):
  pose_row = list(
      np.array(
          [[landmark.x, landmark.y, landmark.z] for landmark in results.landmark]).flatten()
  )
  return pose_row



def pose_estimation(img, filename, name, label=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
    pose_row , hand_row = [], []
    all_visibility = []
    # 啟用姿勢偵測
    with mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2) as pose:
            results = pose.process(img_rgb)
            if results.pose_landmarks:                
                for landmark in results.pose_landmarks.landmark:
                    all_visibility.append(landmark.visibility)

                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # cv2.imwrite(f'./result/pose/mediapipe_{filename}', img)
                pose_row = flat_landmarl(results.pose_landmarks)
            else:
                all_visibility = [0] * 33

    # 啟用手勢偵測
    mp_hands = mp.solutions.hands
    hand_left, hand_right = [0] * 63, [0] * 63

    with mp_hands.Hands(min_detection_confidence=0.2,  min_tracking_confidence=0.2) as hands:

        
            # if not ret:
            #     print("Cannot receive frame")
            #     break
            # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = hands.process(img_rgb)                 # 偵測手掌
        if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks:
            #     # 將節點和骨架繪製到影像中
            #     mp_drawing.draw_landmarks(
            #         img,
            #         results.multi_hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 印出 handedness（左手/右手）與 confidence score
                handedness = results.multi_handedness[idx].classification[0]
                # label = handedness.label  # "Left" 或 "Right"
                score = handedness.score  # confidence (float)
                print(f"Handedness: {label}, Score: {score:.2f}")
                all_visibility.append(score)
                if handedness.label == 'Left':
                    hand_left = flat_landmarl(hand_landmarks)
                else:
                    hand_right = flat_landmarl(hand_landmarks)


            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,  # 這裡應該是一個單一的 `landmark_list`，而不是 list
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        # cv2.imshow('MediaPipe Hands', img)
                hand_row += flat_landmarl(hand_landmarks)
            
            # cv2.imwrite(f'./result/pose/{name}/mediapipe_{filename}', img)
    
    if len(all_visibility) < 33 +  2:
        all_visibility += [0] * ((33 + 2) - len(all_visibility))
    print("all_visibility = ", all_visibility)
    print("len(all_visibility) = ", len(all_visibility))
    if len(pose_row) == 0:
        return 0, 0
    if len(hand_row) < 126:
        hand_row += [0] * (126 - len(hand_row))
    # print(len(pose_row), len(hand_row))
    result = pose_row + hand_left + hand_right
    # result = pose_row + hand_row
    # max_right = 0
    # max_left = 0
    result.insert(0, label)
    if label == 'Left':
        print('wahhahahahaha')
    result.insert(1, filename)
    # 找各指尖與腕部的距離的最大值
    # for i in range(110, 161, 12):
    #     for j in range(0, 3, 1):
    #         if(abs(result[i+j]-result[100+j]) > max_right):
    #             max_right = abs(result[i+j]-result[100+j])
    #     i+=2
    # for i in range(173, 225, 12):
    #     for j in range(0, 3, 1):
    #         if(abs(result[i+j]-result[163+j]) > max_left):
    #             max_left = abs(result[i+j]-result[163+j])
    #     i+=2
    # print('max_right =', max_right)
    # 加入各指尖與腕部的距離
    tmp = []
    for i in range(110, 161, 12):
        for j in range(0, 3, 1):
            # tmp.append(((result[i+j]-result[100+j])/max_right))
            tmp.append(((result[i+j]-result[100+j])))
            # print(row[0], '[', i+j, ']', '[', 100+j, ']', result[i+j], result[100+j])
        i+=2
    for i in range(173, 225, 12):
        for j in range(0, 3, 1):
            # if max_left != 0:
            #     tmp.append(((result[i+j]-result[163+j])/max_left))
                tmp.append(((result[i+j]-result[163+j])))
            # else:
            #     tmp.append(0)
            # print(row[0], '[', i+j, ']', '[', 163+j, ']', result[i+j], result[163+j])
        i+=2
    # 加入中指指尖與無名指、小指的距離
    # for i in range(146, 161, 12):
    #     for j in range(0, 3, 1):
    #         tmp.append(result[i+j]-result[134+j])
    # for i in range(210, 225, 12):
    #     for j in range(0, 3, 1):
    #         tmp.append(result[i+j]-result[198+j])
    for i in range(0, len(tmp), 1):
        result.append(tmp[i])
    # result = [label, filename, (pose 33 * 3) + (hand 21 * 3) + (指尖與腕部距離 * 5 * 2 * 3)]
    # len result = 2 + (33 * 3) + (21 * 2 * 3) + (5 * 2 * 3) = 257
    return all_visibility, result


def normalize_pose(row):
    tmp_max = [-10000.0] * 3
    tmp_min = [10000.0] * 3
    length = len(row)
    for j in range(0, length, 3):
        for k in range(0, 3, 1):
            if(row[j+k] < tmp_min[k]):
                tmp_min[k] = row[j+k]
            if(row[j+k] > tmp_max[k]):
                tmp_max[k] = row[j+k]
    for k in range(0, length, 3):
        for l in range(0, 3, 1):
            if(row[k+l] == 0):
                continue
            row[k+l] = (row[k+l]-tmp_min[l])/(tmp_max[l]-tmp_min[l])
    return row

def detect_pose(name=None, img=None, save_csv=False):
    global all_visibilitys
    df_result = []
    if img is None: # 從資料夾讀取 csv (圖片檔名)
        df = pd.read_csv(f'./data/{name}/_classes.csv')
        
        for _, row in df.iterrows(): 
            if row['Unlabeled'] == 1:
                print('Unlabeled', row['filename'], '============================================')
                continue
            else:
                for col in df.columns:
                    if row[col] == 1:
                        label = col
                        print('label =', label)
                        label = label.replace(' ', '')
                        break
            image = f"./data/{name}/"+row['filename']
            image = cv2.imread(image)

            all_visibility, result = pose_estimation(image, row['filename'], name, label)
            if result == 0:
                print(f'{name} pose estimation error')
                # with open(f'./result/csv/error.csv', 'a') as f:
                #     f.write(f'{name},{row["filename"]}\n')
                continue
            
            df_result.append(result)
            all_visibilitys.append(all_visibility)
    else:
        all_visibility, result = pose_estimation(img, name, name, None)

        if result == 0:
            print(f'{name} pose estimation error')
            return 0, 0
        
        df_result.append(result)
        all_visibilitys = all_visibility
          
    # 標準化
    # max_body = 0
    # # print('row =', len(df_result),  df_result[0])  
    # for i in range(0, len(df_result), 1):
    #     df_result[i].append((df_result[i][34]-df_result[i][37])/2)
    #     df_result[i].append((df_result[i][35]-df_result[i][38])/2)
    #     df_result[i].append((df_result[i][36]-df_result[i][39])/2)
    #     for j in range(2, 76, 1):
    #         for k in range(0, 3, 1):
    #             if not isinstance(df_result[i][j], float):
    #                 df_result[i][j] = float(df_result[i][j])
    #             if not isinstance(df_result[i][257+k], float):
    #                 df_result[i][257+k] = float(df_result[i][257+k])
    #             df_result[i][j] = (df_result[i][j] - df_result[i][257+k])
    #             if(abs(df_result[i][j]) > max_body):
    #                 max_body = abs(df_result[i][j])
    # for i in range(0, len(df_result), 1):
    #     for j in range(2, 76, 1):
    #         df_result[i][j] = df_result[i][j] / max_body
    
    # 正規化
    for i in range(0, len(df_result), 1):
        # result = [label, filename, (pose 33 * 3) + (hand 21 * 3) + (指尖與腕部距離 * 5 * 2 * 3)]
        # len result = 2 + (33 * 3) + (21 * 2 * 3) + (5 * 2 * 3) = 257
        df_result[i][2:77] = normalize_pose(df_result[i][2:77])
        df_result[i][101:164] = normalize_pose(df_result[i][101:164])
        df_result[i][164:227] = normalize_pose(df_result[i][164:227])
        

    df_result = pd.DataFrame(df_result)
    if save_csv:
        df_result.to_csv(f'./result/csv/{name}_result.csv', index=False)
    print(f'{name} pose estimation done')
    
    return all_visibilitys, df_result.values.tolist()

def detect_post_vedio(name):
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
    cap = cv2.VideoCapture(name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 取得影片寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 取得影片高度
    fps = cap.get(cv2.CAP_PROP_FPS)                  # 取得影片的每秒幀數 (FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = "output.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("影片已播放完畢或無法讀取影片")
                break

            # 將 BGR 轉換成 RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 取得姿勢偵測結果
            results = pose.process(img_rgb)

            # 根據姿勢偵測結果，標記身體節點和骨架
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # 存 image

            # 將結果寫入輸出影片
            out.write(frame)
    cap.release()
    out.release()


def main():
    result = []
    result.extend(detect_pose(name='train', save_csv=True))
    result.extend(detect_pose(name='test', save_csv=True))
    result.extend(detect_pose(name='valid', save_csv=True))
    result = pd.DataFrame(result)
    # result.to_csv('./result/csv/all_visibility.csv', index=False)
    # detect_post_vedio('IMG_7227.MOV')
    

if __name__ == '__main__':
    main()