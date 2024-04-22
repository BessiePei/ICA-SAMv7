import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX

if __name__ == "__main__":
    mode = "all_predict"
    crop            = False
    count           = False
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval   = 100
    fps_image_path  = ""
    dir_origin_path = ""
    dir_save_path = ""
    heatmap_save_path = ""
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Can not read video. Please check your camera.")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "all_predict":
        import os
        import json

        from tqdm import tqdm

        reverse_json = {
            1, 4, 5, 6, 7, 8,
            11, 13, 14, 15, 17,
            18, 20, 23, 29, 31,
            36, 37, 38, 39, 41,
            42, 43, 46, 48, 50,
            52, 53, 60, 61, 62,
            65, 69, 71, 73, 74,
            77, 78, 80, 81, 84,
            86, 92, 94, 95, 102,
            103, 105, 106, 109, 110,
            111, 113, 114, 122, 124,
            125, 127, 129, 132, 134,
            137, 140, 141, 142, 144, }

        root_path = dir_origin_path

        i = 0
        for dir_name in os.listdir(root_path):
            i = i + 1
            output_data = {dir_name: []}
            subdir = os.path.join(root_path, dir_name)
            img_names = os.listdir(subdir)
            if i in reverse_json:
                print("!!!!reverse: "+dir_name)
                img_names = sorted(img_names, reverse=True)
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(subdir, img_name)
                    image = Image.open(image_path)
                    r_image = yolo.detect_image(image)

                    all_boxs, all_scores = yolo.return_Lists()
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    if not os.path.exists(os.path.join(dir_save_path, dir_name)):
                        os.makedirs(os.path.join(dir_save_path, dir_name))
                    r_image.save(os.path.join(os.path.join(dir_save_path, dir_name), img_name.replace(".jpg", ".png")), quality=95,
                                 subsampling=0)
                    print(all_boxs)

                    output_data[dir_name].append({
                        "box": all_boxs,
                        "score": all_scores
                    })
            if not os.path.exists(os.path.join(dir_save_path, "JSONs")):
                os.makedirs(os.path.join(dir_save_path, "JSONs"))
            output_path = os.path.join(os.path.join(dir_save_path, "JSONs"), dir_name+".json")
            with open(output_path, 'w') as json_file:
                json.dump(output_data, json_file, indent=2)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
