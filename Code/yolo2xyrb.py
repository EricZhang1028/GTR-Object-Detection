import os
import cv2
import glob
import shutil

def yolobbox2bbox(x,y,w,h,img_size):
    x0, y0 = (x-w/2)*img_size[0], (y-h/2)*img_size[1]
    x1, y1 = (x+w/2)*img_size[0], (y+h/2)*img_size[1]

    return round(x0), round(y0), round(x1), round(y1)

def main():
    root_dir = "datasets"
    des_dir = os.path.join(root_dir, "val_labels_xyrb")
    if os.path.exists(des_dir):
        shutil.rmtree(des_dir)
    os.mkdir(des_dir)
    
    src_dir = os.path.join(root_dir, "val_labels")
    img_dir = os.path.join(root_dir, "val")
    count = 0
    for src_file in glob.glob(os.path.join(src_dir, "*.txt")):
        name = src_file.split("/")[-1].replace(".txt", "")

        # get image size
        img = cv2.imread(img_dir + "/" + name + ".jpg")
        h, w, c = img.shape
        img_size = [float(w), float(h)]

        des_file = des_dir + "/" + name + ".txt"
        with open(src_file, "r") as src_f:
            with open(des_file, "w") as des_f:
                lines = src_f.readlines()
                for line in lines:
                    line = line.strip("\n").split(" ")
                    cls_id = line[0]
                    yolo_x = float(line[1])
                    yolo_y = float(line[2])
                    yolo_w = float(line[3])
                    yolo_h = float(line[4])
                    x0, y0, x1, y1 = yolobbox2bbox(yolo_x, yolo_y, yolo_w, yolo_h, img_size)
                    des_f.write(f"{cls_id} {x0} {y0} {x1} {y1}\n")
        count += 1

    print(f"convert val data from yolo xywh to xyrb, total = {count}")

if __name__ == '__main__':
    main()