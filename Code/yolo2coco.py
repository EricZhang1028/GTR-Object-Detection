import json
import cv2
import os, glob
import imagesize
import shutil

root_dir = "datasets"

classes = ["car"]
coco_format = {
    "images": [{}], 
    "categories": [],
    "annotations": [{}]
}

def create_image_annotation(file_name: str, width: int, height: int, image_id: int):
    image_annotation = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation

def create_annotations(min_x, min_y, width, height, image_id, category_id, annotation_id):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        # "segmentation": [],
    }

    return annotation

def get_img_info_and_anno(n):
    annotations = []
    image_annotations = []

    image_id = 0
    annotation_id = 1
    for file_path in glob.glob(os.path.join(root_dir, n, "*.jpg")):
        file_name = file_path.split("/")[-1]
        file_id = file_name.replace(".jpg", "")
        
        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(file_name, w, h, image_id)
        image_annotations.append(image_annotation)

        annotation_path = os.path.join(root_dir, f"{n}_labels", f"{file_id}.txt")
        with open(annotation_path, "r") as label_file:
            label_lines = label_file.readlines()
        
        for label_line in label_lines:
            label_li = label_line.split()
            category_id = int(label_li[0]) + 1
            yolo_center_x, yolo_center_y = float(label_li[1]), float(label_li[2])
            yolo_w, yolo_h = float(label_li[3]), float(label_li[4])

            float_x_center = w * yolo_center_x
            float_y_center = h * yolo_center_y
            float_width = w * yolo_w
            float_height = h * yolo_h

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)
            # center_x, center_y = yolo_center_x * w, yolo_center_y * h
            # width, height = yolo_w * w, yolo_h * h

            # min_x = round(center_x - width / 2)
            # min_y = round(center_y - height / 2)
            # width, height = round(width), round(height)

            annotation = create_annotations(min_x, min_y, width, height, image_id, category_id, annotation_id)
            annotations.append(annotation)
            annotation_id += 1
        
        image_id += 1

    return image_annotations, annotations

def main():
    des_dir = os.path.join(root_dir, "annotations")
    if os.path.exists(des_dir):
        shutil.rmtree(des_dir)
    os.mkdir(des_dir)

    for n in ["train", "val"]:
        output_path = os.path.join(des_dir, f"{n}.json")
        
        (coco_format["images"], coco_format["annotations"]) = get_img_info_and_anno(n)
        # print(f"{n} data len = {len(coco_format['images'])}")

        for index, label in enumerate(classes):
            category = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(category)

        with open(output_path, "w") as out_file:
            json.dump(coco_format, out_file, indent=4)
        
        print(f"create {n} annotations from {len(coco_format['images'])} {n} data finished. Annotations path = {output_path}")

if __name__ == '__main__':
    main()