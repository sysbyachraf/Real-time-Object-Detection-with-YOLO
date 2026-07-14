import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.ones((400, 600, 3), dtype=np.uint8) * 255
plt.imshow(image)
plt.title("Image vide")
plt.axis("off")
plt.show()
ground_truth_box = (100, 100, 300, 250)

predicted_box = (150, 120, 320, 270)


def draw_box(img, box, color, label):
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

image_boxes = image.copy()

draw_box(image_boxes, ground_truth_box, (0, 255, 0), "Ground Truth")
draw_box(image_boxes, predicted_box, (255, 0, 0), "Prediction")

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_boxes, cv2.COLOR_BGR2RGB))
plt.title("Bounding Boxes")
plt.axis("off")
plt.show()

def box_area(box):
    x_min, y_min, x_max, y_max = box
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    return width * height

gt_area = box_area(ground_truth_box)
pred_area = box_area(predicted_box)

print("Aire Ground Truth :", gt_area)
print("Aire Prediction   :", pred_area)

def intersection_area(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    return inter_width * inter_height

inter_area = intersection_area(ground_truth_box, predicted_box)
print("Aire de l'intersection :", inter_area)

def union_area(box1, box2):
    area1 = box_area(box1)
    area2 = box_area(box2)
    inter = intersection_area(box1, box2)
    return area1 + area2 - inter

union = union_area(ground_truth_box, predicted_box)
print("Aire de l'union :", union)

def compute_iou(box1, box2):
    inter = intersection_area(box1, box2)
    union = union_area(box1, box2)
    if union == 0:
        return 0
    return inter / union

iou = compute_iou(ground_truth_box, predicted_box)
print("IoU :", round(iou, 4))

image_result = image.copy()

draw_box(image_result, ground_truth_box, (0, 255, 0), "Ground Truth")
draw_box(image_result, predicted_box, (255, 0, 0), "Prediction")

cv2.putText(image_result, f"IoU = {iou:.4f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
plt.title("Evaluation de la prediction")
plt.axis("off")
plt.show()

ground_truth = {
    "box": (100, 100, 300, 250),
    "class": "car"
}

prediction = {
    "box": (150, 120, 320, 270),
    "class": "car",
    "score": 0.89
}

if ground_truth["class"] == prediction["class"]:
    print("Classe correcte")
else:
    print("Classe incorrecte")

predictions = [
    {"box": (105, 105, 295, 245), "class": "car", "score": 0.95},
    {"box": (150, 120, 320, 270), "class": "car", "score": 0.89},
    {"box": (250, 200, 380, 320), "class": "car", "score": 0.72},
    {"box": (100, 100, 300, 250), "class": "truck", "score": 0.91}
]

for i, pred in enumerate(predictions, start=1):
    iou_value = compute_iou(ground_truth["box"], pred["box"])
    same_class = ground_truth["class"] == pred["class"]

    print(f"Prediction {i}")
    print("  Classe prédite :", pred["class"])
    print("  Score          :", pred["score"])
    print("  IoU            :", round(iou_value, 4))
    print("  Classe correcte:", same_class)
    print("-" * 40)