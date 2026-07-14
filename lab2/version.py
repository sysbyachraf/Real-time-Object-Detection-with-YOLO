import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_box(img, box, color, label):
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def box_area(box):
    x_min, y_min, x_max, y_max = box
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    return width * height

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

def union_area(box1, box2):
    return box_area(box1) + box_area(box2) - intersection_area(box1, box2)

def compute_iou(box1, box2):
    inter = intersection_area(box1, box2)
    union = union_area(box1, box2)
    if union == 0:
        return 0
    return inter / union
def evaluate_detection(gt, pred):
    iou = compute_iou(gt["box"], pred["box"])
    same_class = gt["class"] == pred["class"]

    if same_class and iou >= 0.5:
        return "Bonne détection"
    elif same_class and iou >= 0.3:
        return "Détection partielle"
    else:
        return "Mauvaise détection"

image = np.ones((400, 600, 3), dtype=np.uint8) * 255

ground_truth = {
    "box": (100, 100, 300, 250),
    "class": "car"
}
ground_truth2 = {
    "box": (130, 140, 330, 270),
    "class": "person"
}
prediction = {
    "box": (110, 100, 310, 260),
    "class": "car",
    "score": 0.89
}
prediction2 = {
    "box": (130, 140, 330, 280),
    "class": "car",
    "score": 0.91
}

iou = compute_iou(ground_truth["box"], prediction["box"])
iou2 = compute_iou(ground_truth2["box"], prediction2["box"])

print("Classe correcte :", ground_truth["class"] == prediction["class"])
print("IoU :", round(iou, 4))

print("Classe correcte (ex2) :", ground_truth2["class"] == prediction2["class"])
print("IoU (ex2) :", round(iou2, 4))

image_result = image.copy()
draw_box(image_result, ground_truth["box"], (0, 255, 0), "Ground Truth: car")
draw_box(image_result, (80, 80, 280, 230),   (255, 0, 0), "Pred1 score=0.95")
draw_box(image_result, (160, 130, 370, 290), (255, 0, 0), "Pred2 score=0.72")
draw_box(image_result, (300, 200, 500, 360), (255, 0, 0), "Pred3 score=0.45")

cv2.putText(image_result, f"IoU = {iou:.4f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
plt.title("Manipulation des bounding boxes et calcul de l'IoU")
plt.axis("off")
plt.show()

print(evaluate_detection(ground_truth, prediction))
print(evaluate_detection(ground_truth2, prediction2))

# Interprétation :
# IoU = 0.8507 → la boîte prédite couvre bien la boîte réelle.
# Classe correcte = True → la détection est valide et bonne.
