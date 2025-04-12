import torch
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)

pro = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


inp= pro(images=img, return_tensors="pt")
outp = m(**inp)


t_size = torch.tensor([img.size[::-1]]) 
res =pro.post_process_object_detection(outp, target_sizes=t_size, threshold=0.9)[0]


# n_detected = len(res["scores"])
# print(n_detected)

for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
    
    print(f"{m.config.id2label[l.item()]}: {round(s.item(), 3)}")
    print(f"  Box: {b.tolist()}")


def plot_results(image, results):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        xmin, ymin, xmax, ymax = box.detach().numpy()  
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
        
        ax.text(xmin, ymin, f"{m.config.id2label[label.item()]}: {round(score.item(),2)}", bbox=dict(facecolor='yellow', alpha=0.5))
        
    plt.axis("off")
    plt.show()



plot_results(img, res)
