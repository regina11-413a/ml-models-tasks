import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

im = "C:/Users/regin/PycharmProjects/ml-models-tasks/samples/i.jpg"
results = model(im)
results.print()
results.save()