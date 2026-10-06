import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers.video_utils import load_video
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
model.eval()

video_path = "C:/Users/regin/PycharmProjects/ml-models-tasks/samples/5408910_Coll_wavebreak_People_1280x720.mp4"

video, _ = load_video(video_path, num_frames=16, backend="opencv")
inputs = processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

top5 = torch.topk(probs, 5)
print("Топ-5 предсказанных классов:")
for score, idx in zip(top5.values, top5.indices):
    label = model.config.id2label[idx.item()]
    print(f"  {label}: {score.item():.4f}")