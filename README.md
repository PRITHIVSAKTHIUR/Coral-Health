![xbvsxdfgb.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/pun-5Yr4DKWrimbFX7BFE.png)

# **Coral-Health**

> **Coral-Health** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify coral reef images into two health conditions using the **SiglipForImageClassification** architecture.

```py
Classification Report:
                 precision    recall  f1-score   support

Bleached Corals     0.8677    0.7561    0.8081      4850
 Healthy Corals     0.7665    0.8742    0.8168      4442

       accuracy                         0.8125      9292
      macro avg     0.8171    0.8151    0.8124      9292
   weighted avg     0.8193    0.8125    0.8122      9292
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/KMlOnMf0JTq1-5_7qGhjL.png)

The model categorizes images into two classes:

- **Class 0:** Bleached Corals  
- **Class 1:** Healthy Corals  

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Coral-Health"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated labels
labels = {
    "0": "Bleached Corals",
    "1": "Healthy Corals"
}

def coral_health_detection(image):
    """Predicts the health condition of coral reefs in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=coral_health_detection,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Coral Health Detection",
    description="Upload an image of coral reefs to classify their condition as Bleached or Healthy."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

---

# **Intended Use:**

The **Coral-Health** model is designed to support marine conservation and environmental monitoring. Potential use cases include:

- **Coral Reef Monitoring:** Helping scientists and conservationists track coral bleaching events.  
- **Environmental Impact Assessment:** Analyzing reef health in response to climate change and pollution.  
- **Educational Tools:** Raising awareness about coral reef health in classrooms and outreach programs.  
- **Automated Drone/ROV Analysis:** Enhancing automated underwater monitoring workflows.
