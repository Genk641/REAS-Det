import os
import pandas as pd
import matplotlib.pylab as plt

# Get current working directory
pwd = os.getcwd()

# List of training data file names
names = ['yolov8n', 'yolov8n+RFA', 'yolov8n+MultiSEAM', 'yolov8n+C3RFEM', 'yolov8n+MultiSEAM+C3RFEM', 'yolov8n+C3RFEM+softNMS', 'yolov8n+RFA+MultiSEAM','yolov8n+RFA+C3RFEM','yolov8n+RFA+MultiSEAM+C3RFEM', 'REAS-Det']

# Plotting metrics
plt.figure(figsize=(10, 10))

# Plot precision
plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['   metrics/precision(B)'], label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()

# Plot recall
plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['      metrics/recall(B)'], label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()

# Plot mAP_0.5
plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['       metrics/mAP50(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()

# Plot mAP_0.5:0.95
plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['    metrics/mAP50-95(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png saved in {pwd}/metrice_curve.png')

# Plotting loss
plt.figure(figsize=(15, 10))

# Plot train/box_loss
plt.subplot(2, 3, 1)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['         train/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/box_loss')
plt.legend()

# Plot train/dfl_loss
plt.subplot(2, 3, 2)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['         train/dfl_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/dfl_loss')
plt.legend()

# Plot train/cls_loss
plt.subplot(2, 3, 3)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['         train/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/cls_loss')
plt.legend()

# Plot val/box_loss
plt.subplot(2, 3, 4)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['           val/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/box_loss')
plt.legend()

# Plot val/dfl_loss
plt.subplot(2, 3, 5)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['           val/dfl_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/dfl_loss')
plt.legend()

# Plot val/cls_loss
plt.subplot(2, 3, 6)
for i in names:
    data = pd.read_csv(f'runs/detect/{i}/results.csv')
    plt.plot(data['           val/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/cls_loss')
plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png')
print(f'loss_curve.png saved in {pwd}/loss_curve.png')
