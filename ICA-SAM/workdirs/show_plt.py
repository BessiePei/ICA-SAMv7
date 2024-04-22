import re
import matplotlib.pyplot as plt

log_file  = ''

# Initialize lists to store the extracted data
epochs = []

ious = []
dices = []
train_losses = []

# Read the log file and extract the relevant data
with open(log_file, 'r') as file:
    for line in file:
        match = re.search(r"epoch: (\d+).*Train loss: (-?[\d.]+).*'iou': '([\d.]+)'.*'dice': '([\d.]+)'", line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            iou = float(match.group(3))
            dice = float(match.group(4))
            epochs.append(epoch)
            train_losses.append(train_loss)
            ious.append(iou)
            dices.append(dice)

# Plot the line graph
plt.plot(epochs, train_losses, label='Train loss')
plt.plot(epochs, ious, label='iou')
plt.plot(epochs, dices, label='dice')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Train Loss, IOU, and Dice Scores over Epochs')
plt.legend()
plt.savefig(log_file.split('.')[0]+'.png')
plt.show()