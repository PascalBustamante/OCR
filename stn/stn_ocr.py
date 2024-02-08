import torch
import torch.nn as nn
import torch.nn.functional as F

class StnOcr(nn.Module):
    def __init__(self, input_shape, nb_classes, detection_filter, recognition_filter):
        super(StnOcr, self).__init__()
        self.num_labels = 3
        self.num_steps = 1
        self.detection_filter = detection_filter
        self.recognition_filter = recognition_filter
        
        # Define convolutional layers for detection and recognition
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Residual blocks
        self.residualNet_block1 = self._make_residual_block(detection_filter[0], size=3, stride=1)
        self.residualNet_block2 = self._make_residual_block(detection_filter[1], size=3, stride=1)
        self.residualNet_block3 = self._make_residual_block(detection_filter[2], size=3, stride=1)
        
        # LSTM for detection
        self.lstm = nn.LSTM(input_size=self.num_steps * 256, hidden_size=256, bidirectional=True, batch_first=True)
        self.fc_theta = nn.Linear(512, 6)

        # Dense layers for recognition
        self.fc_recognition = nn.Sequential(
            nn.Linear(self.num_steps * recognition_filter[-1], 256),
            nn.ReLU(),
            nn.Linear(256, self.num_steps * self.num_labels * 11)
        )

    def _make_residual_block(self, filters, size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(32, filters, kernel_size=size, stride=stride, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x, flag='detection'):
        if flag == 'detection':
            x = self._forward_detection(x)
        else:
            x = self._forward_recognition(x)
        return x

    def _forward_detection(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.residualNet_block1(x)
        x = self.residualNet_block2(x)
        x = self.residualNet_block3(x)
        x = F.avg_pool2d(x, kernel_size=5)
        x = x.view(x.size(0), -1)
        x = x.view(-1, self.num_steps, x.size(-1))
        x, _ = self.lstm(x)
        theta = self.fc_theta(x)
        return theta

    def _forward_recognition(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.residualNet_block1(x)
        x = self.residualNet_block2(x)
        x = self.residualNet_block3(x)
        x = F.avg_pool2d(x, kernel_size=5)
        x = x.view(x.size(0), -1)
        x = self.fc_recognition(x)
        x = x.view(-1, self.num_steps, self.num_labels, 11)
        x = F.softmax(x, dim=-1)
        return x

if __name__ == "__main__":
    detection_filter = [32, 48, 48]
    recognition_filter = [32, 64, 128]
    input_shape = (1, 128, 128)  # Channels, Height, Width
    nb_classes = 10
    stn_model = StnOcr(input_shape, nb_classes, detection_filter, recognition_filter)

    input_data = torch.randn(1, 1, 128, 128)  # Batch, Channels, Height, Width
    detection_output = stn_model(input_data, flag='detection')
    print("Detection output shape:", detection_output.shape)

    recognition_output = stn_model(input_data, flag='recognition')
    print("Recognition output shape:", recognition_output.shape)