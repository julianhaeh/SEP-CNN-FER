import torch
import torch.nn as nn

class GAP_4_64_96_128_196(torch.nn.Module):

    def __init__(self):
        super(GAP_4_64_96_128_196, self).__init__()
        
        self.feature = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64,96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(196, 6)

    def forward(self, x):
        
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
if __name__ == "__main__":
    model = GAP_4_64_96_128_196()
    print(f"Architecture has {sum(p.numel() for p in model.parameters())} parameters")