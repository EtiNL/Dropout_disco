from torch import nn

class MotionCNN1D(nn.Module):
    def __init__(self, num_features=384, latent_dim=512):
        super(MotionCNN1D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        # x.shape = (Batch, Frames, 384)
        
        # (Batch, 384, Frames)
        x = x.transpose(1, 2) 
        
        x = self.encoder(x) 
  
        x = x.view(x.size(0), -1) 
        
        return self.fc(x)
