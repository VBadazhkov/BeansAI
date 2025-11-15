from PIL import Image
import torch, torchvision
import psycopg2
import streamlit as st
import torch.nn as nn
import torchvision.transforms.v2

def prediction_sample(image):
    params = torch.load("app/models/coffee_model_best.pth", map_location="cpu")
    classes = params['classes']

    new_model = CoffeeCNN()
    new_model.load_state_dict(params['model_state_dict'])

    T = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                        torchvision.transforms.v2.RGB(),
                                        torchvision.transforms.ToTensor()])

    image = T(image)

    new_model.eval()

    prediction = new_model(image.unsqueeze(0))
    prediction = torch.softmax(prediction, dim=1)
    conf, cls = torch.max(prediction, 1)
    return (classes[cls.item()], round(conf.item() * 100, 4))

class CoffeeCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
    self.act = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)

    self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
    self.pool2 = nn.MaxPool2d(2)

    self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
    self.pool3 = nn.MaxPool2d(2)

    self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)
    self.pool4 = nn.MaxPool2d(2)
    self.adapool = nn.AdaptiveAvgPool2d((6, 6))

    self.flat = nn.Flatten()
    self.drop = nn.Dropout(0.5)
    self.lin1 = nn.Linear(512 * 6 * 6, 512)
    self.lin2 = nn.Linear(512, 128)
    self.lin3 = nn.Linear(128, 4)


  def forward(self, x):
    x = self.pool1(self.act(self.conv1(x)))
    x = self.pool2(self.act(self.conv2(x)))
    x = self.pool3(self.act(self.conv3(x)))
    x = self.pool4(self.act(self.conv4(x)))
    x = self.adapool(x)
    x = self.flat(x)
    x = self.drop(self.act(self.lin1(x)))
    x = self.drop(self.act(self.lin2(x)))
    out = self.lin3(x)
    return out
  
class DataBaseConnection:
  def __init__(self, connection):
    self.connection = connection

  def __enter__(self):
    self.cursor = self.connection.cursor()
    return self.cursor

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is None:
      self.connection.commit()
    else:
      self.connection.rollback()

    self.cursor.close()
    self.connection.close()

def get_connection():
  connection = psycopg2.connect(
  user=st.secrets.get("USER", "postgres.qnlezixijjjpapvldanz"),
  password=st.secrets["PASSWORD"],
  host=st.secrets.get("HOST", "aws-1-eu-west-1.pooler.supabase.com"),
  port=st.secrets.get("PORT", "5432"),
  database=st.secrets.get("DATABASE", "postgres"),
  sslmode="require"
  )
  return connection