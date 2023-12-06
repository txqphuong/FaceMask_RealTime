import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog , QMessageBox
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn as nn
from PyQt5.QtCore import *
from PyQt5.QtGui import *



# dữ liệu
data_dir = 'dataset'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'
# định hình 
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
# Tạo dataset
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)
# Tạo dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)


# Xây dựng mô hình
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, len(train_data.classes))
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = CNN()



### tải lại mô hình đã lưu
# Tạo mô hình giống với mô hình đã được huấn luyện trước đó
model = torchvision.models.resnet18()
# Tải trọng số mô hình đã lưu vào mô hình mới
PATH = '../DoAnML/CNN/model.pth'
model.load_state_dict(torch.load(PATH))


# Sử dụng mô hình để dự đoán
input_data = torch.randn(1, 3, 224, 224)  # ví dụ input data ngẫu nhiên
output = model(input_data)


# tạo giao diện
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle('Dự đoán trái cây và rau củ quả  🍅')
window.setGeometry(100, 100, 400, 200)


# Tạo các thành phần giao diện
image_path = None
image_label = QLabel('KHÔNG CÓ BỨC ẢNH NÀO ĐƯỢC CHỌN. 🍅 ')
image_label.setStyleSheet("background-color: #F0F0F0;font-weight:bold; font-size: 24px; color: red;")
image_label.setFont(QFont('Arial', 12))

image_display = QLabel()
image_display.setFixedSize(224, 224)

browse_button = QPushButton('Tải ảnh lên...')
browse_button.setStyleSheet("background-color: 'black'; font-size: 12pt; font-weight: bold; color:white")

predict_button = QPushButton('Dự Đoán')
predict_button.setStyleSheet("background-color: 'green' ; font-size: 12pt; font-weight: bold; color:white")


recovery_button = QPushButton('Khôi Phục Ảnh Gốc')
recovery_button.setStyleSheet("background-color: 'Blue'; font-size: 12pt; font-weight: bold; color:white")

result_label = QLabel('')
result_label.setFont(QFont('Arial', 12))

# Xử lý sự kiện khi nút "Browse" được nhấn
def browse_image():
    global image_path
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_path, _ = QFileDialog.getOpenFileName(window, "Chọn ảnh", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
    if file_path:
        image_path = file_path
        image_label.setText('Ảnh đầu vào : ' + image_path)        
        # Chuyển đổi hình ảnh PIL thành QImage
        img = Image.open(image_path)
        img = Image.open(image_path)
        img = img.resize((224, 224))
        gray_img = img.convert('L')  # Chuyển đổi sang ảnh màu xám
        qimage = QImage(gray_img.tobytes(), gray_img.size[0], gray_img.size[1], QImage.Format_Grayscale8)
        image_display.setPixmap(QPixmap.fromImage(qimage))


# Xử lý sự kiện khi nút "Predict" được nhấn
def predict_image():
    
    if image_path:
        # Đọc và tiền xử lý hình ảnh đầu vào
        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)        
     # Dùng mô hình để dự đoán
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            
        # Chuyển đổi kết quả dự đoán sang loại rau củ hay trái cây và tên loại
        class_names = train_data.classes
        if predicted.item() == 0:
            result_label.setText("Ảnh này được phân loại là một loại hoa quả  .\nTên loại là: " + class_names[predicted.item()])
        else:
            result_label.setText("Ảnh này được phân loại là một loại rau củ   .\nTên loại là: " + class_names[predicted.item()])
            
            
# Recovery image
       
       
    else:
        message_box = QMessageBox()
        message_box.setWindowTitle("Thông báo")
        message_box.setText("Hãy chọn vào 1 bức ảnh.")
        message_box.exec_()
        
        
def recovery_image():
     img_pixmap = QPixmap(image_path).scaled(image_display.size(), Qt.KeepAspectRatio)
     image_display.setPixmap(img_pixmap)



# Kết nối các thành phần với sự kiện
browse_button.clicked.connect(browse_image)
predict_button.clicked.connect(predict_image)
recovery_button.clicked.connect(recovery_image)

# Bố trí các thành phần trên cửa sổ giao diện
image_layout = QVBoxLayout()
image_layout.addWidget(image_label)
image_layout.addWidget(image_display)

button_layout = QHBoxLayout()
button_layout.addWidget(browse_button)

button_layout = QHBoxLayout()
button_layout.addWidget(browse_button)
button_layout.addWidget(predict_button)
button_layout.addWidget(recovery_button)


main_layout = QVBoxLayout()
main_layout.addLayout(image_layout)
main_layout.addLayout(button_layout)
main_layout.addWidget(result_label)





window.setLayout(main_layout)

window.show()
sys.exit(app.exec_())

