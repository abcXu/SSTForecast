from PIL import Image
import os

# 设置输入和输出文件路径
input_file = 'path/to/your-input.png'
output_file = 'path/to/your-output.eps'

# 打开PNG图像
img = Image.open(input_file)

# 将PNG图像保存为EPS格式
img.save(output_file, 'EPS')


