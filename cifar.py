import tkinter as tk
from tkinter import Canvas, Label, Button, Scale, OptionMenu, StringVar
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageTk

# 載入預訓練的CIFAR-10模型
model = load_model('cifar_model.keras')
cifar10_classes = {
    0: "飛機",
    1: "汽車",
    2: "鳥",
    3: "貓",
    4: "鹿",
    5: "狗",
    6: "青蛙",
    7: "馬",
    8: "船",
    9: "卡車"
}

# 創建Tkinter應用程序窗口
app = tk.Tk()
app.title("CIFAR-10 圖像辨識")
app.geometry("700x650")  # 設定視窗大小

# 創建一個彩色畫布，用於繪製圖像
canvas = Canvas(app, width=448, height=448, bg='white')  # 設定背景為白色
canvas.pack()

# 創建一個標籤，用於顯示預測的類別
label = Label(app, text="預測類別: ", font=("TkDefaultFont", 16, "bold"))
label.pack()

# 創建一個滑動條，用於調整筆畫寬度
pen_width = Scale(app, from_=1, to=15, orient='horizontal', label='筆畫寬度')
pen_width.set(3)
pen_width.pack()

# 創建一個選單，用於選擇畫筆顏色，包括白色作為橡皮擦
pen_color_label = Label(app, text="選擇畫筆顏色:")
pen_color_label.pack()

pen_colors = ["black", "red", "blue", "green", "yellow", "white", "brown", "purple"]
selected_color = StringVar()
selected_color.set("black")

pen_color_menu = OptionMenu(app, selected_color, *pen_colors)
pen_color_menu.pack()

# 開始繪製圖像
drawing = False
last_x, last_y = None, None
drawn_items = []

def start_drawing(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def end_drawing(event):
    global drawing
    drawing = False
    last_x, last_y = None, None
    predict_image()

def draw(event):
    if drawing:
        x, y = event.x, event.y
        canvas.create_oval(x, y, x+pen_width.get(), y+pen_width.get(), fill=selected_color.get(), width=0)
        drawn_items.append(canvas.create_oval(x, y, x+pen_width.get(), y+pen_width.get(), fill=selected_color.get(), width=0))
        last_x, last_y = x, y

# 添加反悔功能
def undo_last():
    if drawn_items:
        canvas.delete(drawn_items.pop())

# 清除畫布
def clear_canvas():
    canvas.delete("all")
    label.config(text="預測類別: ")

canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<ButtonRelease-1>", end_drawing)
canvas.bind("<B1-Motion>", draw)

undo_button = Button(app, text="反悔", command=undo_last)
undo_button.pack()
clear_button = Button(app, text="清除", command=clear_canvas)
clear_button.pack()

def predict_image():
    image = canvas_to_image(canvas)
    predictions = model.predict(image)
    print(format_cifar10_predictions(predictions))
    predicted_class = np.argmax(predictions[0])
    class_name = cifar10_classes.get(predicted_class, "未知")
    label.config(text=f"預測類別: {class_name}")

def canvas_to_image(canvas):
    # 創建一個空白圖像
    image = Image.new('RGB', (448, 448), 'white')  # 設定背景為白色，並支援彩色圖像
    draw = ImageDraw.Draw(image)

    # 獲取畫布上的所有項目
    items = canvas.find_all()

    for item in items:
        item_coords = canvas.coords(item)
        # 提取x和y坐標
        x1, y1, x2, y2 = item_coords
        # 設定筆畫顏色和寬度
        draw.line([x1, y1, x2, y2], fill=selected_color.get(), width=pen_width.get())

    # 調整圖像大小為32x32像素，與CIFAR-10圖像大小相符
    image = image.resize((32, 32), Image.ANTIALIAS)

    # 將圖像轉換為NumPy數組
    image_array = np.array(image)

    # 正規化圖像並返回
    image_array = image_array.reshape(1, 32, 32, 3).astype('float32') / 255
    return image_array

def format_cifar10_predictions(predictions):
    # CIFAR-10類別名稱
    cifar10_classes = ["飛機", "汽車", "鳥", "貓", "鹿", "狗", "青蛙", "馬", "船", "卡車"]

    # 處理預測結果
    result = ""
    for i in range(len(cifar10_classes)):
        class_name = cifar10_classes[i]
        probability = predictions[0][i] * 100  # 將概率轉換為百分比
        result += f"{class_name} : {probability:.2f}%\n"

    return result

app.mainloop()
