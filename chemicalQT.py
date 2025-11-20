import sys
import joblib
import openpyxl
from openpyxl import Workbook
import tkinter as tk
from tkinter import messagebox
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QPushButton, \
    QLineEdit, QMessageBox, QScrollArea
from PyQt5.QtWidgets import QFileDialog,QTextEdit,QDialog, QFormLayout, QLabel, QLineEdit,QDialogButtonBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QRect
from rdkit import Chem
from rdkit.Chem import Draw

from dataset.feature_cal import MolCal
from model.Net import Net
from login import LoginWidget


class MolPredictionApp(QWidget):

    global_form_data = {}


    def __init__(self):
        super().__init__()
        #设置软件名称
        self.setWindowTitle('基于图神经网络的化学分子性质预测系统')

        # 创建一个整体垂直布局 3
        main_vertical_layout = QVBoxLayout()

        # 添加大标题标签
        title_label = QLabel('化学分子性质预测软件')
        # 设置标题样式
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; margin-bottom: 10px; background-color: #80bae1;")
        title_label.setAlignment(Qt.AlignCenter)
        #固定标题尺寸
        title_label.setFixedHeight(220)
        main_vertical_layout.addWidget(title_label)
        # 创建主水平布局，用于放置数据选择和显示模块
        main_layout = QHBoxLayout()

        # 数据选择部分，使用QGroupBox来创建一个带标题的框
        data_group = QGroupBox('预测模块')
        data_layout = QVBoxLayout()
        # 添加数据输入部分，使用QTextEdit作为一个多行文本输入框
        self.output_text = QTextEdit()
        # 设置文本框只读
        self.output_text.setReadOnly(True)
        # 设置文本框固定高度
        self.output_text.setFixedHeight(200)
        self.output_text.setLineWrapMode(QTextEdit.WidgetWidth)
        # 创建表单布局，用于放置表单元素
        form_layout = QFormLayout()

        input1 = QLineEdit()
        # input1.setText('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')  # 设置默认值
        form_layout.addRow(QLabel('请输入需要预测的分子SMILES表达式:'))
        form_layout.addRow(input1)

        # 创建开始预测按钮，并连接到get_inputs槽函数，用于处理预测逻辑
        button1 = QPushButton('开始预测')
        button1.clicked.connect(lambda checked, input1=input1: self.get_inputs(input1.text(), input2, input3, input4))


        input2 = QLineEdit()
        input3 = QLineEdit()
        input4 = QLineEdit()
        # 创建标签和对应的水平布局，用于展示LogP、QED、SAS值
        label2 = QLabel("——— LogP值 ———")
        label3 = QLabel("——— QED值 ———")
        label4 = QLabel("——— SAS值 ———")

        hbox = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()

        vbox1.addWidget(label2)
        vbox1.addWidget(input2)
        vbox2.addWidget(label3)
        vbox2.addWidget(input3)
        vbox3.addWidget(label4)
        vbox3.addWidget(input4)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

        data_layout.addLayout(form_layout)
        data_layout.addWidget(button1)
        widget = QWidget()
        widget.setLayout(hbox)
        data_layout.addWidget(widget)
        data_group.setLayout(data_layout)
        # 将数据选择组添加到主布局中
        main_layout.addWidget(data_group)

        # 显示模块
        mol_group = QGroupBox('显示模块')
        mol_layout = QVBoxLayout()
        mol_image_label = QLabel('分子图像输出区域')
        mol_image_label.setFixedSize(400, 200)
        mol_image_label.setStyleSheet("border: 1px solid black")

        start_prediction_button = QPushButton("图像显示")
        start_prediction_button.clicked.connect(lambda: self.mol_image(input1.text(), mol_image_label))

        mol_layout.addWidget(mol_image_label)
        mol_group.setLayout(mol_layout)
        mol_layout.addWidget(start_prediction_button)

        main_layout.addWidget(mol_group)

        # 模型性能显示模块
        part3_layout = QHBoxLayout()

        data_group2 = QGroupBox('模型性能')
        data_layout3 = QVBoxLayout()

        form_layout2 = QFormLayout()
        input6 = QLineEdit()
        input6.setReadOnly(True)
        input6.setPlaceholderText(f'')
        form_layout2.addRow(QLabel('数据集:'), input6)
        input7 = QLineEdit()
        input7.setReadOnly(True)
        input7.setPlaceholderText(f'')
        form_layout2.addRow(QLabel('数据样本总量:'), input7)
        input8 = QLineEdit()
        input8.setReadOnly(True)
        input8.setPlaceholderText(f'')
        form_layout2.addRow(QLabel('训练集数量:'), input8)
        input9 = QLineEdit()
        input9.setReadOnly(True)
        input9.setPlaceholderText(f'')
        form_layout2.addRow(QLabel('测试集数量:'), input9)

        button3 = QPushButton('显示数据集参数')
        button3.clicked.connect(lambda: self.show_params(input6, input7, input8, input9,pie_chart1,pie_chart2))

        data_layout3.addLayout(form_layout2)
        data_layout3.addWidget(button3)

        # 添加两个饼图以及对应的标签
        data_layout2 = QHBoxLayout()

        pie_label1 = QLabel("训练集:")
        pie_chart1 = PieChartWidget(0)  # 初始时为 0%
        data_layout2.addWidget(pie_label1)
        data_layout2.addWidget(pie_chart1)

        pie_label2 = QLabel("测试集:")
        pie_chart2 = PieChartWidget(0)  # 初始时为 0%
        data_layout2.addWidget(pie_label2)
        data_layout2.addWidget(pie_chart2)

        data_layout3.addLayout(data_layout2)
        data_group2.setLayout(data_layout3)
        part3_layout.addWidget(data_group2)

        # 模型显示模块
        model_group = QGroupBox('模型显示')
        model_layout = QVBoxLayout()
        model_img_layout = QHBoxLayout()

        model_image_label1 = QLabel('模型训练收敛图像输出区域')
        model_image_label1.setFixedSize(350, 300)
        model_image_label1.setStyleSheet("border: 1px solid black")

        model_image_label2 = QLabel('模型训练效果输出区域')
        model_image_label2.setFixedSize(350, 300)
        model_image_label2.setStyleSheet("border: 1px solid black")

        button4 = QPushButton('导入模型图像数据')
        button4.clicked.connect(lambda: self.show_image(model_image_label1, model_image_label2))

        model_img_layout.addWidget(model_image_label1)
        model_img_layout.addWidget(model_image_label2)

        model_layout.addWidget(button4)
        model_layout.addLayout(model_img_layout)
        model_group.setLayout(model_layout)

        part3_layout.addWidget(model_group)

        # 历史数据导出模块

        part4_layout = QVBoxLayout()

        button5 = QPushButton('历史数据导出')

        # 点击按钮事件连接函数 history_data
        button5.clicked.connect(lambda: self.history_data(history_data))

        # 创建 QTextEdit 控件
        history_data = QTextEdit()
        history_data.setFixedHeight(1000)  # 固定最大高度
        history_data.setReadOnly(True)  # 设置为只读模式

        # 将 QTextEdit 放入 QScrollArea 中
        scroll = QScrollArea()
        scroll.setWidget(history_data)
        scroll.setWidgetResizable(True)
        # 限制 QScrollArea 的高度为 300 像素，并提供滚动条
        scroll.setFixedHeight(300)

        # 如果内容超过固定高度，自动调整高度
        history_data.document().contentsChanged.connect(
            lambda: history_data.setFixedHeight(min(history_data.document().size().height(), 300)))

        part4_layout.addWidget(button5)
        # 添加 QScrollArea 而不是直接添加 QTextEdit
        part4_layout.addWidget(scroll)

        # 将主布局添加到垂直布局中
        main_vertical_layout.addLayout(main_layout)
        main_vertical_layout.addLayout(part3_layout)
        main_vertical_layout.addLayout(part4_layout)

        self.setLayout(main_vertical_layout)

# 以下为功能函数——————————————————————————————————————————————————————————————————————————

    def history_data(self, history_data):
        # 获取全局表单数据
        global_form_data = self.global_form_data
        print(global_form_data)

        # 组装历史数据文本
        history_text = "历史数据：\n"
        if 'results' in global_form_data:
            for idx, result in enumerate(global_form_data['results']):
                history_text += f"数据{idx + 1}:"
                history_text += f"     SMILES: {result.get('smiles', 'N/A')}\n"
                history_text += f"     LogP: {result.get('predict_logp', 'N/A')}"
                history_text += f"     QED: {result.get('predict_qed', 'N/A')}"
                history_text += f"     SAS: {result.get('predict_sas', 'N/A')}\n"

        # 设置文本显示
        history_data.setText(history_text)

        # 创建一个Excel Workbook对象
        wb = Workbook()
        ws = wb.active
        ws.title = "历史数据"

        # 添加数据到Excel表格中
        ws.append(["SMILES", "LogP", "QED", "SAS"])

        if 'results' in global_form_data:
            for result in global_form_data['results']:
                ws.append([result.get('smiles', 'N/A'),
                           result.get('predict_logp', 'N/A'),
                           result.get('predict_qed', 'N/A'),
                           result.get('predict_sas', 'N/A')])

        # 保存Excel文件
        wb.save("history_data.xlsx")

        # 弹出提示框
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("提示", "历史数据已保存到当前目录下")

    def mol_image(self, text, mol_image_label):
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            # 生成分子结构图像
            img = Draw.MolToImage(mol, size=(300, 150))
            img.save(f"./image/{text}.png", format='PNG')
            # 将 PIL.Image 转换为 QImage
            image = QImage(img.tobytes("raw", "RGB"), img.size[0], img.size[1], QImage.Format_RGB888)

        # 将 QImage 对象转换为 QPixmap 对象并设置给 QLabel
        pixmap = QPixmap.fromImage(image)
        mol_image_label.setPixmap(pixmap)
        mol_image_label.setScaledContents(True)

    def get_inputs(self, text, input2, input3, input4):
        print(text)
        mc = MolCal([text])
        X = mc.fea([text])  # 调用MolCal计算DataFrame
        scaler = joblib.load('../scaler.pkl')
        X_sca = scaler.transform(X)
        loaded_model = Net(238, 128)
        X_tensor = torch.tensor(X_sca, dtype=torch.float32)

        predictions = {}  # 为了累积结果，创建一个新的字典
        predictions['smiles'] = text

        # 加载预测 LogP 模型
        loaded_model.load_state_dict(torch.load('../qt/prelogp_model.pth'))
        y_predlogp = loaded_model(X_tensor)
        predictions['predict_logp'] = str(y_predlogp.item())

        # 加载预测 QED 模型
        loaded_model.load_state_dict(torch.load('../qt/preqed_model.pth'))
        y_predqed = loaded_model(X_tensor)
        predictions['predict_qed'] = str(y_predqed.item())

        # 加载预测 SAS 模型
        loaded_model.load_state_dict(torch.load('../qt/presas_model.pth'))
        y_predsas = loaded_model(X_tensor)
        predictions['predict_sas'] = str(y_predsas.item())

        # 存储到全局表单中
        if 'results' in self.global_form_data:
            self.global_form_data['results'].append(predictions)
        else:
            self.global_form_data['results'] = [predictions]

        # 设置到相应输入框中
        input2.setText(predictions['predict_logp'])
        input3.setText(predictions['predict_qed'])
        input4.setText(predictions['predict_sas'])



    def show_image(self, model_image_label1, model_image_label2):
        """            显示模型训练的图像。
               参数:
               model_image_label1: QLabel，用于显示损失图像的标签。
               model_image_label2: QLabel，用于显示模型图像的标签。
               """
        try:
            # 创建QPixmap对象，加载本地路径下的图像文件
            pixmap1 = QPixmap('./image/loss.png')
            # 将图像设置到对应的QLabel上
            model_image_label1.setPixmap(pixmap1)
            # 设置QLabel的缩放内容为True，以适应标签大小
            model_image_label1.setScaledContents(True)

            pixmap2 = QPixmap('./image/model2.png')
            model_image_label2.setPixmap(pixmap2)
            model_image_label2.setScaledContents(True)
        except Exception as e:
            # 如果发生异常，打印错误信息
            print(f"Error loading image: {e}")

    def show_params(self, input6, input7, input8, input9, pie_chart1, pie_chart2):
        """             显示和更新数据集参数。
                参数:
                input6, input7, input8, input9: QLineEdit，用于显示数据集参数的输入框。
                pie_chart1: PieChartWidget，第一个饼图组件。
                pie_chart2: PieChartWidget，第二个饼图组件。
                """
        inputs = [input6, input7, input8, input9]
        # 打开数据集参数的文本文件进行读取
        with open('dataparams.txt', 'r', encoding='UTF-8') as file:
            next(file)  # 跳过第一行
            for input_field in inputs:
                # 读取文件中的一行数据，并去除首尾空白字符
                line_data = file.readline().strip()
                # 将读取的数据设置到对应的输入框中
                input_field.setText(line_data)
        pie_chart1.set_percentage(70)  # 更新第一个饼图为 70%
        pie_chart2.set_percentage(30)  # 更新第二个饼图为 30%



class PieChartWidget(QWidget):
    def __init__(self, percentage, parent=None):
            super().__init__(parent)
            self.percentage = percentage  # 饼图所占比例

    # 更改饼图所占比例
    def set_percentage(self, percentage):
        self.percentage = percentage
        self.update()
    # 绘制图片
    def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

            # 定义饼图绘制区域
            # 设置饼图直径大小
            rect = QRect(0, 0, 100, 100)

            # 绘制背景圆
            painter.setBrush(Qt.lightGray)
            painter.drawEllipse(rect)

            # 绘制饼图扇形区域（根据百分比）
            painter.setBrush(QColor(2,117,195))
            painter.setPen(QPen(Qt.white, 0.5))
            startAngle = 0
            spanAngle = int(360 * self.percentage / 100)  # 换算角度
            painter.drawPie(rect, startAngle * 16, spanAngle * 16)

            # 在饼图中心显示百分比数字
            center = rect.center()
            text = f"{self.percentage}%"
            painter.drawText(center, text)

# 定义一个全局变量，用于在登录成功后存储下一个窗口的实例
next_window = None

def show_next_window():
    """
        这个函数被设计为一个槽函数，当登录成功信号被发射时，它会创建MolPredictionApp的实例
        并显示它。
    """
    global next_window
    # 创建MolPredictionApp的实例，并将其赋值给next_window
    next_window = MolPredictionApp()
    # 创建MolPredictionApp的实例，并将其赋值给next_window
    next_window.show()

if __name__ == '__main__':
    app = QApplication([])
    # 创建LoginWidget的实例
    login_widget = LoginWidget()
    # 将登录成功信号连接到show_next_window槽函数
    # 当用户成功登录时，会调用show_next_window函数显示下一个窗口
    login_widget.login_successful.connect(show_next_window)
    login_widget.show()
    # 执行QApplication的事件循环
    sys.exit(app.exec_())