from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal

class LoginWidget(QWidget):
    # 定义一个信号，用于通知外部登录成功
    login_successful = pyqtSignal()

    def __init__(self):
        super().__init__()
        # 设置窗口标题
        self.setWindowTitle('基于图神经网络的化学分子性质预测系统')

        # 创建垂直布局，作为主布局
        layout = QVBoxLayout()

        # 添加欢迎标签
        welcome_label = QLabel('欢迎使用化学分子性质预测系统')
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet('font-size: 20pt; font-weight: bold; margin-bottom: 50px;')

        layout.setAlignment(Qt.AlignCenter)  # 设置布局的对齐方式为居中

        # 设置背景颜色，字体，和大小
        self.setStyleSheet('font-size: 10pt;')
        self.setMinimumSize(1200, 600)

        # 将欢迎标签添加到布局中
        layout.addWidget(welcome_label)

        # 创建用户名和密码输入布局
        formname_layout = QHBoxLayout()
        formpass_layout = QHBoxLayout()
        # 添加用户名标签和输入框
        username_label = QLabel('用户名：')
        username_label.setStyleSheet('font-size: 14pt;')
        formname_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('请输入用户名')
        self.username_input.setText('admin')  # 设置默认值
        self.username_input.setStyleSheet('font-size: 10pt; background-color: lightgray;height:50px')
        formname_layout.addWidget(self.username_input)
        layout.addLayout(formname_layout)

        # 添加密码标签和输入框
        password_label = QLabel('密  码：')
        password_label.setStyleSheet('font-size: 14pt;')
        # 添加到水平布局中
        formpass_layout.addWidget(password_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('请输入密码')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet('font-size: 10pt; background-color: lightgray; height:50px')
        # 添加到水平布局中
        formpass_layout.addWidget(self.password_input)
        # 将水平布局添加到垂直布局中
        layout.addLayout(formpass_layout)

        # 添加登录按钮
        login_button = QPushButton('登录')
        login_button.setStyleSheet('font-size: 14pt; background-color: #80bae1; width: 100px; height: 50px; margin-top: 20px;')
        # 连接登录按钮的信号到登录处理槽函数
        login_button.clicked.connect(self.login_clicked)
        layout.addWidget(login_button)

        # 将布局应用到当前窗口
        self.setLayout(layout)

    def login_clicked(self):
        # 获取用户名和密码输入框的文本
        username = self.username_input.text()
        password = self.password_input.text()

        # 在这里可以添加登录逻辑，比较用户名和密码等
        # 这里假设用户名为 'admin'，密码为 '1' 时登录成功
        if username == 'admin' and password == '1':
            msg_box = QMessageBox()
            msg_box.information(self, '登录成功', '登录成功！')
            self.login_successful.emit()  # 发射登录成功信号
            # 登录成功后关闭登录界面
            self.close()
        else:
            # 如果用户名或密码错误，弹出警告框
            msg_box = QMessageBox()
            msg_box.warning(self, '登录失败', '登录失败！请检查用户名和密码')