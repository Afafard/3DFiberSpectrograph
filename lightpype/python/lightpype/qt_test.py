#%%

import sys
from PyQt5.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)

label = QLabel('Hello World')
label.show()

sys.exit(app.exec())