#%%
import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from PyQt5.QtWidgets import QApplication
print(os.environ)

##
# Unset QT_DEBUG_PLUGINS to disable plugin debugging messages
if 'QT_DEBUG_PLUGINS' in os.environ:
    del os.environ['QT_DEBUG_PLUGINS']


import sys
from PyQt5.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)

label = QLabel('Hello World')
label.show()

sys.exit(app.exec())