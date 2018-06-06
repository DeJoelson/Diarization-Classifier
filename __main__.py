import sys
from PyQt5 import QtWidgets
from software_model.diarizer import Diarizer
from software_view.view import MainWindow

import software_model.classifier as classy
"""
Below are examples highlighting how the software can be used
"""


def show_gui():
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


def train_net():
    files_to_train_on = ['HS_D{0:0=2d}'.format(i) for i in range(1, 38)]
    del files_to_train_on[files_to_train_on.index('HS_D11')]
    del files_to_train_on[files_to_train_on.index('HS_D22')]

    classy.train_classifier(files_to_train_on, None)


def evaluate_on_a_particular_file():
    previously_saved_network_model = "Model/ultimate_model_saved_weights.ckpt"
    dire = Diarizer(previously_saved_network_model)
    dire.annotate_wav_file("HS_D01")


if __name__ == '__main__':
    print("Go to line 38 in the file '__main__.py' and uncomment a subsequent line")

    train_net()
    # show_gui()
    # evaluate_on_a_particular_file()
