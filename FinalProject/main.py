import tkinter as tk
from tkinter import filedialog
import numpy
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from numpy import array
from keras.src.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn import *
from keras.src.metrics import *
import flet as ft
import pandas as pd
import datetime

#numpy.random.seed(7)

dataset = None
X = None
Y = None
Y_one_hot = None
history = None

lossType = ["binary_crossentropy", "categorical_crossentropy", "hinge", "mean_squared_error",
            "mean_absolute_error", "sparse_categorical_crossentropy"]
optimizerType = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
metricType = ["accuracy", "precision", "recall", "f1_score", "mean_squared_error", "mean_absolute_error"]
activationMode = ["relu", "sigmoid", "tanh", "softmax", "linear"]


def open_csv(event=None):
    global dataset, X, Y, Y_one_hot

    # Crear una instancia de Tkinter y ocultarla
    root = tk.Tk()
    root.withdraw()

    # Obtener las dimensiones de la pantalla
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calcular las coordenadas para centrar la ventana de diálogo
    x = (screen_width - root.winfo_reqwidth()) / 2
    y = (screen_height - root.winfo_reqheight()) / 2

    # Configurar la geometría de la ventana de diálogo
    root.geometry("+%d+%d" % (x, y))

    # Mostrar la ventana de diálogo y hacerla activa y visible en primer plano
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()

    # Obtener la ruta del archivo
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    root.destroy()

    if file_path:
        try:
            dataset = numpy.loadtxt(file_path, delimiter=",", skiprows=1, encoding="utf-8")
            X = dataset[0:, 0:13]
            Y = dataset[0:, -1]
            Y_one_hot = to_categorical(Y)
            print(dataset)
        except Exception as e:
            print("Error loading CSV file:", e)
            return None
    else:
        return None


class Home(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page


class Configuration(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page
        self.lossfunction = ft.Dropdown(
            label="Loss Function",
            hint_text="Choose a loss function",
            border_color="blue",
            options=[
                ft.dropdown.Option(opt) for opt in lossType
            ]
        )
        self.optimizer = ft.Dropdown(
            label="Optimizer",
            hint_text="Choose a optimizer",
            border_color="blue",
            options=[
                ft.dropdown.Option(opt) for opt in optimizerType
            ],
            autofocus=True
        )
        self.metrics = ft.Dropdown(
            label="Metrics",
            hint_text="Choose a metric",
            border_color="blue",
            options=[
                ft.dropdown.Option(opt) for opt in metricType
            ]
        )
        self.validationsplit = ft.Slider(
            value=10,
            min=10,
            max=100,
            divisions=9,
            label="{value}%"
        )
        self.epochs = ft.TextField(label="Epochs", border_color="blue",
                                   input_filter=ft.NumbersOnlyInputFilter(),
                                   max_length=3)
        self.batchsize = ft.TextField(label="Batch Size", border_color="blue",
                                      input_filter=ft.NumbersOnlyInputFilter(),
                                      max_length=2)

        self.OpenFile = ft.Container(
            bgcolor="#222222",
            border_radius=10,
            col=4,
            padding=10,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Open the CSV File",
                            size=40,
                            text_align="center",
                            ),
                    ft.Container(
                        content=ft.Row(
                            spacing=5,
                            alignment=ft.MainAxisAlignment.CENTER,
                            controls=[
                                ft.TextButton(text="Open file",
                                              icon=ft.icons.UPLOAD_FILE,
                                              icon_color="white",
                                              style=ft.ButtonStyle(color="white", bgcolor="blue"),
                                              on_click=open_csv,
                                              ),
                            ]
                        )
                    )
                ]
            )
        )

        self.CompileConfigModule = ft.Container(
            bgcolor="#222222",
            border_radius=10,
            col=4,
            padding=10,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Configuration for compile",
                            size=40,
                            text_align="center",
                            ),
                    self.lossfunction,
                    self.optimizer,
                    self.metrics,
                ]
            )
        )

        self.TrainConfigModule = ft.Container(
            bgcolor="#222222",
            border_radius=10,
            col=4,
            padding=10,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Configuration for training",
                            size=40,
                            text_align="center",
                            ),
                    ft.Text("Validation Split",
                            text_align="center",
                            ),
                    self.validationsplit,
                    self.epochs,
                    self.batchsize,
                    ft.TextButton(text="Train",
                                  icon=ft.icons.SAVE_ROUNDED,
                                  icon_color="white",
                                  style=ft.ButtonStyle(color="white", bgcolor="blue"),
                                  on_click=self.TrainAction,
                                  ),
                ]
            )
        )

        self.content = ft.ResponsiveRow(
            controls=[
                self.OpenFile,
                self.CompileConfigModule,
                self.TrainConfigModule,
            ]
        )

    def TrainAction(self, e):
        global history
        loss = self.lossfunction.value
        optimizer = self.optimizer.value
        metrics = self.metrics.value
        validationsplit = self.validationsplit.value / 100
        epochs = int(self.epochs.value)
        batchsize = int(self.batchsize.value)

        model = Sequential()
        model.add(Dense(units=15, input_dim=20, activation="relu"))
        model.add(Dense(units=10, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        history = model.fit(X, Y_one_hot, validation_split=validationsplit, epochs=epochs, batch_size=batchsize,
                            verbose=1)
        print(history.history.keys())

        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Modelo de Exactitud')
        plt.ylabel('Exactitud')
        plt.xlabel('Epoca')
        plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
        plt.show()

    def build(self):
        return self.content


class PredictNewCase(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page


def main(page: ft.Page):
    page.update()

    def changeTab(e):
        my_index = e.control.selected_index
        home.visible = True if my_index == 0 else False
        predictNewCase.visible = True if my_index == 1 else False
        configuration.visible = True if my_index == 2 else False
        page.update()

    page.title = "Neuronal Model"
    page.window_min_width = 1100
    page.window_min_height = 500
    page.navigation_bar = ft.NavigationBar(
        bgcolor="#222222",
        selected_index=0,
        on_change=changeTab,
        destinations=[
            ft.NavigationDestination(
                icon=ft.icons.HOME_ROUNDED,
                label='Home'),
            ft.NavigationDestination(
                icon=ft.icons.PERSON_2_ROUNDED,
                label="Predict",
            ),
            ft.NavigationDestination(
                icon=ft.icons.SETTINGS_ROUNDED,
                label="Settings",
            ),
        ],
    )

    home = Home(page)
    home.visible = True
    predictNewCase = PredictNewCase(page)
    predictNewCase.visible = False
    configuration = Configuration(page)
    configuration.visible = False

    page.add(
        home,
        predictNewCase,
        configuration,
    )
    page.theme_mode = ft.ThemeMode.DARK


ft.app(main)
