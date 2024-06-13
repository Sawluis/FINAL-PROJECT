import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from keras.src.saving import load_model
from sklearn.metrics import confusion_matrix
import flet as ft

np.random.seed(7)

dataset = None
X = None
Y = None
Y_one_hot = None
history = None
TotalCases = None
Accuracy = None
MissClassificationRate = None
Recall = None
Especifity = None
Precition = None
PrecitionNeg = None
neuronal_network_filename = "neuronal_network.keras"
precision_plot_filename = "precision-plot.png"

lossType = ["binary_crossentropy", "categorical_crossentropy", "hinge", "mean_squared_error",
            "mean_absolute_error", "sparse_categorical_crossentropy"]
optimizerType = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
metricType = ["accuracy", "precision", "recall", "f1_score", "mean_squared_error", "mean_absolute_error"]
activationMode = ["relu", "sigmoid", "tanh", "softmax", "linear"]
boolType = ["True", "False"]
gender = ["Female", "Male"]
decimalFilter = ft.InputFilter(regex_string=r'[0-9]+[.]{0,1}[0-9]*', allow=True, replacement_string="")


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
            dataset = np.loadtxt(file_path, delimiter=",", skiprows=1, encoding="utf-8")
            dataset[:, 2] /= 100
            dataset[:, 6] /= 100000
            X = dataset[0:, 0:12]
            Y = dataset[0:, -1]
            Y_one_hot = to_categorical(Y)
            #unique, counts = numpy.unique(Y, return_counts=True)
            #print("Matriz X \n", X[:, 6])
            #print("Matriz Y \n", Y)
            #print("Cantidad por categoria: \n", dict(zip(unique, counts)))
            #print(Y_one_hot)
        except Exception as e:
            print("Error loading CSV file:", e)
            return None
    else:
        return None


class Home(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page
        self.Title = ft.Text(
            "Home",
            size=40,
            text_align="center",
        )
        self.total_cases_text = ft.Text(
            f"The total number of cases in the dataset is {TotalCases}",
            size=20,
        )
        self.accuracy_text = ft.Text(
            f"The accuracy is {Accuracy}",
            size=20,
        )
        self.Home = ft.Container(
            bgcolor="#222222",
            border_radius=10,
            expand=True,
            padding=10,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
                controls=[
                    self.Title,
                    ft.ResponsiveRow(
                        alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        spacing=10,
                        controls=[
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.ListTile(
                                                leading=ft.Icon(ft.icons.ALBUM),
                                                title=ft.Text("Total Cases"),
                                                subtitle=self.total_cases_text
                                            ),
                                        ]
                                    )
                                ),
                            )
                        ],
                    ),
                    ft.ResponsiveRow(
                        alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        spacing=10,
                        controls=[
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.ListTile(
                                                leading=ft.Icon(ft.icons.ALBUM),
                                                title=ft.Text("Accuracy"),
                                                subtitle=self.accuracy_text
                                            ),
                                        ]
                                    )
                                ),
                            )
                        ],
                    ),
                ]
            )
        )

        self.content = ft.ResponsiveRow(
            controls=[
                self.Home
            ]
        )

    def build(self):
        return self.content

    def update_text(self, total_cases, accuracy):
        self.total_cases_text.value = f"The total number of cases in the dataset is {total_cases}"
        self.accuracy_text.value = f"The accuracy is {accuracy:.2f}"
        self.update()


class PredictNewCase(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page
        self.Age = ft.TextField(label="Age", border_color="#a0a1a4",
                                helper_text="Age of the patient",
                                hint_text="40-95",
                                input_filter=ft.NumbersOnlyInputFilter(),
                                max_length=2,
                                )
        self.Anaemia = ft.Dropdown(
            label="Anaemia",
            helper_text="Decrease of red blood cells or hemoglobin",
            hint_text="Yes or No",
            border_color="#a0a1a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
            autofocus=True
        )
        self.HighBloodPressure = ft.Dropdown(
            label="High blood pressure",
            helper_text="If a patient has hypertension",
            hint_text="Yes or No",
            border_color="#a4a4a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
            autofocus=True
        )
        self.CreatininePhosphoKinase = ft.TextField(
            label="Creatinine PhosphoKinase (CPK)",
            helper_text="Level of the CPK enzyme in the blood (mcg/L)",
            hint_text="23-7861",
            border_color="#a4a4a4",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=4,
        )
        self.Diabetes = ft.Dropdown(
            label="Diabetes",
            helper_text="If the patient has diabetes",
            hint_text="Yes or No",
            border_color="#a4a4a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
            autofocus=True
        )
        self.EjectionFraction = ft.TextField(
            label="Ejection fraction", border_color="#a4a4a4",
            helper_text="Percentage of blood leaving the heart at each contraction",
            hint_text="14-80",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=2,
        )
        self.Sex = ft.Dropdown(
            label="Sex",
            helper_text="Woman or man",
            hint_text="Gender",
            border_color="#a4a4a4",
            options=[
                ft.dropdown.Option(opt) for opt in gender
            ],
            autofocus=True
        )
        self.Platelets = ft.TextField(
            label="Platelets", border_color="#f9f9f9",
            helper_text="Platelets in the blood (kiloplatelets/mL)",
            hint_text="25.01-850.00",
            input_filter=decimalFilter,
            max_length=6,
        )
        self.SerumCreatinine = ft.TextField(
            label="Serum creatinine", border_color="#f9f9f9",
            helper_text="Level of creatinine in the blood (mg/dL)",
            hint_text="0.50-9.40",
            input_filter=decimalFilter,
            max_length=4,
        )
        self.SerumSodium = ft.TextField(
            label="Serum sodium", border_color="#f9f9f9",
            helper_text="Level of sodium in the blood (mEq/L)",
            hint_text="114-148",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=3,
        )
        self.Smoking = ft.Dropdown(
            label="Smoking",
            helper_text="If the patient smokes",
            hint_text="Yes or No",
            border_color="#f9f9f9",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
            autofocus=True
        )
        self.Time = ft.TextField(
            label="Time (Days)", border_color="#f9f9f9",
            helper_text="Follow-up period",
            hint_text="4-285",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=3,
        )
        self.Title = ft.Text(
            "Prediction of a new case",
            size=40,
            text_align="center",
        )

        self.predictCase = ft.Container(
            bgcolor="#222222",
            border_radius=10,
            expand=True,
            padding=10,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
                controls=[
                    self.Title,
                    ft.ResponsiveRow(
                        alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        spacing=10,
                        controls=[
                            ft.Column(
                                col={"sm": 4},
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=30,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    self.Age,
                                    self.Anaemia,
                                    self.HighBloodPressure,
                                    self.CreatininePhosphoKinase,
                                ]
                            ),
                            ft.Column(
                                col={"sm": 4},
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=30,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    self.Diabetes,
                                    self.EjectionFraction,
                                    self.Sex,
                                    self.Platelets,
                                ]
                            ),
                            ft.Column(
                                col={"sm": 4},
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=30,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    self.SerumCreatinine,
                                    self.SerumSodium,
                                    self.Smoking,
                                    self.Time,
                                ]
                            ),
                        ],
                    ),
                    ft.TextButton(text="Predict",
                                  icon=ft.icons.START,
                                  icon_color="white",
                                  style=ft.ButtonStyle(color="white", bgcolor="blue"),
                                  on_click=self.PredictNewCase,
                                  ),
                ]
            )
        )

        self.content = ft.ResponsiveRow(
            controls=[
                self.predictCase,
            ]
        )

    def PredictNewCase(self):
        model = load_model(neuronal_network_filename)

        # Aqui van los values de los labels de la nueva prediccion pero hay que validarlos
        Xnew = np.array(
            [
                self.Age.value,
            ]
        )

        Ynew = model.predict(Xnew)
        newPredictions = [round(x[0]) for x in Ynew]
        print("X=%s, Predicted %s " % (Xnew[0], newPredictions[0]))

    def build(self):
        return self.content


class Configuration(ft.UserControl):
    def __init__(self, page, home):
        super().__init__(expand=True)
        self.page = page
        self.home = home
        self.lossfunction = ft.Dropdown(
            label="Loss Function",
            value="binary_crossentropy",
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
            value="adam",
            options=[
                ft.dropdown.Option(opt) for opt in optimizerType
            ],
            autofocus=True
        )
        self.metrics = ft.Dropdown(
            label="Metrics",
            value="accuracy",
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
                                   value="150",
                                   input_filter=ft.NumbersOnlyInputFilter(),
                                   max_length=3)
        self.batchsize = ft.TextField(label="Batch Size", border_color="blue",
                                      input_filter=ft.NumbersOnlyInputFilter(),
                                      value="10",
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
        global TotalCases, Accuracy, MissClassificationRate, Recall, Especifity, Precition, PrecitionNeg

        if dataset is None:
            return
        else:
            pass

        model = Sequential()
        model.add(Dense(units=12, input_dim=12, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=5, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        history = model.fit(X, Y, validation_split=validationsplit, epochs=epochs, batch_size=batchsize,
                            verbose=0)
        #print(history.history.keys())

        model.save(filepath=neuronal_network_filename, overwrite=True)
        predictions = model.predict(X)
        rounded = [round(x[0]) for x in predictions]

        Xnew = np.array(
            [
                [75, 0, 5.82, 0, 20, 1, 2.65, 1.9, 130, 1, 0, 3],
                [55, 0, 78.61, 0, 38, 0, 2.6335803, 1.1, 136, 1, 0, 4],
                [65, 0, 1.46, 0, 20, 0, 1.62, 1.3, 129, 1, 1, 7]
            ]
        )

        Ynew = model.predict(Xnew)
        newPredictions = [round(x[0]) for x in Ynew]
        #print("X=%s, Predicted %s " % (Xnew[0], newPredictions[0]))

        matrix = confusion_matrix(Y, rounded)
        print(matrix)

        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Modelo de Exactitud')
        plt.ylabel('Exactitud')
        plt.xlabel('Epoca')
        plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
        plt.savefig(precision_plot_filename)

        TP = matrix[1, 1]
        FN = matrix[1, 0]
        FP = matrix[0, 1]
        TN = matrix[0, 0]

        TotalCases = TP + FN + FP + TN
        Accuracy = (TP + TN) / TotalCases
        MissClassificationRate = (FP + FN) / TotalCases
        Recall = TP / (TP + FN)
        Especifity = TN / (TN + FP)

        #Precision de positivos que clasifica correctamente
        Precition = TP / (TP + FP)

        # Precision de negativos que clasifica correctamente
        PrecitionNeg = TN / (TN + FN)

        self.home.update_text(TotalCases, Accuracy)

    def build(self):
        return self.content


def main(page: ft.Page):
    page.update()

    def changeTab(e):
        my_index = e.control.selected_index
        home.visible = True if my_index == 0 else False
        predictNewCase.visible = True if my_index == 1 else False
        configuration.visible = True if my_index == 2 else False
        page.update()

    page.title = "Neuronal Model"
    page.window_width = 1100
    page.window_height = 800
    page.window_min_width = 900
    page.window_min_height = 800
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
    configuration = Configuration(page, home)
    configuration.visible = False

    page.add(
        home,
        predictNewCase,
        configuration,
    )
    page.theme_mode = ft.ThemeMode.DARK


ft.app(main)
