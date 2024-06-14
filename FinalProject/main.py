import tkinter as tk
import threading
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from keras.src.saving import load_model
from sklearn.metrics import confusion_matrix
import flet as ft

# Final Project of Artificial Inteligence Class
# Developed by Luis Pineda: 2020-0251U
#           && Engel Reyes: 2020-0505U

# Cambia el backend de Matplotlib
plt.switch_backend('Agg')

np.random.seed(7)

dataset = X = Y = None
Y_one_hot = history = None
Accuracy = MissClassificationRate = Recall = Specificity = None
Precition = PrecitionNeg = None
TotalCases = None
TP = FN = FP = TN = 0
TNPercentage = FNPercentage = TPPercentage = FPPercentage = 25
ResultPredict = "There is no prediction"
neuronal_network_filename = "neuronal_network.keras"
precision_plot_filename = "precision-plot.png"

lossType = ["binary_crossentropy", "categorical_crossentropy", "hinge", "mean_squared_error",
            "mean_absolute_error", "sparse_categorical_crossentropy"]
optimizerType = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
metricType = ["accuracy", "precision", "recall", "f1_score", "mean_squared_error", "mean_absolute_error"]
activationMode = ["relu", "sigmoid", "tanh", "softmax", "linear"]
boolType = ["True", "False"]
gender = ["Female", "Male"]
matrix_str = ["TN", "FN", "TP", "FP"]
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


style_frame: dict = {
    "expand": True,
    "bgcolor": "#1f2128",
    "border_radius": 10,
    "padding": 20,
}


def to_int(value):
    if value == 'True':
        return 1
    elif value == 'False':
        return 0
    else:
        return int(value)


def badge(icon, size):
    return ft.Container(
        ft.Icon(icon, color=ft.colors.BLACK),
        width=size,
        height=size,
        border=ft.border.all(1, ft.colors.BLACK),
        border_radius=size / 2,
        bgcolor=ft.colors.WHITE,
    )


class Graph(ft.Container):
    def __init__(self):
        super().__init__(**style_frame)
        self.normal_radius = 80
        self.hover_radius = 90
        self.normal_title_style = ft.TextStyle(
            size=16, color=ft.colors.BLACK, weight=ft.FontWeight.BOLD
        )
        self.hover_title_style = ft.TextStyle(
            size=16,
            color=ft.colors.WHITE,
            weight=ft.FontWeight.BOLD,
            shadow=ft.BoxShadow(blur_radius=2, color=ft.colors.BLACK),
        )
        self.normalBadgeSize = 30
        self.content = ft.PieChart(
            sections=[
                ft.PieChartSection(
                    TNPercentage,
                    title=matrix_str[0],
                    title_style=self.normal_title_style,
                    color=ft.colors.RED,
                    radius=self.normal_radius,
                    badge=badge(ft.icons.CHECK, self.normalBadgeSize),
                    badge_position=1
                ),
                ft.PieChartSection(
                    FNPercentage,
                    title=matrix_str[1],
                    title_style=self.normal_title_style,
                    color=ft.colors.BLUE,
                    radius=self.normal_radius,
                    badge=badge(ft.icons.CLOSE_OUTLINED, self.normalBadgeSize),
                    badge_position=1
                ),
                ft.PieChartSection(
                    TPPercentage,
                    title=matrix_str[2],
                    title_style=self.normal_title_style,
                    color=ft.colors.ORANGE,
                    radius=self.normal_radius,
                    badge=badge(ft.icons.CHECK, self.normalBadgeSize),
                    badge_position=1
                ),
                ft.PieChartSection(
                    FPPercentage,
                    title=matrix_str[3],
                    title_style=self.normal_title_style,
                    color=ft.colors.GREEN,
                    radius=self.normal_radius,
                    badge=badge(ft.icons.CLOSE_OUTLINED, self.normalBadgeSize),
                    badge_position=1
                ),
            ],
            sections_space=5,
            center_space_radius=0,
            on_chart_event=self.on_chart_event,
            expand=True,
        )

    def on_chart_event(self, e):
        for idx, section in enumerate(self.content.sections):
            if idx == e.section_index:
                section.radius = self.hover_radius
                section.title_style = self.hover_title_style
                if idx == 0:
                    section.title = str(TN)
                elif idx == 1:
                    section.title = str(FN)
                elif idx == 2:
                    section.title = str(TP)
                elif idx == 3:
                    section.title = str(FP)
            else:
                section.radius = self.normal_radius
                section.title_style = self.normal_title_style
                section.title = matrix_str[idx]
        self.content.update()

    def build(self):
        return self.content


class Home(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page
        graph: ft.Container = Graph()
        card_text_size = 15
        self.Title = ft.Text(
            "Home",
            size=40,
            text_align="center",
        )
        self.total_cases_text = ft.Text(
            f"The total number of cases in the dataset is {TotalCases}",
            size=card_text_size,
        )
        self.accuracy_text = ft.Text(
            f"The accuracy is {Accuracy}",
            size=card_text_size,
        )
        self.missClassificationRate_text = ft.Text(
            f"The miss classification rate is {MissClassificationRate}",
            size=card_text_size,
        )
        self.recall_text = ft.Text(
            f"The recall is {Recall}",
            size=card_text_size,
        )
        self.specificity_text = ft.Text(
            f"The specificity is {Specificity}",
            size=card_text_size,
        )
        self.precition_text = ft.Text(
            f"The percentage that classifies correctly when predicting positives is {Precition}",
            size=card_text_size,
        )
        self.precitionNeg_text = ft.Text(
            f"The percentage that classifies correctly when predicting negatives is {PrecitionNeg}",
            size=card_text_size,
        )
        card_data = [
            ("Total Cases", self.total_cases_text),
            ("Accuracy", self.accuracy_text),
            ("Miss Classification Rate", self.missClassificationRate_text),
            ("Recall", self.recall_text),
            ("Specificity", self.specificity_text),
            ("Precition", self.precition_text),
            ("Precition of Negative ", self.precitionNeg_text)
        ]
        cards = []
        for title, subtitle in card_data:
            card = ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.ListTile(
                                leading=ft.Icon(ft.icons.ALBUM),
                                title=ft.Text(title),
                                subtitle=subtitle
                            ),
                        ]
                    )
                ),
            )
            cards.append(card)

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
                    cards[0],
                    cards[1],
                    cards[2],
                    cards[3],
                    cards[4],
                    cards[5],
                    cards[6],
                    graph
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

    def update_text(self, total_cases, accuracy, missclassificationrate, recall, specificity, precition, negprecition):
        self.total_cases_text.value = f"The total number of cases in the dataset is {total_cases}"
        self.accuracy_text.value = f"The accuracy is {accuracy:.2f}%"
        self.missClassificationRate_text.value = f"The miss classification rate is {missclassificationrate:.2f}%"
        self.recall_text.value = f"The recall is {recall:.2f}%"
        self.specificity_text.value = f"The specificity is {specificity:.2f}%"
        self.precition_text.value = f"The percentage that classifies correctly when predicting positives is {precition:.2f}%"
        self.precitionNeg_text.value = f"The percentage that classifies correctly when predicting negatives is {negprecition:.2f}%"
        self.update()


class PredictNewCase(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page
        self.text_result = ft.Text(
            value=ResultPredict,
            color=ft.colors.BLUE
        )
        self.page.banner = ft.Banner(
            bgcolor=ft.colors.AMBER_100,
            leading=ft.Icon(ft.icons.INFO_OUTLINED, color=ft.colors.AMBER, size=40),
            content=self.text_result,
            actions=[
                ft.TextButton("Close", on_click=self.close_banner),
            ],
        )
        self.Age = ft.TextField(label="Age", border_color="#a0a1a4",
                                helper_text="Age of the patient",
                                value="75",
                                hint_text="40-95",
                                input_filter=ft.NumbersOnlyInputFilter(),
                                max_length=2,
                                )
        self.Anaemia = ft.Dropdown(
            label="Anaemia",
            helper_text="Decrease of red blood cells or hemoglobin",
            value="False",
            hint_text="Yes or No",
            border_color="#a0a1a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
        )
        self.CreatininePhosphoKinase = ft.TextField(
            label="Creatinine PhosphoKinase (CPK)",
            helper_text="Level of the CPK enzyme in the blood (mcg/L)",
            value="582",
            hint_text="23-7,861",
            border_color="#a4a4a4",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=4,
        )
        self.Diabetes = ft.Dropdown(
            label="Diabetes",
            helper_text="If the patient has diabetes",
            value="False",
            hint_text="Yes or No",
            border_color="#a4a4a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
        )
        self.EjectionFraction = ft.TextField(
            label="Ejection fraction", border_color="#a4a4a4",
            helper_text="Percentage of blood leaving the heart at each contraction",
            value="20",
            hint_text="14-80",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=2,
        )
        self.HighBloodPressure = ft.Dropdown(
            label="High blood pressure",
            helper_text="If a patient has hypertension",
            value="True",
            hint_text="Yes or No",
            border_color="#a4a4a4",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
        )
        self.Platelets = ft.TextField(
            label="Platelets", border_color="#a4a4a4",
            helper_text="Platelets in the blood (kiloplatelets/mL)",
            value="265000",
            hint_text="25,100-850,000",
            input_filter=decimalFilter,
            max_length=9,
        )
        self.SerumCreatinine = ft.TextField(
            label="Serum creatinine", border_color="#a4a4a4",
            helper_text="Level of creatinine in the blood (mg/dL)",
            value="1.9",
            hint_text="0.50-9.40",
            input_filter=decimalFilter,
            max_length=4,
        )
        self.SerumSodium = ft.TextField(
            label="Serum sodium", border_color="#f9f9f9",
            helper_text="Level of sodium in the blood (mEq/L)",
            value="130",
            hint_text="114-148",
            input_filter=ft.NumbersOnlyInputFilter(),
            max_length=3,
        )
        self.Sex = ft.Dropdown(
            label="Sex",
            helper_text="Woman or man",
            value=gender[1],
            hint_text="Gender",
            border_color="#f9f9f9",
            options=[
                ft.dropdown.Option(opt) for opt in gender
            ],
        )
        self.Smoking = ft.Dropdown(
            label="Smoking",
            helper_text="If the patient smokes",
            value="False",
            hint_text="Yes or No",
            border_color="#f9f9f9",
            options=[
                ft.dropdown.Option(opt) for opt in boolType
            ],
        )
        self.Time = ft.TextField(
            label="Time (Days)", border_color="#f9f9f9",
            helper_text="Follow-up period",
            value="4",
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
                                    self.CreatininePhosphoKinase,
                                    self.Diabetes,
                                ]
                            ),
                            ft.Column(
                                col={"sm": 4},
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=30,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    self.EjectionFraction,
                                    self.HighBloodPressure,
                                    self.Platelets,
                                    self.SerumCreatinine,
                                ]
                            ),
                            ft.Column(
                                col={"sm": 4},
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=30,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    self.SerumSodium,
                                    self.Sex,
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

    def PredictNewCase(self, e):
        model = load_model(neuronal_network_filename)

        if model is None:
            return

        if not self.Age.value.strip() or not self.CreatininePhosphoKinase.value.strip()\
                or not self.EjectionFraction.value.strip() or not self. Platelets.value.strip()\
                or not self.SerumCreatinine.value.strip() or not self.SerumSodium.value.strip()\
                or not self.Time.value.strip():
            return

        age_value = int(self.Age.value)
        anaemia_value = to_int(self.Anaemia.value)
        creatininePhosphoKinase_value = int(self.CreatininePhosphoKinase.value)
        diabetes_value = to_int(self.Diabetes.value)
        ejection_fraction_value = int(self.EjectionFraction.value)
        high_blood_pressure_value = to_int(self.HighBloodPressure.value)
        platelets_value = float(self.Platelets.value)
        serum_creatinine_value = float(self.SerumCreatinine.value)
        serum_sodium_value = int(self.SerumSodium.value)
        sex_value = self.Sex.value
        smoking_value = to_int(self.Smoking.value)
        time_value = int(self.Time.value)
        global ResultPredict

        if sex_value == gender[0]:
            sex_value = 0
        else:
            sex_value = 1

        if (age_value < 1 or creatininePhosphoKinase_value < 1 or ejection_fraction_value < 1
                or platelets_value < 1 or serum_sodium_value < 1 or time_value < 1):
            return

        creatininePhosphoKinase_value /= 100
        platelets_value /= 100000

        Xnew = np.array(
            [
                [
                    age_value,
                    anaemia_value,
                    creatininePhosphoKinase_value,
                    diabetes_value,
                    ejection_fraction_value,
                    high_blood_pressure_value,
                    platelets_value,
                    serum_creatinine_value,
                    serum_sodium_value,
                    sex_value,
                    smoking_value,
                    time_value,
                ]
            ]
        )

        Ynew = model.predict(Xnew)
        newPredictions = [round(x[0]) for x in Ynew]
        print(newPredictions[0])
        print(Ynew)

        if newPredictions[0] == 1:
            ResultPredict = "The patient died during the follow-up period"
        else:
            ResultPredict = "The patient survived"

        self.show_banner_click()

    def close_banner(self, e=None):
        self.page.banner.open = False
        self.page.update()

    def show_banner_click(self):
        self.page.banner.open = True
        self.text_result.value = ResultPredict
        self.page.update()
        threading.Timer(5.0, self.close_banner).start()

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
        batchsize_value = self.batchsize.value.strip()
        epochs_value = self.epochs.value.strip()

        global TotalCases, Accuracy, MissClassificationRate, Recall, Specificity, Precition, PrecitionNeg
        global TP, FN, FP, TN

        if dataset is None or validationsplit == 1:
            return

        if not batchsize_value or not epochs_value:
            return

        batchsize = int(self.batchsize.value)
        epochs = int(self.epochs.value)
        if batchsize < 1 or epochs < 1:
            return

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

        matrix = confusion_matrix(Y, rounded)
        #print(matrix)

        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Modelo de Exactitud')
        plt.ylabel('Exactitud')
        plt.xlabel('Epoca')
        plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
        plt.savefig(precision_plot_filename)

        TN = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]
        TP = matrix[1, 1]

        TotalCases = TP + FN + FP + TN
        Accuracy = ((TP + TN) / TotalCases) * 100
        MissClassificationRate = ((FP + FN) / TotalCases) * 100
        Recall = (TP / (TP + FN)) * 100
        Specificity = (TN / (TN + FP)) * 100

        #Precision de positivos que clasifica correctamente
        Precition = (TP / (TP + FP)) * 100

        # Precision de negativos que clasifica correctamente
        PrecitionNeg = (TN / (TN + FN)) * 100

        self.home.update_text(TotalCases, Accuracy, MissClassificationRate, Recall, Specificity, Precition,
                              PrecitionNeg)

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
    page.window_height = 900
    page.window_min_width = 900
    page.window_min_height = 900
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
