# Importamos las librerias que utilzaremos
# Si no tienes python instalado te recomendamos que lo descargues de su pagina oficial
# La version de python que usamos es la 3.12.4
# https://www.python.org/downloads/
# Luego instalar las librerias correspondientes
# pip install flet
# pip install tensorflow
# pip install matplotlib
# pip install numpy
# pip install keras
# pip install scikit-learn

import os.path
import threading
import tkinter as tk
import warnings
from tkinter import filedialog

import flet as ft
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import Dense
from keras.src.models import Sequential
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Ignora todas las advertencias de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Final Project of Artificial Inteligence Class
# Developed by Luis Pineda: 2020-0251U
#           && Engel Reyes: 2020-0505U

# Cambia el backend de Matplotlib
# Para permitir solo guardar gráficos directamente en archivos (por ejemplo, PNG, PDF).
# Este backend no abre una ventana para mostrar los gráficos.
plt.switch_backend('Agg')

# Establecer una semilla asegura la reproducibilidad de
# los resultados cuando se utilizan funciones aleatorias
np.random.seed(7)

# Estas variables están inicializadas en None, lo que significa
# que aún no tienen un valor asignado
dataset = X = Y = None
Y_one_hot = history = None
Accuracy = MissClassificationRate = Recall = Specificity = None
Precition = PrecitionNeg = None
TotalCases = None
# Variables inicializadas en cero que estan relacionadas con la matriz de confusión
TP = FN = FP = TN = 0
# Estas variables son el tamaño de cada parte del grafico de pastel
TNPercentage = FNPercentage = TPPercentage = FPPercentage = 25
# Otras variables que se utilizan en el programa
ResultPredict = "There is no prediction"
TrainResult = "Any 1.1"
neuronal_network_filename = "neuronal_network.keras"
precision_plot_filename = "precision-plot.png"

# Variable con los Tipos de funciones de pérdida
lossType = ["binary_crossentropy", "categorical_crossentropy", "hinge", "mean_squared_error",
            "mean_absolute_error", "sparse_categorical_crossentropy"]
# Variable con los Tipos de optimizadores
optimizerType = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
# Variable con los Tipos de métricas
metricType = ["accuracy", "precision", "recall", "mean_squared_error", "mean_absolute_error"]
# Variable con los Modos de activación
activationMode = ["relu", "sigmoid", "tanh", "softmax", "linear"]
# Variable con Valores booleanos
boolType = ["True", "False"]
# Variable con Opciones de género que pueden ser usadas
gender = ["Female", "Male"]
# Variable con los Nombres de las categorías en la matriz de confusión
matrix_str = ["TN", "FN", "TP", "FP"]
# Configura un filtro de entrada
# Este filtro de entrada está diseñado para permitir números decimales
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

    #
    if file_path:
        try:
            # Carga el archivo CSV en un arreglo de NumPy
            dataset = np.loadtxt(file_path, delimiter=",", skiprows=1, encoding="utf-8")

            # Normaliza los valores en la columna 2 y 6
            dataset[:, 2] /= 100
            dataset[:, 6] /= 100000

            # Extrae las características (X) y las etiquetas (Y)
            X = dataset[0:, 0:12]
            Y = dataset[0:, -1]

            # Convierte las etiquetas en one-hot encoding
            Y_one_hot = to_categorical(Y)

            # Comentarios, para verificar el contenido de las matrices y la distribución de categorías
            # unique, counts = numpy.unique(Y, return_counts=True)
            # print("Matriz X \n", X[:, 6])
            # print("Matriz Y \n", Y)
            # print("Cantidad por categoria: \n", dict(zip(unique, counts)))
            # print(Y_one_hot)
        except Exception as e:
            # Maneja errores en la carga del archivo
            print("Error loading CSV file:", e)
            return None
    else:
        return None


# style_frame configura los estilos parael grafico de pastel
style_frame: dict = {
    "expand": True,
    "bgcolor": "#1f2128",
    "border_radius": 10,
    "padding": 20,
}


# Convierte una cadena booleana al numero entero que corresponde y lo retorna
def to_int(value):
    if value == 'True':
        return 1
    elif value == 'False':
        return 0
    else:
        return int(value)


# Este componente consiste en crear un icono centrado dentro de un contenedor circular con un borde negro
def badge(icon, size):
    return ft.Container(
        ft.Icon(icon, color=ft.colors.BLACK),
        width=size,
        height=size,
        border=ft.border.all(1, ft.colors.BLACK),
        border_radius=size / 2,
        bgcolor=ft.colors.WHITE,
    )


# Muestra un grafico de pastel con interactividad mediante el metodo on_chart_event
class Graph(ft.Container):
    def __init__(self):
        # Esto aplica los estilos definidos en style_frame al contenedor.
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

        # Creación del Gráfico de Pastel con secciones basadas en los porcentajes definidos en las variables
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

    # Este método ajusta el radio y el estilo del título de la sección cuando se interactúa
    # con ella (por ejemplo, al pasar el ratón sobre la sección del grafico), mostrando el valor
    # correspondiente (TN, FN, TP, FP).
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

    # Este método retorna el contenido del gráfico, listo para ser utilizado
    # en la interfaz.
    def build(self):
        return self.content


class Home(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page

        # Instancia del gráfico definido en la clase Graph.
        graph: ft.Container = Graph()
        # Tamaño de la fuente para el texto de las tarjetas.
        card_text_size = 15

        #Titulo principal de la pagina
        self.Title = ft.Text(
            "Home",
            size=40,
            text_align="center",
        )

        # Textos para cada una de las métricas que se mostrarán en las tarjetas.
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

        # Creación de Tarjetas
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

        # Contenedor principal de la página que contiene todos los controles
        # (título, tarjetas y gráfico)
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

    # Método para actualizar los textos de las métricas.
    # Cada texto es actualizado con los nuevos valores proporcionados y
    # el componente se actualiza mediante self.update().
    def update_text(self, total_cases, accuracy, missclassificationrate, recall, specificity, precition, negprecition):
        self.total_cases_text.value = f"The total number of cases in the dataset is {total_cases}"
        self.accuracy_text.value = f"The accuracy is {accuracy:.2f}%"
        self.missClassificationRate_text.value = f"The miss classification rate is {missclassificationrate:.2f}%"
        self.recall_text.value = f"The recall is {recall:.2f}%"
        self.specificity_text.value = f"The specificity is {specificity:.2f}%"
        self.precition_text.value = f"The percentage that classifies correctly when predicting positives is {precition:.2f}%"
        self.precitionNeg_text.value = f"The percentage that classifies correctly when predicting negatives is {negprecition:.2f}%"
        self.update()


# Predice nuevos casos utilizando un modelo de red neuronal cargado
# desde un archivo, ese archivo es el que se guarda despues de compilar y entrenar el modelo
# desde el modulo de settings
class PredictNewCase(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page

        # Mostrará el resultado de la predicción
        self.text_result = ft.Text(
            value=ResultPredict,
            color=ft.colors.BLUE
        )

        # Contiene el texto del resultado y un botón para cerrar la ventana emergente
        self.bsContainerColum = ft.Column(
            [
                self.text_result,
                ft.ElevatedButton("Close", on_click=self.close_bs)
            ],
            tight=True,
        )
        self.bsContainer = ft.Container(
                self.bsContainerColum,
                padding=10,
            )
        self.bs = ft.BottomSheet(
            self.bsContainer,
            open=False,
        )
        self.page.overlay.append(self.bs)

        # Campos para rellenar para la nueva prediccion
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
        global ResultPredict
        if os.path.exists(neuronal_network_filename):
            pass
        else:
            ResultPredict = "Please train a model"
            self.show_bs()
            return

        model = load_model(neuronal_network_filename)

        if not self.Age.value.strip() or not self.CreatininePhosphoKinase.value.strip() \
                or not self.EjectionFraction.value.strip() or not self.Platelets.value.strip() \
                or not self.SerumCreatinine.value.strip() or not self.SerumSodium.value.strip() \
                or not self.Time.value.strip():
            ResultPredict = "Please enter all fields"
            self.show_bs()
            return

        # Convertir valores a los tipos de datos adecuados
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

        # Convertir el valor de sexo a un valor numérico
        if sex_value == gender[0]:
            sex_value = 0
        else:
            sex_value = 1

        # Validar que los valores no sean menores que 1
        if (age_value < 1 or creatininePhosphoKinase_value < 1 or ejection_fraction_value < 1
                or platelets_value < 1 or serum_sodium_value < 1 or time_value < 1):
            ResultPredict = "Values cannot be less than 1"
            self.show_bs()
            return

        # Normalizar algunos valores
        creatininePhosphoKinase_value /= 100
        platelets_value /= 100000

        # Crear un array con los datos de entrada
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

        # Predecir utilizando el modelo cargado
        Ynew = model.predict(Xnew)
        newPredictions = [round(x[0]) for x in Ynew]
        print(newPredictions[0])
        print(Ynew)

        # Determinar el resultado de la predicción
        if newPredictions[0] == 1:
            ResultPredict = "The patient will die during the follow-up period"
        else:
            ResultPredict = "The patient will survive"

        print(ResultPredict)

        # Mostrar el resultado en el BottomSheet
        self.show_bs()

    # Cierra el BottomSheet configurando open=False y actualiza la interfaz
    def close_bs(self, e=None):
        self.bs.open = False
        self.bs.update()

    # Muestra el BottomSheet configurando open=True, actualiza el texto
    # del resultado y programa su cierre después de 5 segundos
    def show_bs(self):
        self.bs.open = True
        self.text_result.value = ResultPredict
        self.bs.update()
        threading.Timer(5.0, self.close_bs).start()

    def build(self):
        return self.content


# Convierte una imagen ubicada en file_path a una cadena en formato base64
# Esto lo hicimos ya que si le pasabamos la misma ruta de imagen no se actualizaba la imagen en la interfaz
def image_to_base64(file_path):
    import base64

    if not os.path.exists(file_path):
        return

    with open(file_path, "rb") as image_file:
        image_data = image_file.read()
        encoded_image = base64.b64encode(image_data)
        base64_string = encoded_image.decode('utf-8')
    return base64_string


# Muestra una imagen en una interfaz de usuario
class ShowImage(ft.UserControl):
    def __init__(self, page):
        super().__init__(expand=True)
        self.page = page

        # Utiliza la función image_to_base64 para obtener la representación en base64
        # de la imagen especificada por precision_plot_filename
        imageBase64 = image_to_base64(precision_plot_filename)

        # Si imageBase64 es None, indica que no se pudo cargar la imagen desde
        # el archivo especificado, por lo que se usa una imagen de respaldo, el cual es error 404
        if imageBase64 is not None:
            self.image = ft.Image(
                src_base64=imageBase64,
                fit=ft.ImageFit.CONTAIN,
            )
        else:
            self.image = ft.Image(
                src="oops.svg",
                fit=ft.ImageFit.CONTAIN,
            )

        self.content = ft.ResponsiveRow(
            controls=[
                self.image
            ]
        )

    def build(self):
        return self.content

    def update_image(self):
        global precision_plot_filename

        imageBase64 = image_to_base64(precision_plot_filename)
        # Actualiza el atributo src_base64 de self.image con la nueva representación base64
        self.image.src_base64 = imageBase64
        # Llama al método update() para reflejar los cambios en la interfaz de usuario
        self.update()


# La clase Configuration que utiliza es para configurar diferentes aspectos relacionados
# con la compilación y el entrenamiento del modelo de aprendizaje automático.
class Configuration(ft.UserControl):
    def __init__(self, page, home, showimage):
        super().__init__(expand=True)
        self.page = page
        self.home = home
        self.showImage = showimage

        # Texto para mostrar el resultado del entrenamiento
        self.text_result = ft.Text(
            value=TrainResult,
            color=ft.colors.BLUE
        )

        # Botón para cerrar el banner
        self.text_button = ft.TextButton(
            "Close",
            on_click=self.close_banner,
        )

        # Banner para mostrar mensajes, como resultados del entrenamiento
        self.page.banner = ft.Banner(
            bgcolor=ft.colors.AMBER_100,
            content=self.text_result,
            actions=[
                self.text_button,
            ],
        )

        # Configuración de la función de pérdida
        self.lossfunction = ft.Dropdown(
            label="Loss Function",
            value="binary_crossentropy",
            hint_text="Choose a loss function",
            border_color="blue",
            options=[
                ft.dropdown.Option(opt) for opt in lossType
            ]
        )

        # Configuración del optimizador
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

        # Configuración de las métricas
        self.metrics = ft.Dropdown(
            label="Metrics",
            value="accuracy",
            hint_text="Choose a metric",
            border_color="blue",
            options=[
                ft.dropdown.Option(opt) for opt in metricType
            ]
        )

        # Configuración del split de validación
        self.validationsplit = ft.Slider(
            value=10,
            min=10,
            max=100,
            divisions=9,
            label="{value}%"
        )

        # Configuración de las épocas de entrenamiento
        self.epochs = ft.TextField(label="Epochs", border_color="blue",
                                   value="150",
                                   input_filter=ft.NumbersOnlyInputFilter(),
                                   max_length=3)

        # Configuración del tamaño del batch
        self.batchsize = ft.TextField(label="Batch Size", border_color="blue",
                                      input_filter=ft.NumbersOnlyInputFilter(),
                                      value="10",
                                      max_length=2)

        # Módulo para abrir un archivo CSV
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

        # Módulo para configurar la compilación del modelo
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

        # Módulo para configurar el entrenamiento del modelo
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

        # Contenedor principal con filas responsivas que contienen los módulos de configuración
        self.content = ft.ResponsiveRow(
            controls=[
                self.OpenFile,
                self.CompileConfigModule,
                self.TrainConfigModule,
            ]
        )

    # Este método se encarga de compilar y entrenar el modelo de red neuronal
    def TrainAction(self, e):
        global history

        # Obtenemos los valores de los parámetros de configuración
        loss = self.lossfunction.value
        optimizer = self.optimizer.value
        metrics = self.metrics.value
        validationsplit = self.validationsplit.value / 100
        batchsize_value = self.batchsize.value.strip()
        epochs_value = self.epochs.value.strip()

        # Definimos algunas variables globales para almacenar resultados
        global TotalCases, Accuracy, MissClassificationRate, Recall, Specificity, Precition, PrecitionNeg
        global TP, FN, FP, TN, TrainResult

        # Verificamos si se ha cargado un conjunto de datos
        if dataset is None:
            TrainResult = "Please load a dataset"
            self.show_banner_click()
            return
        # Verificamos que el split de validación no sea 100%
        elif validationsplit == 1:
            TrainResult = "The validation split cannot be 100%"
            self.show_banner_click()
            return
        # Verificamos que se haya ingresado el tamaño del batch y las épocas
        elif not batchsize_value:
            TrainResult = "Please enter the batch size"
            self.show_banner_click()
            return
        elif not epochs_value:
            TrainResult = "Please enter the epochs"
            self.show_banner_click()
            return

        # Convertimos los valores del tamaño del batch y las épocas a enteros
        batchsize = int(self.batchsize.value)
        epochs = int(self.epochs.value)

        # Verificamos que el tamaño del batch y las épocas sean mayores que cero
        if batchsize < 1 or epochs < 1:
            TrainResult = "The batch size or epochs cannot be 0"
            self.show_banner_click()
            return

        # Mostramos un mensaje de que se está compilando y entrenando el modelo
        TrainResult = "Compiling and training the model"
        self.show_banner_click()

        # Definimos la arquitectura del modelo
        model = Sequential()
        model.add(Dense(units=12, input_dim=12, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=5, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        # Compilamos el modelo con la función de pérdida, optimizador y métricas especificadas
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        # Entrenamos el modelo con los datos X, Y, utilizando el split de validación, épocas y tamaño del lote
        history = model.fit(X, Y, validation_split=validationsplit, epochs=epochs, batch_size=batchsize,
                            verbose=0)
        # print(history.history.keys())

        # Guardamos el modelo entrenado en un archivo
        # Esto lo utilizamos en el modulod de predecir un nuevo caso
        model.save(filepath=neuronal_network_filename, overwrite=True)

        # Realizamos predicciones con el modelo entrenado
        predictions = model.predict(X)
        rounded = [round(x[0]) for x in predictions]

        # Calculamos la matriz de confusión y la guardamos
        matrix = confusion_matrix(Y, rounded)
        # print(matrix)

        # Creamos un gráfico con las métricas de entrenamiento y validación
        plt.clf()
        plt.plot(history.history[metrics])
        plt.plot(history.history[f'val_{metrics}'])
        plt.title(f'{metrics} model')
        plt.ylabel(f'{metrics}')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Test'], loc='upper left')
        plt.savefig(precision_plot_filename)

        # Actualizamos las variables globales con los resultados de la matriz de confusión
        TN = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]
        TP = matrix[1, 1]

        TotalCases = TP + FN + FP + TN
        Accuracy = ((TP + TN) / TotalCases) * 100
        MissClassificationRate = ((FP + FN) / TotalCases) * 100
        Recall = (TP / (TP + FN)) * 100
        Specificity = (TN / (TN + FP)) * 100

        # Calculamos la precisión de los positivos y negativos clasificados correctamente
        if TP != 0 or FP != 0:
            Precition = (TP / (TP + FP)) * 100
        else:
            Precition = 0

        if TN != 0 or FN != 0:
            PrecitionNeg = (TN / (TN + FN)) * 100
        else:
            PrecitionNeg = 0

        # Actualizamos la interfaz de usuario de la página principal con los resultados
        self.home.update_text(TotalCases, Accuracy, MissClassificationRate, Recall, Specificity, Precition,
                              PrecitionNeg)

        # Actualizamos la imagen mostrada en la interfaz con el gráfico generado
        self.showImage.update_image()

        # Mostramos que el entrenamiento ha sido completado
        TrainResult = "Completed"
        self.show_banner_click()

    # Simplemente cierra el banner de la interfaz de usuario al establecer self.page.banner.open
    # en False y luego actualiza la página para reflejar este cambio
    def close_banner(self, e=None):
        self.page.banner.open = False
        self.page.update()

    # Se encarga de mostrar el banner de la interfaz de usuario con un icono y colores
    # específicos dependiendo del estado de TrainResult. Si TrainResult es "Compiling
    # and training the model" o "Completed", se muestra un icono de verificación verde
    # y se establece el color del texto y botón en verde. En caso contrario, se muestra
    # un icono de información ámbar y se establece el color del texto y botón en azul
    def show_banner_click(self):
        if TrainResult == "Compiling and training the model" or TrainResult == "Completed":
            self.page.banner.leading = ft.Icon(ft.icons.CHECK_OUTLINED, color=ft.colors.GREEN, size=40)
            self.page.banner.bgcolor = ft.colors.GREEN_100
            self.text_result.color = ft.colors.GREEN
            self.text_button.style = ft.ButtonStyle(
                color={
                    ft.MaterialState.HOVERED: ft.colors.BLACK,
                    ft.MaterialState.FOCUSED: ft.colors.BLUE,
                    ft.MaterialState.DEFAULT: ft.colors.GREEN,
                }
            )
        else:
            self.page.banner.leading = ft.Icon(ft.icons.INFO_OUTLINED, color=ft.colors.AMBER, size=40)
            self.page.banner.bgcolor = ft.colors.AMBER_100
            self.text_result.color = ft.colors.BLUE
            self.text_button.style = ft.ButtonStyle(
                color={
                    ft.MaterialState.HOVERED: ft.colors.BLACK,
                    ft.MaterialState.FOCUSED: ft.colors.BLACK,
                    ft.MaterialState.DEFAULT: ft.colors.BLUE,
                }
            )

        self.page.banner.open = True
        self.text_result.value = TrainResult
        self.page.update()
        threading.Timer(4.0, self.close_banner).start()

    def build(self):
        return self.content


# Funcion principal donde cargamos todos los modulos
def main(page: ft.Page):
    # Actualiza la página para asegurar que esté lista para ser mostrada.
    page.update()

    # Es una función interna que se llama cuando se cambia la pestaña
    # en la barra de navegación (on_change=changeTab).
    def changeTab(e):
        my_index = e.control.selected_index
        home.visible = True if my_index == 0 else False
        predictNewCase.visible = True if my_index == 1 else False
        configuration.visible = True if my_index == 2 else False
        showImage.visible = True if my_index == 3 else False
        page.update()

    # Se establecen diversas propiedades de la ventana principal
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
            ft.NavigationDestination(
                icon=ft.icons.AUTO_GRAPH,
                label="Graphic",
            ),
        ],
    )

    # Son instancias de las clases que representan la interfaz de usuario
    home = Home(page)
    home.visible = True
    predictNewCase = PredictNewCase(page)
    predictNewCase.visible = False
    showImage = ShowImage(page)
    showImage.visible = False
    configuration = Configuration(page, home, showImage)
    configuration.visible = False

    # Agrega todas las vistas
    page.add(
        home,
        predictNewCase,
        configuration,
        showImage,
    )
    # Establece el modo de tema oscuro para la aplicación,
    # lo que afectará a todos los elementos de la interfaz que soporten este modo.
    page.theme_mode = ft.ThemeMode.DARK


# Inicia la aplicación utilizando la función main como el punto de entrada principal.
ft.app(main)
# ft.app(main, view=ft.AppView.WEB_BROWSER)
