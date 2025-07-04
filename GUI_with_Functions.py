import customtkinter as ctk
import multiprocessing
multiprocessing.freeze_support()
multiprocessing.set_start_method('spawn', force=True)
import pandas as pd
import numpy as np
import mammoth
import threading
from tkinterweb import HtmlFrame
from tkinter import messagebox, Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from Functions import (
    forecast_with_fedot,
    generate_timeseries,
    split_train_test,
    describe_timeseries,
    forecast_with_ar,
    forecast_with_sarima,
    forecast_with_catboost,
    forecast_with_prophet,
    evaluate_forecast,
    forecast_with_fedot
)

# Настройка темы
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TimeSeriesApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Анализ временных рядов")
        self.geometry("1200x840")
        self.resizable(False, False)

        # Справочная кнопка
        self.help_btn = ctk.CTkButton(self, text="?", width=30, height=30,
                                       corner_radius=15, command=self._open_help)
        self.help_btn.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")

         # Настройка сетки главного окна
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Левая область: график и характеристики
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=3)
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(1, weight=1)

        self._init_plot()
        self._init_stats()

        # Правая область: контролы
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        for i in range(6):
            self.controls_frame.grid_rowconfigure(i, weight=0)
        self.controls_frame.grid_rowconfigure(5, weight=1)
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # Выбор данных
        data_frame = ctk.CTkFrame(self.controls_frame)
        data_frame.grid(row=0, column=0, sticky="ew", pady=(0,5))
        ctk.CTkLabel(data_frame, text="Выбор данных:").pack(anchor="w", padx=5, pady=(5,0))
        self.data_var = ctk.StringVar(value="Сгенерированные данные")
        self.data_combo = ctk.CTkComboBox(
            data_frame,
            values=["Сгенерированные данные", "Курс австраллийского доллара с 2005 по 2025", "Температура в Хиросиме летом 1945"],
            variable=self.data_var
        )
        self.data_combo.pack(fill="x", padx=5, pady=5)
        self.data_var.trace_add('write', self._on_data_change)

        # Параметры генерации
        self.gen_params_frame = ctk.CTkFrame(self.controls_frame)
        self.gen_params_frame.grid(row=1, column=0, sticky="ew", pady=(0,10))
        params = [
            ("Начальная дата", "2025-01-01"),
            ("Конечная дата", "2025-12-31"),
            ("Коэффициент тренда", "0.0"),
            ("Амплитуда сезонности", "1.0"),
            ("Длина одного цикла сезонности (в днях)", "30"),
            ("Уровень шума (стандартное отклонение)", "0.1"),
        ]
        self.gen_entries = {}
        for label, default in params:
            container = ctk.CTkFrame(self.gen_params_frame)
            container.pack(fill="x", padx=5, pady=3)
            ctk.CTkLabel(container, text=label+":").pack(side="left")
            var = ctk.StringVar(value=default)
            entry = ctk.CTkEntry(container, textvariable=var)
            entry.pack(side="right", fill="x", expand=True)
            self.gen_entries[label] = var

        # Выбор доли разбиения датасета на тренировочный и проверочный
        split_frame = ctk.CTkFrame(self.controls_frame)
        split_frame.grid(row=2, column=0, sticky="ew", pady=(0,10))
        ctk.CTkLabel(split_frame, text="Доля обучения (%):").pack(anchor="w", padx=5, pady=(5,0))
        self.split_var = ctk.StringVar(value="70.0")
        self.split_entry = ctk.CTkEntry(split_frame, textvariable=self.split_var)
        self.split_entry.pack(fill="x", padx=5, pady=5)

        # Выбор метода анализа
        method_frame = ctk.CTkFrame(self.controls_frame)
        method_frame.grid(row=3, column=0, sticky="ew", pady=(0,10))
        ctk.CTkLabel(method_frame, text="Метод анализа:").pack(anchor="w", padx=5, pady=(5,0))
        self.method_var = ctk.StringVar(value="Авторегрессия (statsmodels)")
        self.method_combo = ctk.CTkComboBox(
            method_frame,
            values=[
                "Авторегрессия (statsmodels)",
                "Модель SARIMA",
                "CatBoostRegressor с использованием лагов",
                "Модель Prophet",
                "Модель FEDOT AutoML"
            ],
            variable = self.method_var,
        )
        self.method_combo.pack(fill="x", padx=5, pady=5)


        # Кнопка запуска
        self.run_btn = ctk.CTkButton(
            self.controls_frame,
            text="Запустить",
            command=self._on_run
        )
        self.run_btn.grid(row=4, column=0, pady=5, sticky="ew", padx=5)

        # Скрыть параметры генерации если нужно
        self._on_data_change()

        self.help_btn.lift()

    def _init_plot(self):
        # График занимает всю первую строку, оба столбца
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Временной ряд")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, columnspan=2, sticky="nsew")

    def _init_stats(self):
        self.sample_stats = ctk.CTkLabel(self.plot_frame, text="Статистики выборки:")
        self.sample_stats.grid(row=1, column=0, sticky="nw", padx=10, pady=5)
        self.metrics = ctk.CTkLabel(self.plot_frame, text="Метрики качества:")
        self.metrics.grid(row=1, column=1, sticky="nw", padx=10, pady=5)

    def _on_data_change(self, *args):
        if self.data_var.get() == "Сгенерированные данные":
            self.gen_params_frame.grid()
        else:
            self.gen_params_frame.grid_remove()


    def _on_run(self):
        try:
            split_pct = float(self.split_var.get())
            if not 0 < split_pct < 100:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка ввода", "Введите корректный процент разделения (0-100).")
            return

        # Блокирование кнопки, чтобы пользователь не кликал не неё
        self.run_btn.configure(state='disabled') 

        # Запуск фонового потока
        threading.Thread(target=self._background_run, daemon=True).start()

    def _background_run(self):
        try:
            if self.data_var.get() == "Сгенерированные данные":
                params = { 
                    'start_date': self.gen_entries['Начальная дата'].get(),
                    'end_date': self.gen_entries['Конечная дата'].get(),
                    'trend_coef': float(self.gen_entries['Коэффициент тренда'].get()),
                    'seasonality_amplitude': float(self.gen_entries['Амплитуда сезонности'].get()),
                    'season_length': int(self.gen_entries['Длина одного цикла сезонности (в днях)'].get()),
                    'noise_level': float(self.gen_entries['Уровень шума (стандартное отклонение)'].get())
                }
                df = generate_timeseries(**params)
            elif self.data_var.get() == "Курс австраллийского доллара с 2005 по 2025":
                df = pd.read_csv("australlian_dollar_values.csv", parse_dates=["timestamp"])
            else:
                df = pd.read_csv("hiroshima_summer_1945_temps.csv", parse_dates=["timestamp"])
            
            stats = describe_timeseries(df)


            train_df, test_df = split_train_test(df, float(self.split_var.get())/100)
            method = self.method_var.get()
            if method == "Авторегрессия (statsmodels)":
                pred_df = forecast_with_ar(train_df, test_df)
            elif method == "Модель SARIMA":
                pred_df = forecast_with_sarima(train_df, test_df)
            elif method == "CatBoostRegressor с использованием лагов":
                pred_df = forecast_with_catboost(train_df, test_df)
            elif method == "Модель FEDOT AutoML":
                pred_df = forecast_with_fedot(train_df, test_df)
            else:
                pred_df = forecast_with_prophet(train_df, test_df)

            metrics = evaluate_forecast(df, pred_df)

            # Передача результата в mainloop
            self.after(0, lambda: self._update_ui(df, stats, pred_df, metrics))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ошибка прогнозирования", str(e)))
            return

        finally:
            # Разблокирование кнопки
            self.after(0, lambda: self.run_btn.configure(state='normal'))

    def _update_ui(self, df, stats, pred_df, metrics):
        stats_text = "Статистики выборки:\n"
        for k in ['mean','median','std','var','min','max','range','quantile_25','quantile_75']:
            stats_text += f"- {k}: {stats.get(k, ''):.4f}\n"
        self.sample_stats.configure(text=stats_text)
       

        metrics_text = "Метрики качества:\n"
        for m in ['MAE','RMSE','MAPE','R2']:
            metrics_text += f"- {m}: {metrics.get(m, 0):.4f}\n"
        self.metrics.configure(text=metrics_text)

        # Обновление графика
        self.ax.clear()
        self.ax.plot(df['timestamp'], df['value'], label='Фактические данные')
        self.ax.plot(pred_df['timestamp'], pred_df['value'], label='Предсказание')
        self.ax.legend()
        self.ax.set_title('Фактические данные и Предсказание')
        self.canvas.draw()

    def _open_help(self):
        # создаём окно справки
        help_win = Toplevel()
        help_win.title("Справка")
        help_win.geometry("800x600")
        help_win.rowconfigure(0, weight=1)
        help_win.columnconfigure(0, weight=1)
        
        # контейнер для всего содержимого
        help_container = ctk.CTkFrame(help_win)
        help_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        help_container.rowconfigure(0, weight=1)
        help_container.columnconfigure(0, weight=1)  # для дерева
        help_container.columnconfigure(1, weight=3)  # для текста
        
        # 1) Дерево оглавления
        tree = ttk.Treeview(help_container)
        tree.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        
        # 2) HTML-виджет для отображения содержимого .docx
        html_frame = HtmlFrame(help_container)
        html_frame.grid(row=0, column=1, sticky="nsew")
        
        # --- Заполнение оглавления ---
        # Здесь указываем разделы, которые у вас лежат в отдельных .docx
        tree.insert("", "end", text="Приветственное слово")
        root2 = tree.insert("", "end", text="Модели анализа")
        tree.insert(root2, "end", text="Авторегрессия")
        tree.insert(root2, "end", text="SARIMA")
        tree.insert(root2, "end", text="Градиентный бустинг")
        tree.insert(root2, "end", text="Prophet")
        tree.insert(root2, "end", text="FEDOT AutoML")
        root3 = tree.insert("", "end", text="Характеристики выборки")
        tree.insert(root3, "end", text="Дисперсия")
        tree.insert(root3, "end", text="Медиана")
        tree.insert(root3, "end", text="Стандартное отклонение")
        root4 = tree.insert("", "end", text="Метрики качества прогноза")
        tree.insert(root4, "end", text="MAE")

        # удаляем пустые узлы, если нужно
        for iid in tree.get_children(""):
            if not tree.item(iid, "text"):
                tree.delete(iid)

        # Функция-обработчик выбора раздела
        def on_tree_select(event):
            sel = tree.selection()
            if not sel:
                return
            title = tree.item(sel[0], "text")
            safe   = title.replace(" ", "").replace(".", "_")
            filename = f"{safe}.docx"

            try:
                with open(filename, "rb") as docx_file:
                    result = mammoth.convert_to_html(
                        docx_file,
                        convert_image=mammoth.images.data_uri
                    )
                    html = result.value
            except Exception as e:
                html = f"<h2>Ошибка загрузки {filename}</h2><pre>{e}</pre>"

            html_frame.load_html(html)

        # Привязываем событие
        tree.bind("<<TreeviewSelect>>", on_tree_select)



if __name__ == "__main__":
    app = TimeSeriesApp()
    app.mainloop()
