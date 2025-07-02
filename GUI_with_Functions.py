import customtkinter as ctk
import pandas as pd
import numpy as np
from tkinter import filedialog
from docx import Document
from datetime import datetime
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
    evaluate_forecast
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
        # Делим plot_frame на 2 строки и 2 столбца
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
            values=["Сгенерированные данные", "Данные пользователя (выбор через проводник)"],
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
            command = self._on_method_change
        )
        self.method_combo.pack(fill="x", padx=5, pady=5)

        # Длина прогноза (только для FEDOT)
        self.horizon_frame = ctk.CTkFrame(self.controls_frame)
        self.horizon_frame.grid(row=4, column=0, sticky="ew", pady=(0,10))
        ctk.CTkLabel(self.horizon_frame, text="Длина прогноза (шаги):").pack(anchor="w", padx=5, pady=(5,0))
        self.horizon_var = ctk.StringVar(value="10")
        self.horizon_entry = ctk.CTkEntry(self.horizon_frame, textvariable=self.horizon_var)
        self.horizon_entry.pack(fill="x", padx=5, pady=5)


        # Кнопка запуска
        self.run_btn = ctk.CTkButton(
            self.controls_frame,
            text="Запустить",
            command=self._on_run
        )
        self.run_btn.grid(row=5, column=0, pady=5, sticky="ew", padx=5)

        # Скрыть параметры генерации если нужно
        self._on_data_change()
        self._on_method_change(self.method_var.get())

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
        # Статистики выборки (левая колонка, вторая строка)
        self.sample_stats = ctk.CTkLabel(self.plot_frame, text="Статистики выборки:")
        self.sample_stats.grid(row=1, column=0, sticky="nw", padx=10, pady=5)
        # Метрики качества (правая колонка, вторая строка)
        self.metrics = ctk.CTkLabel(self.plot_frame, text="Метрики качества:")
        self.metrics.grid(row=1, column=1, sticky="nw", padx=10, pady=5)

    def _on_data_change(self, *args):
        if self.data_var.get() == "Сгенерированные данные":
            self.gen_params_frame.grid()
        else:
            self.gen_params_frame.grid_remove()

    def _on_method_change(self, selected_method):
        if selected_method == "Модель FEDOT AutoML":
            self.horizon_frame.grid()
        else:
            self.horizon_frame.grid_remove()

    def _on_run(self):
        try:
            split_pct = float(self.split_var.get())
            if not 0 < split_pct < 100:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка ввода", "Введите корректный процент разделения (0-100).")
            return
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
            else:
                path = filedialog.askopenfilename(
                    title="Выбирите CSV-файл",
                    filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")])
                if not path:
                    return # отмена выбора произвольного набора данных
                df = pd.read_csv(path, parse_dates=["timestamp"])
        except Exception as e:
            messagebox.showerror("Ошибка генерации или ввода данных", str(e))
            return

        if self.method_var.get() == "Модель FEDOT AutoML":
            try:
                horizon = int(self.horizon_var.get())
                if horizon <= 0:
                    raise ValueError("Должно быть > 0")
            except Exception:
                messagebox.showerror("Ошибка ввода", "Длина прогноза должна быть целым числом > 0.")
                return
        try:
            stats = describe_timeseries(df)
        except Exception as e:
            messagebox.showerror("Ошибка описания ряда", str(e))
            return
        stats_text = "Статистики выборки:\n"
        for k in ['mean','median','std','min','max','range','quantile_25','quantile_75']:
            stats_text += f"- {k}: {stats.get(k, ''):.4f}\n"
        self.sample_stats.configure(text=stats_text)

        # Разделение и прогноз
        try:
            train_df, test_df = split_train_test(df, split_pct/100)
            method = self.method_var.get()
            if method == "Авторегрессия (statsmodels)":
                pred_df = forecast_with_ar(train_df, test_df)
            elif method == "Модель SARIMA":
                pred_df = forecast_with_sarima(train_df, test_df)
            elif method == "CatBoostRegressor с использованием лагов":
                pred_df = forecast_with_catboost(train_df, test_df)
            elif method == "Модель FEDOT AutoML":
                pred_df = forecast_with_fedot(train_df, test_df, forecast_length = horizon)
            else:
                pred_df = forecast_with_prophet(train_df, test_df)
        except Exception as e:
            messagebox.showerror("Ошибка прогнозировани", str(e))
            return


        # Оценка прогноза
        try:
            metrics = evaluate_forecast(df, pred_df)
        except Exception as e:
            messagebox.showerror("Ошибка оценки прогноза", str(e))
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
        help_win = Toplevel(self)
        help_win.title("Справка")
        help_win.geometry("600x400")
        # Устанавливаем белый фон для окна и фрейма
        help_win.configure(bg="white")
        container = ctk.CTkFrame(help_win, fg_color="white")
        container.pack(fill="both", expand=True)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=3)
        container.grid_rowconfigure(0, weight=1)

        # Оглавление
        style = ttk.Style(help_win)
        style.theme_use('default')
        style.configure("Treeview", background="white", fieldbackground="white")
        tree = ttk.Treeview(container)
        tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        # Заполняем оглавление
        sections = {}
        for i in range(1, 4):
            parent = tree.insert('', 'end', text=f"Раздел {i}")
            for j in range(1, 4):
                tree.insert(parent, 'end', text=f"Подраздел {i}.{j}")
        # Удаляем возможный пустой корневой элемент
        for iid in tree.get_children(''):
            if not tree.item(iid, 'text'):
                tree.delete(iid)

        # Текстовое окно
        text = ctk.CTkTextbox(
            container,
            fg_color="white",    # фон самой рамки
            text_color="black",  # цвет текста
            )
        text.configure(state="disabled")
        text.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        # Отключаем редактирование
        text.configure(state="disabled")

        def on_tree_select(self, event):
        # Получаем выбранный элемент
            sel = tree.selection()
            if not sel:
                return
            item = sel[0]
            title = tree.item(item, "text")  # например, "Раздел 1" или "Подраздел 2.3"

            # Формируем имя файла: удаляем пробелы и точки, добавляем .docx
            # Например, "Раздел 1" → "Раздел1.docx", "Подраздел 2.3" → "Подраздел2_3.docx"
            safe_name = title.replace(" ", "").replace(".", "_")
            filename = f"./help_docs/{safe_name}.docx"

            try:
                # Загружаем документ
                doc = Document(filename)
                content = "\n".join(para.text for para in doc.paragraphs)
            except Exception as e:
                content = f"Не удалось загрузить {filename}:\n{e}"

            # Показываем в текстовом окне
            text.configure(state="normal")
            text.delete("0.0", "end")
            text.insert("0.0", content)
            text.configure(state="disabled")


        tree.bind('<<TreeviewSelect>>', on_tree_select)


if __name__ == "__main__":
    app = TimeSeriesApp()
    app.mainloop()