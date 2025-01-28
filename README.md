# My_AI_projects
Проекты (Data science, AI, нейронные сети, машинное обучение)

## Финальный проект. Telegram-бот для школы живописи
[Ссылка на Google Drive](https://drive.google.com/drive/folders/1rnkOdUOpUpa8ReD5dlPzgQz-dzYLo57R?usp=sharing)
Тема: Telegram-бот для школы живописи
Цель - создать Телеграм-бота, которого можно интегрировать для бизнеса (школа/курсы рисования). Задача бота - заинтересовать пользователя, убедить его начать обучение в школе или купить курсы. Бот отвечает на вопросы о стилях живописи, техниках рисования, а также предоставляет подробную информацию о данной школе (стоимость, сроки обучения, типы занятий). Пользователь может отправить свой рисунок, бот определяет принадлежность работы пользователя к одному из 6 стилей (в процентном соотношении: например, это 89% импрессионизм, 9% реализм, 2% поп-арт). Большинство людей рисуют, не зная, к какому именно стилю живописи можно отнести их работы, какова их ценность с точки зрения искусствоведения, задача бота - даль эту информацию, замотивировать пользователя совершенствовать свои навыки и заниматься рисованием профессионально в этой школе.

В Телеграм-бот  интегрированы 2 нейросети: 1) дообученная на нашей базе модель СhatGPT и интегрированная в бот по API (будем обучать на базе документа, содержащего информацию о стилях живописи, жанрах, техниках рисования, а также информацию о данной школе живописи (школа может быть любой, добавим произвольную информацию о расписании, стоимости обучения, типах курсов - всё то, что должен знать человек, который хочет обучаться в этой школе); 2) нейросеть для задачи классификации изображений (тестирование и сравнительный анализ нейросетей с разными архитектурами на основе Conv2D, VGG16, ResNet).

Этот проект включает в себя предобработку данных, создание архитектуры нейросети, дообучение языковой модели, интегрирование модель в Telegram-бот (можно запустить ноутбук 5 (финальный), чтобы протестировать работу моделей.

## Проекты по обработке изображений

### Задачи распознавания текста на изображении
Предобработка изображений с помощью библиотеки OpenCV для улучшения качества изображения с целью повышения точности распознавания текста на изображении. Оптическое распознавание символов (OCR). Тестирование работы Tesseract. 

#### Распознавание текста на изображении. OCR
[Ссылка на Google Colab](https://colab.research.google.com/drive/15U9NPytJsGX1rzWROn3x0k-59VnF2Btf?usp=sharing)
Протестируем работу OCR. Возьмём произовольный документ (СНИЛС, свидетельство о рождении и т.п.), проведём распознавание документа с помощью OCR, выделите основные поля документа, сформируем словарь с координатами, выполним "умное" распознавание с использованием SpellChecker.

#### OpenCV и Tesseract
[Ссылка на Google Colab](https://colab.research.google.com/drive/1lWSJ50jjquaAijpgHP1a0p9q0366l_pk?usp=sharing)
Проект посвящен оптимизации работы технологий оптического распознавания символов (Tesseract) благодаря предобработке изображений с помощью инструментов библиотеки OpenCV.

#### Тестирование Tesseract, EasyOCR, PaddleOCR (с OpenCV и без)
[Ссылка на Google Colab](https://colab.research.google.com/drive/1ctFUA4jHmoEGL7M40Qb7MTBMKPMuFR-l?usp=sharing)
Проект посвящен тестированию технологий оптического распознавания символов (OCR), включая Tesseract, PaddleOCR и EasyOCR. 
Цель: сравнение производительности, точности и скорости работы Tesseract, PaddleOCR и EasyOCR на различных наборах данных.

### Задачи сегментации изображений

#### Сегментация изображений на U-Net
[Ссылка на Google Colab](https://colab.research.google.com/drive/1zwNxqceY71mAQ71oEECnCiP7bx8MWVUH?usp=sharing)
В этом проекте проводим сравнительный анализ моделей U-Net, Simple U-Net и расширенной модели U-Net для задачи сегментации изображения. Разработаем и обучим модели для достижения лучшего результата. 

#### Сегментация изображений на фреймворке TerraSegmentation
[Ссылка на Google Colab](https://colab.research.google.com/drive/1EknGUKwUHf2lWiAOiceLTBWGpPPeI6qZ?usp=sharing)
В этом проекте проводим сравнительный анализ разных архитектур моделей, созданных на фреймворке TerraSegmentation (U-Net). Применяем метод фрагментации изображений (crop-4, crop-16), проводим анализ результатов, представляем их в сравнительной таблице.

#### Трёхклассовая классификация. U-Net, PSPNet
Проект по сегментации изображений с целью определения оптимального пути для прокладывания маршрута для робота путём распознавания неподвижных объектов (пней, валежника, деревьев, оврагов и т.д.) на пути робота.
[Ссылка на Google Colab](https://colab.research.google.com/drive/1jGMlMMoVCGGvYwEgRAqwn5jDuFuse32s?usp=sharing)
Проект посвящен созданию архитектуры нейросети модели и подбора гиперпараметров для обучения для наиболее высокой точности определения оптимального пути для робота. Трёхклассовая классификация (дорога, препятствие, небо).

#### Двухклассовая классификация. U-Net, PSPNet
Проект по сегментации изображений с целью определения оптимального пути для прокладывания маршрута для робота путём распознавания неподвижных объектов (пней, валежника, деревьев, оврагов и т.д.) на пути робота.
[Ссылка на Google Colab](https://colab.research.google.com/drive/1y4yPwTR9d3UEFk_baIXkwDyo4xM-Cucs?usp=sharing)
[Ссылка на Google Colab](https://colab.research.google.com/drive/1BV15mM883OkTYcnad6wPXYFKXwKxUivi?usp=sharing)
Проект посвящен созданию архитектуры нейросети модели и подбора гиперпараметров для обучения для наиболее высокой точности определения оптимального пути для робота. Двухклассовая классификация (дорога, препятствие).

### Работа с изображениями и видео
[Ссылка на Google Colab](https://colab.research.google.com/drive/1g6xsPTYmfcsEhDmtrAIyje02J-LKjP87?usp=sharing)
Проводим обработку изображений и видео с использованием инструментов различных библиотек: Matplotlib, PIL, OpenCV, VideoGear и т.д.

### Задача классификации изображений
[Ссылка на Google Colab](https://colab.research.google.com/drive/1WedjAzxcjlIMCITo5HYVzQrrVbgXaNDR?usp=sharing)
Проект по классификации изображений пассажиров автобуса на входящих и выходящих. Создание и обучение модели на базе свёрточных нейронных сетей.

### Задача Object Detection
[Ссылка на Google Colab](https://colab.research.google.com/drive/1FXaEoDUTLPaiv0oXARAcu3l8nfhzNujU?usp=sharing)
Проект по обучению модели TerraYolo для задачи распознавания объектов на изображении. Спавнительный анализ оригинальной и дообученной модели.

## Проекты Data Science, AI, машинное обучение

### Работа с данными, задача классификации
[Ссылка на Google Colab](https://colab.research.google.com/drive/1sKFJio7l_JAHk1x9QntU8NMMFVSDRZWD?usp=sharing)
Проект посвящен созданию архитектуры модели для классификации вин по параметрам. Мы используем библиотеку Scikit-learn и методы нормализации данных.

### Задача регрессии
[Ссылка на Google Colab](https://colab.research.google.com/drive/19GCCMFmQJsi-V3XU78g-wfMcRHEE0hAb?usp=sharing)
В этом проекте разрабатываем модель для задачи регрессии. Тестируем различные способы предподготовки и нормализации данных для достижения лучших результатов обучения модели. Используем MinMaxScaler, StandartScaler и т.д.

### Прогнозированием временных рядов
[Ссылка на Google Colab](https://colab.research.google.com/drive/11588MOHxB3Xpd74_ehxLXmAp4O8GeIfb?usp=sharing)
В этом проекте разрабатываем модель для прогнозирования стоимости акций компании "Лукойл". Используем TimeseriesGenerator и т.д. Внедряем различные способы аугментации данных для подачи в нейронную сеть для обучения.

### Прогнозированием временных рядов на фреймворках
[Ссылка на Google Colab](https://colab.research.google.com/drive/1dHJRC13QBIv8Bv0ZC7AA6pCyBtux--wc?usp=sharing)
В этом проекте разрабатываем модель для прогнозирования стоимости акций компании "Газпром". Используем фреймворки Autokeras и Terra-ai-datasets.

### Визуализация данных. Matplotlib
[Ссылка на Google Colab](https://colab.research.google.com/drive/1ysM_r_tHXijwVx47wlCkHNwd5t_90sXq?usp=sharing)
В этом проекте мы тестируем инструменты библиотеки Matplotlib для задач визуализации данных.

## Проекты по обработке текстов, синтезу и распознаванию речи, работе с языковыми моделями

### Задача классификации текстов. Bag of Words, Embeddings
[Ссылка на Google Colab](https://colab.research.google.com/drive/1kcLO3uzt3SvZYmLYN3eGKAaDC1HvlJ-O?usp=sharing)
Проект посвящен созданию архитектуры модели для классификации текстов. Используем Bag of Words, Embeddings, проводим сравнительный анализ эффективности работы моделей. 

### Классификация текстов. Bag of Words, реккурентные и одномерные нейронные сети
[Ссылка на Google Colab](https://colab.research.google.com/drive/1QkLtHPsT-oIKQzygiNo8Lr8XwKxSdlVO?usp=sharing)
В этом проекте мы разрабатываем модель для эффективной классификации текстов с использованием Bag of Words, LSTM, GRU, Conv1D и т.д.

### Классификация текстов на фреймворках
[Ссылка на Google Colab](https://colab.research.google.com/drive/1WoZkDfjPYS8KqvRpYos4zYvuFYcM3fuq?usp=sharing)
В этом проекте мы разрабатываем модель для наиболее эффективной классификации текстов с использованием фреймворков (Autokeras, Automodel, фреймворка для предподготовки данных Terra-ai-datasets).

### Синтез и распознавание речи
[Ссылка на Google Colab](https://colab.research.google.com/drive/1C4XYgzuR_zjYIk927gODO_6lTRWMQKfJ?usp=sharing)
[Ссылка на Google Colab](https://colab.research.google.com/drive/1vTbW1HzHmd2QET64on06zBR7B6sPahVX?usp=sharing)
Тестируем gTTS: Выполним синтез речи/запись с микрофона, распознавание речи, оценим качество распознавания фрагмента с помощью jiwer.

### Дообучение ChatGPT на нашей базе
[Ссылка на Google Colab](https://colab.research.google.com/drive/1MY_dBCNScHNA-Mw0dxtnV4LHGfgMXVoJ?usp=sharing)
Проект по дообучению модели ChatGPT от OpenAI на нашей базе данных и созданию системы голосового помощника. Используем FAISS, langchain, OpenAIEmbeddings, gTTS, sR.

## Проекты с использованием инструментов для интеграции в production AI проектов

### Telegram-bot с Object Detection
[Ссылка на Google Colab](https://colab.research.google.com/drive/1QljZEW113VU0tf3xtTYcf9byfy20WBvB?usp=sharing) - скриншоты, демонстрирующие работу бота
Создадим Telegram-bot с интегрированной моделью TerraYolo для задачи Object Detection. Протестируем работу модели с разными параметрами распознавания.

### Веб-приложение FastAPI и запуск на сервере Uvicorn
[Ссылка на Google Colab](https://colab.research.google.com/drive/1DfvDhh3waIzDInIuMGSkgsin9jH7-uop?usp=sharing) - код
[Ссылка на Google Colab](https://colab.research.google.com/drive/1OiHuahmMBnVSV3pgRFcAAJwuTD-zOpVb?usp=sharing) - скрины, демонстрирующие работу приложения

### Запуск кода на серверах WSGI и ASGI
[Ссылка на Google Colab](https://colab.research.google.com/drive/1AIbyzGpArB62N6J-eNo-UzPvB88QyGRq?usp=sharing) - скрины результатов обращения к модели на серверах
[Ссылка на Google Colab](https://colab.research.google.com/drive/1hoRFgirbZe23BMfK9nluQH8Z4v0fXoSw?usp=sharing) - код
Запуск YOLOv8x на WSGI и ASGI (на GPU и CPU: сравнительный анализ скорости обработки изображений).




