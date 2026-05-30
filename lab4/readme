# Данные

Набор данных -> (Titanic)[https://www.kaggle.com/datasets/yasserh/titanic-dataset]

Итоговая папка (Google Drive)[https://drive.google.com/drive/folders/1tzoiwV-SyZ3Avtpu68fQWoe0IUjUQxQo]

Данные храняться в папке "data". С помощью DVC отслеживаются: "data", с помощью git отслеживаются "data.dv" и ".gitignore"

# Технологии

* DVC
* Git
* Uv
* Python

В качестве хранилища для DVC был использован Google Drive. Подключенный с помощью Google Cloud project, настроенный в соответствии с документацией DVC.

В DVC был установлен autostage = true

Были написаны три скрипта:
* 1_preprocess.py
* 2_fillna.py
* 3_onehot.py

# Процесс

## Этап 1

Инициализация репозитория
```bash
git init
dvc init
dvc remote add -d storage gdrive://1tzoiwV-SyZ3Avtpu68fQWoe0IUjUQxQo
dvc remote modify myremote gdrive_client_id 'client-id'
dvc remote modify myremote gdrive_client_secret 'client-secret'
```

!(init)[img/1.png]

!(conf storage)[img/2.png]

## Этап 2

Удаленние ненужных столбцов и загрузка в DVC
```bash
uv run 1_preprocess.py
dvc add data
git commit -m "Strip Features"
```

## Этап 3

Заполнение пропусков в Age
```bash
uv run 2_fillna.py
dvc add data
git commit -m "Fill NaN in Age"
```

## Этап 4

OneHotEncoding колонки Sex
```bash
uv run 3_onehot.py
dvc add data
git commit -m "OneHotEncoding"
```

# Результаты

Итоговые коммиты
```bash
git log --oneline
```

!(commits)[img/3.png]

Структура Google Drive
!(google drives)[img/4.png]