# Инструмент создания предварительной разметки. Слежение за одним объектом

В tracker нужно предварительно изменить интервалы пропусков кадров (где нет объекта).
## Запуск

`>python -m selection_and_tracking_with_detection --video video.mp4 --tracker csrt`

## Параметры

* `-v`, `--video` - Путь к видео - **Обязательный праметр**
* `-t`, `--tracker` - Название трекера. Возможные значения "csrt", "kcf", "boosting", "mil", "tld", "medianflow", "mosse". По умолчанию *kcf*.

## Выбрать объект для трекинга

Нажать клавишу `a`; Выбрать с помощью мыши объект, нажать `enter`.

## Удалить рамку

Нажать клавишу `d`;

## Сохранить кадр

Нажать клавишу `s`; Сохранено будет в папке, указанной в файле constants.

## Пауза

Нажать клавишу `w`. Для продолжения нажать любую клавишу

## Выход

Нажать клавишу `Esc` или закрыть окно.


---

С помощью данного модуля selection_and_tracking_with_detection была получена предварительная разметка. В дальнейшем она была уточнена посредством CVAT ([проект на github](https://github.com/openvinotoolkit/cvat), [web-версия](https://cvat.org/tasks)). В результате была получена база размеченных видео. 

# База размеченных видеоизображений с БПЛА

### Название базы данных:
База видеоданных с разметкой беспилотных летательных аппаратов (квадрокоптеров)

### Реферат:
База видеоданных беспилотных летательных аппаратов с их покадровой разметкой в виде объемлющего прямоугольника, который содержит объект (дрон). Данная база может быть использована для задач машинного обучения, например, задачи выделения на изображениях летательных аппаратов или классификации объектов. При этом база содержит видео с разных ракурсов с использованием разных локаций, летательный аппарат присутствует на разных масштабах (крупным планом мало фрагментов). Также, поскольку база содержит размеченные видеоданные, то на основе ее могут решаться задачи обработки видео, например, слежения за объектом или классификация объекта по траектории движения. Тип ЭВМ: IBM PC-совместимый ПК; ОС: Microsoft Windows 2000/XP/Vista/7/Lunux.

### Вид и версия системы управления базой данных:
Excel

### Объем базы данных:
2,95 ГБ