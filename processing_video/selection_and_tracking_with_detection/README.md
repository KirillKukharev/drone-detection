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


