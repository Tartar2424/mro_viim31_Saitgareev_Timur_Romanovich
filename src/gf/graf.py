import matplotlib.pyplot as plt
import re

def parse_output_file(filename):
    data = {}
    current_section = ''
    current_class = ''
    current_cluster = ''
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Определяем разделы в файле
        if line.startswith('Точки каждого класса'):
            current_section = 'Classes'
            data[current_section] = {}
            i += 1
            continue
        elif line.startswith('Центры классов'):
            current_section = 'ClassCenters'
            data[current_section] = {}
            i += 1
            continue
        elif line.startswith('Расстояния между центрами классов'):
            current_section = 'ClassDistances'
            data[current_section] = []
            i += 1
            continue
        elif line.startswith('Применение алгоритмов'):
            # Пропускаем этот раздел
            current_section = ''
            i += 1
            continue
        elif line.startswith('Выводим информацию о выбранной метрике') or line == '':
            # Пропускаем пустые строки и ненужную информацию
            i += 1
            continue
        elif line.startswith('Алгоритм') or line.startswith('Метод'):
            current_section = line.strip()
            data[current_section] = {}
            i += 1
            continue

        # Парсим данные в зависимости от текущего раздела
        if current_section == 'Classes':
            # Обработка точек каждого класса
            if line.startswith('Класс'):
                current_class = line.split(' ')[1].strip(':')
                data[current_section][current_class] = []
                i += 1
                continue
            elif line != '':
                # Читаем координаты точки
                point_coords = list(map(float, line.split()))
                data[current_section][current_class].append(point_coords)
                i += 1
                continue
            else:
                i += 1
                continue

        elif current_section == 'ClassCenters':
            # Обработка центров классов
            if line.startswith('Центр класса'):
                parts = line.split(':')
                class_number = parts[0].split(' ')[2]
                center_coords = list(map(float, parts[1].strip().split()))
                data[current_section][class_number] = center_coords
                i += 1
                continue
            else:
                i += 1
                continue

        elif current_section == 'ClassDistances':
            # Обработка расстояний между центрами классов
            match = re.search(r'Расстояние между классами (\d+) и (\d+): (.+)', line)
            if match:
                class1 = match.group(1)
                class2 = match.group(2)
                distance = float(match.group(3))
                data[current_section].append({'classes': (class1, class2), 'distance': distance})
            i += 1
            continue

        elif current_section.startswith('Алгоритм') or current_section.startswith('Метод'):
            # Обработка результатов алгоритмов кластеризации
            if line.startswith('Найдено кластеров:'):
                total_clusters = int(line.split(':')[1].strip())
                data[current_section]['total_clusters'] = total_clusters
                data[current_section]['clusters'] = {}
                i += 1
                continue
            elif line.startswith('Кластер'):
                current_cluster = line.split(' ')[1].strip(':')
                data[current_section]['clusters'][current_cluster] = {'center': [], 'points': []}
                i += 1
                continue
            elif line.startswith('Центр кластера:'):
                center_coords = list(map(float, line.split(':')[1].strip().split()))
                data[current_section]['clusters'][current_cluster]['center'] = center_coords
                i += 1
                continue
            elif line.startswith('Точки кластера'):
                i += 1  # Пропускаем строку с надписью 'Точки кластера'
                while i < len(lines) and lines[i].strip() != '':
                    point_line = lines[i].strip()
                    point_coords = list(map(float, point_line.split()))
                    data[current_section]['clusters'][current_cluster]['points'].append(point_coords)
                    i += 1
                i += 1  # Пропускаем пустую строку
                continue
            else:
                i += 1
                continue

        else:
            i += 1
            continue

    return data

def visualize_classes(data):
    # Проверяем, есть ли данные для классов
    if 'Classes' not in data:
        print("Данные о классах отсутствуют.")
        return
    # Визуализация точек каждого класса вместе с центрами классов
    plt.figure(figsize=(8,6))
    plt.title('Точки каждого класса и центры классов')
    colors = plt.get_cmap('tab10').colors

    for idx, (class_label, points) in enumerate(data['Classes'].items()):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        # Отображаем точки класса
        plt.scatter(xs, ys, color=colors[idx % len(colors)], label=f'Класс {class_label}')
        # Отображаем центр класса
        if class_label in data.get('ClassCenters', {}):
            center = data['ClassCenters'][class_label]
            plt.scatter(center[0], center[1], color=colors[idx % len(colors)], marker='X', s=200, edgecolor='black')
            # Добавляем подпись для центра класса
            plt.text(center[0], center[1], f'Центр {class_label}', fontsize=9, ha='right', va='bottom')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_class_distances(data):
    # Визуализация расстояний между центрами классов
    if 'ClassDistances' not in data:
        print("Данные о расстояниях между центрами классов отсутствуют.")
        return
    plt.figure(figsize=(8,6))
    plt.title('Расстояния между центрами классов')

    centers = data['ClassCenters']
    colors = plt.get_cmap('tab10').colors

    # Отображаем центры классов
    for idx, (class_label, center_coords) in enumerate(centers.items()):
        plt.scatter(center_coords[0], center_coords[1], color=colors[idx % len(colors)], marker='X', s=200, edgecolor='black')
        # Добавляем подпись для центра класса
        plt.text(center_coords[0], center_coords[1], f'Центр {class_label}', fontsize=9, ha='right', va='bottom')

    # Рисуем линии между центрами классов
    for item in data['ClassDistances']:
        class1, class2 = item['classes']
        distance = item['distance']
        center1 = centers[class1]
        center2 = centers[class2]
        plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 'k--')
        # Добавляем подпись расстояния в середине линии
        mid_x = (center1[0] + center2[0]) / 2
        mid_y = (center1[1] + center2[1]) / 2
        plt.text(mid_x, mid_y, f'{distance:.2f}', fontsize=8, ha='center', va='center', backgroundcolor='white')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_clusters(data):
    # Проходим по всем алгоритмам кластеризации
    for key in data.keys():
        if key.startswith('Алгоритм') or key.startswith('Метод'):
            algorithm_name = key
            algorithm_data = data[key]

            # Проверяем, что есть ключ 'clusters' в data[key]
            if 'clusters' not in algorithm_data:
                print(f"В данных алгоритма '{algorithm_name}' отсутствует информация о кластерах.")
                continue

            clusters = algorithm_data['clusters']

            plt.figure(figsize=(8,6))
            plt.title(algorithm_name)
            colors = plt.get_cmap('tab10').colors

            for idx, (cluster_label, cluster_data) in enumerate(clusters.items()):
                points = cluster_data['points']
                if len(points) == 0:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                # Отображаем точки кластера
                plt.scatter(xs, ys, color=colors[idx % len(colors)], label=f'Кластер {cluster_label}')
                # Отображаем центр кластера с подписью
                center = cluster_data['center']
                if center:
                    plt.scatter(center[0], center[1], color='black', marker='X', s=200)
                    # Добавляем подпись для центра кластера
                    plt.text(center[0], center[1], f'Центр {cluster_label}', fontsize=9, ha='right', va='bottom')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.show()

def visualize_ho_kashyap(data):
    # Визуализация результатов метода Хо-Кошьяпа
    if 'Метод Хо-Кошьяпа для первых двух классов:' in data:
        method_data = data['Метод Хо-Кошьяпа для первых двух классов:']
        weights = []
        bias = 0

        # Извлечение весов и смещения из данных
        with open('output.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        capturing_weights = False
        capturing_results = False
        points = []
        labels = []

        for line in lines:
            if line.strip() == 'Веса w:':
                capturing_weights = True
                continue
            if capturing_weights:
                if line.strip().startswith('w['):
                    weight_value = float(line.strip().split('=')[1])
                    weights.append(weight_value)
                elif line.strip().startswith('Смещение'):
                    bias = float(line.strip().split(':')[1])
                else:
                    capturing_weights = False
            if line.strip() == 'Результаты классификации:':
                capturing_results = True
                continue
            if capturing_results:
                if line.strip() == '':
                    continue
                if line.strip().startswith('Точка'):
                    parts = line.strip().split('|')
                    point_part = parts[0].split(':')[1].strip()
                    label_part = parts[1].split(':')[1].strip()
                    coords = list(map(float, point_part.strip().split()))
                    label = int(label_part)
                    points.append(coords)
                    labels.append(label)
                else:
                    capturing_results = False

        if len(points) == 0:
            return

        # Построим график
        plt.figure(figsize=(8,6))
        plt.title('Метод Хо-Кошьяпа')

        # Отобразим точки разных классов
        colors_map = {1: 'blue', -1: 'red'}
        for idx, point in enumerate(points):
            label = labels[idx]
            plt.scatter(point[0], point[1], color=colors_map[label], label=f'Класс {label}' if idx == 0 else '')

        # Построим разделяющую линию
        import numpy as np
        x_vals = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
        y_vals = -(bias + weights[0]*x_vals) / weights[1]
        plt.plot(x_vals, y_vals, '--k', label='Разделяющая линия')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    filename = 'output.txt'
    data = parse_output_file(filename)
    # Визуализируем точки и центры классов на одном графике
    visualize_classes(data)
    # Визуализируем расстояния между центрами классов
    visualize_class_distances(data)
    # Визуализируем результаты алгоритмов кластеризации с подписями центров кластеров
    visualize_clusters(data)
    # Визуализируем результаты метода Хо-Кошьяпа
    visualize_ho_kashyap(data)

if __name__ == '__main__':
    main()
