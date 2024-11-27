import matplotlib.pyplot as plt
import numpy as np
import re

def read_output_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def parse_points(content):
    classes = {}
    class_pattern = r'Класс\s+(\d+):\s*\n((?:[^\n]+\n)+?)(?=\n|$)'
    matches = re.findall(class_pattern, content, re.MULTILINE)
    for class_id, points_str in matches:
        points = []
        for line in points_str.strip().split('\n'):
            line = line.strip()
            if line:
                coords = list(map(float, line.strip().split()))
                points.append(coords)
        if points:
            classes[int(class_id)] = np.array(points)
    return classes

def parse_clusters(content, algorithm_name):
    clusters = []
    pattern = rf'{algorithm_name}.*?\n\nНайдено кластеров: \d+\s*\n((?:Кластер \d+:\s*\nЦентр кластера: .+\nТочки кластера \d+:\s*\n(?:[^\n]+\n)+?\n)+)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return clusters
    clusters_str = match.group(1)
    cluster_pattern = r'Кластер \d+:\s*\nЦентр кластера: (.+)\nТочки кластера \d+:\s*\n((?:[^\n]+\n)+?)\n'
    cluster_matches = re.findall(cluster_pattern, clusters_str)
    for center_str, points_str in cluster_matches:
        center = list(map(float, center_str.strip().split()))
        points = []
        for line in points_str.strip().split('\n'):
            line = line.strip()
            if line:
                coords = list(map(float, line.strip().split()))
                points.append(coords)
        if points:
            clusters.append({
                'center': center,
                'points': np.array(points)
            })
    return clusters

def parse_classification_results(content, method_name):
    # Уточненный шаблон для парсинга результатов классификации перцептроном
    pattern = rf'{method_name}.*?\n((?:Точка \d+: .+\n)+)\nТочность классификации.*?: (.+?)%'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    results_str, accuracy_str = match.groups()
    # Поскольку веса и смещение отсутствуют, установим их в None или по умолчанию
    weights = None
    bias = None
    results = []
    for line in results_str.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        coords_match = re.search(r'Точка \d+: (.+?) \|', line)
        if coords_match:
            coords_str = coords_match.group(1)
            coords = list(map(float, coords_str.strip().split()))
            label_match = re.search(r'Класс: (-?\d+)', line)
            prediction_match = re.search(r'Предсказание: (-?\d+)', line)
            if label_match and prediction_match:
                label = int(label_match.group(1))
                prediction = int(prediction_match.group(1))
                results.append({
                    'coords': coords,
                    'label': label,
                    'prediction': prediction
                })
    accuracy = float(accuracy_str.strip())
    return {
        'weights': weights,
        'bias': bias,
        'results': results,
        'accuracy': accuracy
    }

def visualize_classes(classes):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for idx, (class_id, points) in enumerate(classes.items()):
        plt.scatter(points[:, 0], points[:, 1], c=colors[idx % len(colors)], label=f'Класс {class_id}')
    plt.title('Точки каждого класса')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_clusters(clusters, algorithm_name):
    if not clusters:
        print(f'Кластеры для {algorithm_name} не найдены или пусты.')
        return
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab20').colors
    for idx, cluster in enumerate(clusters):
        points = cluster['points']
        center = cluster['center']
        plt.scatter(points[:, 0], points[:, 1], c=[colors[idx % len(colors)]], label=f'Кластер {idx + 1}')
        plt.plot(center[0], center[1], marker='x', markersize=10, color='k')
    plt.title(f'Результаты кластеризации: {algorithm_name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_classification(results, weights, bias, method_name):
    if not results:
        print(f'Результаты классификации для {method_name} не найдены или пусты.')
        return
    plt.figure(figsize=(8, 6))
    colors = {}
    for res in results:
        coords = res['coords']
        label = res['label']
        prediction = res['prediction']
        if label not in colors:
            colors[label] = plt.cm.rainbow(np.random.rand())
        marker = 'o' if label == prediction else 'x'
        plt.scatter(coords[0], coords[1], c=[colors[label]], marker=marker, label=f'Класс {label}' if marker == 'o' else f'Ошибочная классификация')
    # Не пытаемся построить разделяющую прямую
    plt.title(f'Результаты классификации: {method_name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    filename = 'output.txt'
    content = read_output_file(filename)
    # Парсим точки каждого класса
    classes = parse_points(content)
    # Визуализируем классы
    visualize_classes(classes)
    # Список алгоритмов кластеризации
    clustering_algorithms = [
        'Алгоритм FOREL',
        'Алгоритм максиминного расстояния',
        'Алгоритм ISODATA',
        'Алгоритм, основанный на методе просеивания'
    ]
    # Визуализируем результаты каждого алгоритма кластеризации
    for algorithm_name in clustering_algorithms:
        clusters = parse_clusters(content, algorithm_name)
        if clusters:
            visualize_clusters(clusters, algorithm_name)
        else:
            print(f'Кластеры для {algorithm_name} не найдены или пусты.')
    # Визуализируем результаты классификации перцептроном
    perceptron_results = parse_classification_results(content, 'Результаты классификации перцептроном')
    if perceptron_results:
        visualize_classification(
            perceptron_results['results'],
            perceptron_results['weights'],
            perceptron_results['bias'],
            'Однослойный перцептрон'
        )
    else:
        print('Результаты перцептрона не найдены.')

if __name__ == '__main__':
    main()
