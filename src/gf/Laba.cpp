#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>
#include <limits>
#include <utility>
#include <numeric>

// Определение класса Point
class Point {
private:
    std::vector<double> coordinates;
    int label = 0;       // Метка класса
    bool assigned = false; // Флаг для алгоритмов кластеризации

public:
    Point(int dimensions) : coordinates(dimensions, 0.0) {}

    Point(const std::vector<double>& coords) : coordinates(coords) {}

    int getDimensions() const {
        return static_cast<int>(coordinates.size());
    }

    double getCoordinate(int index) const {
        return coordinates[index];
    }

    void setCoordinate(int index, double value) {
        coordinates[index] = value;
    }

    const std::vector<double>& getCoordinates() const {
        return coordinates;
    }

    int getLabel() const {
        return label;
    }

    void setLabel(int lbl) {
        label = lbl;
    }

    bool isAssigned() const {
        return assigned;
    }

    void setAssigned(bool value) {
        assigned = value;
    }

    bool operator==(const Point& other) const {
        return coordinates == other.coordinates;
    }
};

using Vector = std::vector<double>;

// Определение класса Space
class Space {
public:
    enum DistanceMetric {
        EUCLIDEAN,
        MANHATTAN
    };

    static double calculateDistance(const Point& a, const Point& b, DistanceMetric metric = EUCLIDEAN) {
        if (metric == EUCLIDEAN) {
            double sum = 0.0;
            for (int i = 0; i < a.getDimensions(); ++i) {
                double diff = a.getCoordinate(i) - b.getCoordinate(i);
                sum += diff * diff;
            }
            return std::sqrt(sum);
        } else if (metric == MANHATTAN) {
            double sum = 0.0;
            for (int i = 0; i < a.getDimensions(); ++i) {
                sum += std::abs(a.getCoordinate(i) - b.getCoordinate(i));
            }
            return sum;
        }
        return 0.0;
    }

    Space(int pointCount, int dimensions, double minX, double maxX, double minY, double maxY) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<double> distributionX(minX, maxX);
        std::uniform_real_distribution<double> distributionY(minY, maxY);

        for (int i = 0; i < pointCount; ++i) {
            std::vector<double> coords(dimensions);
            coords[0] = distributionX(generator);
            if (dimensions > 1) {
                coords[1] = distributionY(generator);
            }
            points.emplace_back(coords);
        }
    }

    const std::vector<Point>& getPoints() const {
        return points;
    }

    std::vector<Point>& getPoints() {
        return points;
    }

    Point calculateCenter() const {
        int dimensions = points[0].getDimensions();
        std::vector<double> centerCoords(dimensions, 0.0);

        for (const auto& point : points) {
            for (int i = 0; i < dimensions; ++i) {
                centerCoords[i] += point.getCoordinate(i);
            }
        }
        for (double& coord : centerCoords) {
            coord /= points.size();
        }

        return Point(centerCoords);
    }

private:
    std::vector<Point> points;
};

// Функция для вычисления центра кластера
Point calculateClusterCenter(const std::vector<Point>& cluster) {
    int dimensions = cluster[0].getDimensions();
    std::vector<double> centerCoords(dimensions, 0.0);

    for (const auto& point : cluster) {
        for (int i = 0; i < dimensions; ++i) {
            centerCoords[i] += point.getCoordinate(i);
        }
    }
    for (double& coord : centerCoords) {
        coord /= cluster.size();
    }

    return Point(centerCoords);
}

// Функция для вычисления центра кластера (для вектора указателей на точки)
Point calculateClusterCenter(const std::vector<Point*>& cluster) {
    int dimensions = cluster[0]->getDimensions();
    std::vector<double> centerCoords(dimensions, 0.0);

    for (const auto& pointPtr : cluster) {
        const Point& point = *pointPtr;
        for (int i = 0; i < dimensions; ++i) {
            centerCoords[i] += point.getCoordinate(i);
        }
    }
    for (double& coord : centerCoords) {
        coord /= cluster.size();
    }

    return Point(centerCoords);
}

// Функция алгоритма FOREL
std::vector<std::vector<Point*>> forelClustering(
    std::vector<Point>& dataPoints,
    double R,
    Space::DistanceMetric metric = Space::EUCLIDEAN) {

    std::vector<std::vector<Point*>> clusters;
    bool allAssigned = false;

    while (!allAssigned) {
        // Выбираем случайную неприсвоенную точку
        auto it = std::find_if(dataPoints.begin(), dataPoints.end(), [](const Point& p) {
            return !p.isAssigned();
        });

        if (it == dataPoints.end()) {
            allAssigned = true;
            break;
        }

        Point* currentPoint = &(*it);

        std::vector<Point*> cluster;
        cluster.push_back(currentPoint);
        currentPoint->setAssigned(true);

        bool clusterChanged = true;
        while (clusterChanged) {
            clusterChanged = false;
            // Вычисляем центр кластера
            Point clusterCenter = calculateClusterCenter(cluster);

            // Формируем новый кластер
            std::vector<Point*> newCluster;
            for (auto& point : dataPoints) {
                if (!point.isAssigned()) {
                    double distance = Space::calculateDistance(point, clusterCenter, metric);
                    if (distance <= R) {
                        newCluster.push_back(&point);
                        point.setAssigned(true);
                        clusterChanged = true;
                    }
                }
            }

            cluster.insert(cluster.end(), newCluster.begin(), newCluster.end());
        }

        clusters.push_back(cluster);
    }

    return clusters;
}

// Функция алгоритма Maximin
std::vector<std::vector<Point>> maximinClustering(
    const std::vector<Point>& dataPoints,
    double threshold,
    Space::DistanceMetric metric = Space::EUCLIDEAN) {

    std::vector<Point> centers;
    centers.push_back(dataPoints[0]);

    for (const auto& point : dataPoints) {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto& center : centers) {
            double distance = Space::calculateDistance(point, center, metric);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        if (minDistance > threshold) {
            centers.push_back(point);
        }
    }

    std::vector<std::vector<Point>> clusters(centers.size());
    for (const auto& point : dataPoints) {
        double minDistance = std::numeric_limits<double>::max();
        int minIndex = -1;
        for (size_t i = 0; i < centers.size(); ++i) {
            double distance = Space::calculateDistance(point, centers[i], metric);
            if (distance < minDistance) {
                minDistance = distance;
                minIndex = static_cast<int>(i);
            }
        }
        clusters[minIndex].push_back(point);
    }

    return clusters;
}

// Функция алгоритма ISODATA
std::vector<std::vector<Point>> isodataClustering(
    const std::vector<Point>& dataPoints,
    int initialK,
    int maxIterations,
    int minClusterSize,
    double maxStandardDeviation,
    double minClusterDistance,
    Space::DistanceMetric metric = Space::EUCLIDEAN) {

    int dimensions = dataPoints[0].getDimensions();
    std::vector<Point> centers;
    std::sample(dataPoints.begin(), dataPoints.end(), std::back_inserter(centers),
                initialK, std::mt19937{std::random_device{}()});

    std::vector<std::vector<Point>> clusters;
    bool clustersChanged = true;
    int iterations = 0;

    while (clustersChanged && iterations < maxIterations) {
        clustersChanged = false;
        clusters.assign(centers.size(), std::vector<Point>());

        // Шаг 1: Присвоение точек ближайшим центрам
        for (const auto& point : dataPoints) {
            double minDistance = std::numeric_limits<double>::max();
            int minIndex = -1;
            for (size_t k = 0; k < centers.size(); ++k) {
                double distance = Space::calculateDistance(point, centers[k], metric);
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = static_cast<int>(k);
                }
            }
            clusters[minIndex].push_back(point);
        }

        // Шаг 2: Обновление центров кластеров
        for (size_t k = 0; k < centers.size(); ++k) {
            if (!clusters[k].empty()) {
                centers[k] = calculateClusterCenter(clusters[k]);
            }
        }

        // Шаг 3: Разделение кластеров
        for (size_t k = 0; k < centers.size(); ++k) {
            if (clusters[k].size() >= 2 * minClusterSize) {
                std::vector<double> stdDev(dimensions, 0.0);
                Point center = centers[k];
                for (const auto& point : clusters[k]) {
                    for (int d = 0; d < dimensions; ++d) {
                        double diff = point.getCoordinate(d) - center.getCoordinate(d);
                        stdDev[d] += diff * diff;
                    }
                }
                for (int d = 0; d < dimensions; ++d) {
                    stdDev[d] = std::sqrt(stdDev[d] / clusters[k].size());
                }

                bool splitCluster = false;
                for (int d = 0; d < dimensions; ++d) {
                    if (stdDev[d] > maxStandardDeviation) {
                        splitCluster = true;
                        break;
                    }
                }

                if (splitCluster) {
                    int maxStdDevDim = std::distance(stdDev.begin(), std::max_element(stdDev.begin(), stdDev.end()));
                    Point center1 = centers[k];
                    Point center2 = centers[k];
                    center1.setCoordinate(maxStdDevDim, center1.getCoordinate(maxStdDevDim) + stdDev[maxStdDevDim] / 2);
                    center2.setCoordinate(maxStdDevDim, center2.getCoordinate(maxStdDevDim) - stdDev[maxStdDevDim] / 2);

                    centers[k] = center1;
                    centers.push_back(center2);
                    clustersChanged = true;
                    break;
                }
            }
        }

        // Шаг 4: Объединение кластеров
        for (size_t i = 0; i < centers.size(); ++i) {
            for (size_t j = i + 1; j < centers.size(); ++j) {
                double distance = Space::calculateDistance(centers[i], centers[j], metric);
                if (distance < minClusterDistance) {
                    // Объединяем кластеры i и j
                    clusters[i].insert(clusters[i].end(), clusters[j].begin(), clusters[j].end());
                    clusters.erase(clusters.begin() + j);
                    centers.erase(centers.begin() + j);
                    clustersChanged = true;
                    break;
                }
            }
            if (clustersChanged) {
                break;
            }
        }

        iterations++;
    }

    return clusters;
}

// Функция для вычисления плотности точки (метод просеивания)
double computeDensity(const Point& x_i, const std::vector<Point>& dataPoints, double h, Space::DistanceMetric metric = Space::EUCLIDEAN) {
    double density = 0.0;
    for (const auto& x_j : dataPoints) {
        double distance = Space::calculateDistance(x_i, x_j, metric);
        density += std::exp(- (distance * distance) / (2 * h * h));
    }
    return density;
}

// Функция алгоритма, основанного на методе просеивания
std::vector<std::vector<Point>> sievingMethodClustering(
    const std::vector<Point>& dataPoints,
    double h,
    double R,
    Space::DistanceMetric metric = Space::EUCLIDEAN) {

    int N = static_cast<int>(dataPoints.size());
    // Шаг 1: Вычисляем плотность для каждой точки
    std::vector<std::pair<Point, double>> densities; // Пара (точка, плотность)
    for (const auto& x_i : dataPoints) {
        double density = computeDensity(x_i, dataPoints, h, metric);
        densities.emplace_back(x_i, density);
    }

    // Шаг 2: Сортируем точки по убыванию плотности
    std::sort(densities.begin(), densities.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Шаг 3: Выбор центров кластеров
    std::vector<Point> clusterCenters;
    for (const auto& [point, density] : densities) {
        bool isFar = true;
        for (const auto& center : clusterCenters) {
            double distance = Space::calculateDistance(point, center, metric);
            if (distance < R) {
                isFar = false;
                break;
            }
        }
        if (isFar) {
            clusterCenters.push_back(point);
        }
    }

    // Шаг 4: Присвоение точек кластерам
    std::vector<std::vector<Point>> clusters(clusterCenters.size());
    for (const auto& x_i : dataPoints) {
        double minDistance = std::numeric_limits<double>::max();
        int minCluster = -1;
        for (size_t k = 0; k < clusterCenters.size(); ++k) {
            double distance = Space::calculateDistance(x_i, clusterCenters[k], metric);
            if (distance < minDistance) {
                minDistance = distance;
                minCluster = static_cast<int>(k);
            }
        }
        clusters[minCluster].push_back(x_i);
    }

    return clusters;
}

// Функция метода Хо-Кошьяпа
std::pair<std::vector<double>, double> hoKashyap(
    const std::vector<Point>& dataPoints,
    const std::vector<int>& labels,
    double b0 = 0.0,
    double eta = 0.5,
    int maxIterations = 1000,
    double epsilon = 1e-3) {

    int N = static_cast<int>(dataPoints.size());
    int dimensions = dataPoints[0].getDimensions();

    // Инициализация весов и смещения
    std::vector<double> w(dimensions, 0.0);
    double b = b0;

    // Матрица A и вектор B
    std::vector<std::vector<double>> A(N, std::vector<double>(dimensions));
    std::vector<double> B(N);

    for (int i = 0; i < N; ++i) {
        double yi = labels[i];
        for (int j = 0; j < dimensions; ++j) {
            A[i][j] = yi * dataPoints[i].getCoordinate(j);
        }
        B[i] = -yi * b;
    }

    std::vector<double> e(N, 1.0); // Вектор ошибок
    int iter = 0;

    while (iter < maxIterations) {
        // Вычисляем вектор e
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < dimensions; ++j) {
                sum += A[i][j] * w[j];
            }
            e[i] = sum + B[i];
        }

        // Проверяем условие сходимости
        bool converged = true;
        for (double ei : e) {
            if (ei <= 0 || std::abs(ei) > epsilon) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }

        // Обновляем веса w и смещение b
        for (int j = 0; j < dimensions; ++j) {
            double deltaW = 0.0;
            for (int i = 0; i < N; ++i) {
                deltaW += eta * e[i] * A[i][j];
            }
            w[j] -= deltaW;
        }
        double deltaB = 0.0;
        for (int i = 0; i < N; ++i) {
            deltaB += eta * e[i] * (-labels[i]);
        }
        b -= deltaB;

        iter++;
    }

    return {w, b};
}
// Функция обучения однослойного перцептрона
std::pair<std::vector<double>, double> perceptronLearning(
    const std::vector<Point>& dataPoints,  // Входные данные: вектор точек возвращение пары из вектора и одного double
    const std::vector<int>& labels,        // Метки классов: +1 или -1 для каждой точки
    double learningRate = 0.1,             // (шаг) обучения, по умолчанию 0.1
    int maxIterations = 1000) {            // Максимальное количество итераций, по умолчанию 1000

    // Инициализация переменных
    int N = static_cast<int>(dataPoints.size());   // Количество точек в обучающей выборке
    int dimensions = dataPoints[0].getDimensions(); // Размерность пространства (количество признаков)

    // Инициализация весов и смещения
    std::vector<double> weights(dimensions, 0.0);  // Вектор весов, инициализированный нулями
    double bias = 0.0;                             // Смещение (bias), инициализируется нулем

    // Обучение перцептрона
    for (int iter = 0; iter < maxIterations; ++iter) { // Цикл по итерациям обучения
        bool errorMade = false;                         // Флаг для отслеживания ошибок 

        // Цикл по всем обучающим примерам
        for (int i = 0; i < N; ++i) {
            // Вычисление активации для текущей точки
            double activation = bias; // Инициализируем активацию смещением
            for (int j = 0; j < dimensions; ++j) {
                // Добавляем вклад каждого признака, умноженного на соответствующий вес
                activation += weights[j] * dataPoints[i].getCoordinate(j);
            }
            // Определение предсказанной метки на основе активации
            int predictedLabel = (activation >= 0) ? +1 : -1; // Пороговая функция активации
            int yi = labels[i]; // Истинная метка для текущей точки

            // Проверка, нужно ли обновлять веса
            if (predictedLabel != yi) {
                // Обновление весов и смещения при ошибочном предсказании
                for (int j = 0; j < dimensions; ++j) {
                    // Корректируем вес для каждого признака
                    weights[j] += learningRate * yi * dataPoints[i].getCoordinate(j);
                }
                // Корректируем смещение
                bias += learningRate * yi;
                errorMade = true; // Устанавливаем флаг ошибки
            }
        }
        // Если в текущей эпохе не было ошибок, завершаем обучение досрочно
        if (!errorMade) {
            break;
        }
    }

    // Возвращаем обученные веса и смещение
    return {weights, bias};
}


// Главная функция main()
int main() {
    // Открываем файлы для ввода и вывода
    std::ifstream inputFile("input.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Не удалось открыть файл input.txt для чтения.\n";
        return 1;
    }

    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл output.txt для записи.\n";
        return 1;
    }

    // Читаем количество классов
    int classCount;
    inputFile >> classCount;

    // Создаем вектор пространств для каждого класса
    std::vector<Space> spaces;
    int labelCounter = 1;
    for (int i = 0; i < classCount; ++i) {
        int pointCount, dimensions;
        double minX, maxX, minY, maxY;

        // Читаем параметры для каждого класса из файла
        inputFile >> pointCount >> dimensions >> minX >> maxX >> minY >> maxY;

        // Создаем пространство и добавляем в вектор
        spaces.emplace_back(pointCount, dimensions, minX, maxX, minY, maxY);

        // Устанавливаем метки классов для точек
        for (auto& point : spaces.back().getPoints()) {
            point.setLabel(labelCounter);
        }

        labelCounter++;
    }

    // Выбор метрики расстояния (EUCLIDEAN или MANHATTAN)
    // Расскомментируйте одну из строк ниже, чтобы выбрать метрику
    Space::DistanceMetric distanceMetric = Space::EUCLIDEAN;
    //Space::DistanceMetric distanceMetric = Space::MANHATTAN;

    // Выводим информацию о выбранной метрике
    outputFile << "Выбранная метрика расстояния: " << (distanceMetric == Space::EUCLIDEAN ? "Евклидова" : "Манхэттенская") << "\n\n";

    // Вывод точек каждого класса
    outputFile << "Точки каждого класса:\n";
    for (size_t idx = 0; idx < spaces.size(); ++idx) {
        outputFile << "Класс " << idx + 1 << ":\n";
        const auto& points = spaces[idx].getPoints();
        for (const auto& point : points) {
            for (int d = 0; d < point.getDimensions(); ++d) {
                outputFile << point.getCoordinate(d);
                if (d < point.getDimensions() - 1) {
                    outputFile << " ";
                }
            }
            outputFile << "\n";
        }
        outputFile << "\n";
    }

    // Вычисляем центры классов
    std::vector<Point> centers;
    for (const auto& space : spaces) {
        centers.push_back(space.calculateCenter());
    }

    // Вывод центров классов
    outputFile << "Центры классов:\n";
    for (size_t idx = 0; idx < centers.size(); ++idx) {
        outputFile << "Центр класса " << idx + 1 << ": ";
        const auto& center = centers[idx];
        for (int j = 0; j < center.getDimensions(); ++j) {
            outputFile << center.getCoordinate(j);
            if (j < center.getDimensions() - 1) {
                outputFile << " ";
            }
        }
        outputFile << '\n';
    }

    // Нахождение расстояний между центрами классов с выбранной метрикой
    outputFile << "\nРасстояния между центрами классов:\n";
    for (size_t i = 0; i < centers.size(); ++i) {
        for (size_t j = i + 1; j < centers.size(); ++j) {
            double distance = Space::calculateDistance(centers[i], centers[j], distanceMetric);
            outputFile << "Расстояние между классами " << (i + 1) << " и " << (j + 1) << ": " << distance << '\n';
        }
    }

    // Сбор всех точек из всех классов
    std::vector<Point> allPoints;
    for (auto& space : spaces) {
        auto& pts = space.getPoints();
        allPoints.insert(allPoints.end(), pts.begin(), pts.end());
    }

    // Применение алгоритмов с выбранной метрикой расстояния
    outputFile << "\nПрименение алгоритмов с выбранной метрикой расстояния.\n";

    // 1. Алгоритм FOREL
    {
        outputFile << "\nАлгоритм FOREL:\n";

        double R = 2.0; // Радиус сферы

        // Сбрасываем метку assigned для всех точек
        for (auto& point : allPoints) {
            point.setAssigned(false);
        }

        std::vector<std::vector<Point*>> forelClusters = forelClustering(allPoints, R, distanceMetric);

        outputFile << "Найдено кластеров: " << forelClusters.size() << "\n";
        for (size_t i = 0; i < forelClusters.size(); ++i) {
            Point clusterCenter = calculateClusterCenter(forelClusters[i]);
            outputFile << "Кластер " << i + 1 << ":\n";
            outputFile << "Центр кластера: ";
            for (int j = 0; j < clusterCenter.getDimensions(); ++j) {
                outputFile << clusterCenter.getCoordinate(j) << " ";
            }
            outputFile << "\n";

            // Вывод точек кластера
            outputFile << "Точки кластера " << i + 1 << ":\n";
            for (const auto& pointPtr : forelClusters[i]) {
                const Point& point = *pointPtr;
                for (int d = 0; d < point.getDimensions(); ++d) {
                    outputFile << point.getCoordinate(d);
                    if (d < point.getDimensions() - 1) {
                        outputFile << " ";
                    }
                }
                outputFile << "\n";
            }
            outputFile << "\n";
        }
    }

    // 2. Алгоритм максиминного расстояния
    {
        outputFile << "\nАлгоритм максиминного расстояния:\n";

        double threshold = 2.0; // Пороговое расстояние

        std::vector<std::vector<Point>> maximinClusters = maximinClustering(allPoints, threshold, distanceMetric);

        outputFile << "Найдено кластеров: " << maximinClusters.size() << "\n";
        for (size_t i = 0; i < maximinClusters.size(); ++i) {
            Point clusterCenter = calculateClusterCenter(maximinClusters[i]);
            outputFile << "Кластер " << i + 1 << ":\n";
            outputFile << "Центр кластера: ";
            for (int j = 0; j < clusterCenter.getDimensions(); ++j) {
                outputFile << clusterCenter.getCoordinate(j) << " ";
            }
            outputFile << "\n";

            // Вывод точек кластера
            outputFile << "Точки кластера " << i + 1 << ":\n";
            for (const auto& point : maximinClusters[i]) {
                for (int d = 0; d < point.getDimensions(); ++d) {
                    outputFile << point.getCoordinate(d);
                    if (d < point.getDimensions() - 1) {
                        outputFile << " ";
                    }
                }
                outputFile << "\n";
            }
            outputFile << "\n";
        }
    }

    // 3. Алгоритм ISODATA
    {
        outputFile << "\nАлгоритм ISODATA:\n";

        int initialK = 3; // Начальное количество кластеров
        int maxIterations = 10; // Максимальное количество итераций
        int minClusterSize = 5; // Минимальное количество точек в кластере
        double maxStandardDeviation = 1.0; // Максимальное стандартное отклонение для разделения
        double minClusterDistance = 2.0; // Минимальное расстояние между центрами для объединения

        std::vector<std::vector<Point>> isodataClusters = isodataClustering(
            allPoints, initialK, maxIterations, minClusterSize, maxStandardDeviation, minClusterDistance, distanceMetric);

        outputFile << "Найдено кластеров: " << isodataClusters.size() << "\n";
        for (size_t i = 0; i < isodataClusters.size(); ++i) {
            Point clusterCenter = calculateClusterCenter(isodataClusters[i]);
            outputFile << "Кластер " << i + 1 << ":\n";
            outputFile << "Центр кластера: ";
            for (int j = 0; j < clusterCenter.getDimensions(); ++j) {
                outputFile << clusterCenter.getCoordinate(j) << " ";
            }
            outputFile << "\n";

            // Вывод точек кластера
            outputFile << "Точки кластера " << i + 1 << ":\n";
            for (const auto& point : isodataClusters[i]) {
                for (int d = 0; d < point.getDimensions(); ++d) {
                    outputFile << point.getCoordinate(d);
                    if (d < point.getDimensions() - 1) {
                        outputFile << " ";
                    }
                }
                outputFile << "\n";
            }
            outputFile << "\n";
        }
    }

    // 4. Алгоритм, основанный на методе просеивания
    {
        outputFile << "\nАлгоритм, основанный на методе просеивания:\n";

        double h = 1.0; // Параметр сглаживания для функции плотности
        double R = 2.0; // Пороговое расстояние для выбора центров кластеров

        std::vector<std::vector<Point>> sieveClusters = sievingMethodClustering(
            allPoints, h, R, distanceMetric);

        outputFile << "Найдено кластеров: " << sieveClusters.size() << "\n";
        for (size_t i = 0; i < sieveClusters.size(); ++i) {
            Point clusterCenter = calculateClusterCenter(sieveClusters[i]);
            outputFile << "Кластер " << i + 1 << ":\n";
            outputFile << "Центр кластера: ";
            for (int j = 0; j < clusterCenter.getDimensions(); ++j) {
                outputFile << clusterCenter.getCoordinate(j) << " ";
            }
            outputFile << "\n";

            // Вывод точек кластера
            outputFile << "Точки кластера " << i + 1 << ":\n";
            for (const auto& point : sieveClusters[i]) {
                for (int d = 0; d < point.getDimensions(); ++d) {
                    outputFile << point.getCoordinate(d);
                    if (d < point.getDimensions() - 1) {
                        outputFile << " ";
                    }
                }
                outputFile << "\n";
            }
            outputFile << "\n";
        }
    }

    // 5. Метод Хо-Кошьяпа (для первых двух классов)
    if (spaces.size() >= 2) {
        outputFile << "\nМетод Хо-Кошьяпа для первых двух классов:\n";

        std::vector<Point> dataPoints;
        std::vector<int> labels;

        // Устанавливаем метки +1 и -1
        for (const auto& point : spaces[0].getPoints()) {
            dataPoints.push_back(point);
            labels.push_back(+1);
        }
        for (const auto& point : spaces[1].getPoints()) {
            dataPoints.push_back(point);
            labels.push_back(-1);
        }

        // Выполняем метод Хо-Кошьяпа
        auto [weights, bias] = hoKashyap(dataPoints, labels);

        // Выводим результаты
        outputFile << "Веса w:\n";
        for (size_t i = 0; i < weights.size(); ++i) {
            outputFile << "w[" << i << "] = " << weights[i] << "\n";
        }
        outputFile << "Смещение (bias): " << bias << "\n";

        // Проверка классификации
        outputFile << "\nРезультаты классификации:\n";
        int correct = 0;
        for (size_t i = 0; i < dataPoints.size(); ++i) {
            const auto& point = dataPoints[i];
            Vector x(point.getDimensions());
            for (int j = 0; j < point.getDimensions(); ++j) {
                x[j] = point.getCoordinate(j);
            }
            double yi = labels[i];
            double result = bias;
            for (size_t k = 0; k < weights.size(); ++k) {
                result += weights[k] * x[k];
            }
            int prediction = (result >= 0 ? +1 : -1);
            if (prediction == yi) {
                ++correct;
            }
            outputFile << "Точка " << i + 1 << ": ";
            for (double xi : x) {
                outputFile << xi << " ";
            }
            outputFile << "| Класс: " << yi << " | Предсказание: " << prediction << "\n";
        }

        // Выводим точность классификации
        double accuracy = static_cast<double>(correct) / dataPoints.size() * 100.0;
        outputFile << "\nТочность классификации: " << accuracy << "%\n";
    } else {
        outputFile << "\nНедостаточно классов для выполнения метода Хо-Кошьяпа.\n";
    }

    // 6. Обучение однослойного перцептрона (для первых двух классов)
    
// 6. Обучение однослойного перцептрона (для первых двух классов)
if (spaces.size() >= 2) { // Проверяем, что в наборе данных есть как минимум два класса
    outputFile << "\nОднослойный перцептрон для первых двух классов:\n";

    std::vector<Point> dataPoints; // Вектор для хранения точек обоих классов
    std::vector<int> labels;       // Вектор для хранения меток классов (+1 или -1)

    // Устанавливаем метки +1 для первого класса
    for (const auto& point : spaces[0].getPoints()) {
        dataPoints.push_back(point); // Добавляем точку в общий вектор данных
        labels.push_back(+1);        // Метка класса +1
    }
    // Устанавливаем метки -1 для второго класса
    for (const auto& point : spaces[1].getPoints()) {
        dataPoints.push_back(point); // Добавляем точку в общий вектор данных
        labels.push_back(-1);        // Метка класса -1
    }

    // Обучение перцептрона на подготовленных данных
    auto [weights, bias] = perceptronLearning(dataPoints, labels);

    // Выводим обученные веса и смещение в файл
    outputFile << "Веса w:\n";
    for (size_t i = 0; i < weights.size(); ++i) {
        outputFile << "w[" << i << "] = " << weights[i] << "\n"; // Вывод каждого веса
    }
    outputFile << "Смещение (bias): " << bias << "\n";

    // Проверка классификации на обучающих данных
    outputFile << "\nРезультаты классификации перцептроном:\n";
    int correct = 0; // Счетчик правильно классифицированных точек
    for (size_t i = 0; i < dataPoints.size(); ++i) {
        const auto& point = dataPoints[i]; // Текущая точка
        double activation = bias; // Инициализируем активацию смещением
        for (int j = 0; j < point.getDimensions(); ++j) {
            activation += weights[j] * point.getCoordinate(j); // Вычисляем активацию
        }
        int predictedLabel = (activation >= 0) ? +1 : -1; // Предсказанная метка
        int yi = labels[i]; // Истинная метка

        if (predictedLabel == yi) {
            ++correct; // Увеличиваем счетчик при правильной классификации
        }
        // Выводим информацию о каждой точке
        outputFile << "Точка " << i + 1 << ": ";
        for (int d = 0; d < point.getDimensions(); ++d) {
            outputFile << point.getCoordinate(d) << " "; // Координаты точки
        }
        outputFile << "| Класс: " << yi << " | Предсказание: " << predictedLabel << "\n";
    }

    // Вычисляем и выводим точность классификации
    double accuracy = static_cast<double>(correct) / dataPoints.size() * 100.0; // Точность в процентах
    outputFile << "\nТочность классификации перцептроном: " << accuracy << "%\n";
} else {
    outputFile << "\nНедостаточно классов для обучения перцептрона.\n"; // Вывод сообщения, если классов меньше двух
}

    outputFile.close();
    inputFile.close();

    return 0;
}
