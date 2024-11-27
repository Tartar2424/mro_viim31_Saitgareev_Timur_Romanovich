#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <utility>
#include <algorithm>
#include <limits>

// Определение класса Point
class Point {
public:
    explicit Point(int dimensions) : coordinates(dimensions) {}

    void setCoordinate(int index, double value) {
        coordinates[index] = value;
    }

    double getCoordinate(int index) const {
        return coordinates[index];
    }

    const std::vector<double>& getCoordinates() const {
        return coordinates;
    }

    int getDimensions() const {
        return coordinates.size();
    }

    // Добавляем единичный элемент для смещения (bias)
    std::vector<double> getAugmentedCoordinates() const {
        std::vector<double> augmented = coordinates;
        augmented.push_back(1.0); // добавляем bias
        return augmented;
    }

private:
    std::vector<double> coordinates;
};

// Вынесенная глобальная функция для расчёта центра кластера
Point calculateClusterCenter(const std::vector<Point>& cluster) {
    int dimensions = cluster[0].getDimensions();
    Point center(dimensions);
    for (const auto& point : cluster) {
        for (int i = 0; i < dimensions; ++i) {
            center.setCoordinate(i, center.getCoordinate(i) + point.getCoordinate(i));
        }
    }
    for (int i = 0; i < dimensions; ++i) {
        center.setCoordinate(i, center.getCoordinate(i) / cluster.size());
    }
    return center;
}

// Определение класса Space
class Space {
public:
    Space(int pointCount, int dimensions, double minX, double maxX, double minY, double maxY)
        : minX(minX), maxX(maxX), minY(minY), maxY(maxY), dimensions(dimensions) {
        points.reserve(pointCount);
        generatePoints(pointCount);
    }

    void generatePoints(int pointCount) {
        std::random_device rd;
        std::mt19937 gen(rd());

        if (dimensions == 2) {
            std::uniform_real_distribution<> distX(minX, maxX);
            std::uniform_real_distribution<> distY(minY, maxY);

            for (int i = 0; i < pointCount; ++i) {
                Point point(2);
                point.setCoordinate(0, distX(gen));
                point.setCoordinate(1, distY(gen));
                points.push_back(point);
            }
        } else {
            // Для произвольной размерности
            std::vector<std::uniform_real_distribution<>> distributions;
            distributions.emplace_back(minX, maxX); // Для X
            for (int d = 1; d < dimensions; ++d) {
                distributions.emplace_back(minY, maxY); // Для остальных измерений
            }

            for (int i = 0; i < pointCount; ++i) {
                Point point(dimensions);
                for (int d = 0; d < dimensions; ++d) {
                    point.setCoordinate(d, distributions[d](gen));
                }
                points.push_back(point);
            }
        }
    }

    const Point& getPoint(int index) const {
        return points[index];
    }

    const std::vector<Point>& getPoints() const {
        return points;
    }

    int getPointCount() const {
        return points.size();
    }

    // Используем вынесенную функцию для расчёта центра
    Point calculateCenter() const {
        return calculateClusterCenter(points);
    }

    static double calculateDistance(const Point& p1, const Point& p2) {
        double distance = 0.0;
        int dimensions = p1.getDimensions();
        for (int i = 0; i < dimensions; ++i) {
            distance += std::pow(p1.getCoordinate(i) - p2.getCoordinate(i), 2);
        }
        return std::sqrt(distance);
    }

    static double calculateManhattanDistance(const Point& p1, const Point& p2) {
        double distance = 0.0;
        int dimensions = p1.getDimensions();
        for (int i = 0; i < dimensions; ++i) {
            distance += std::abs(p1.getCoordinate(i) - p2.getCoordinate(i));
        }
        return distance;
    }

    // Алгоритм FOREL
    std::vector<Point> FOREL(double radius) const {
        std::vector<Point> remainingPoints = points;
std::vector<Point> centers;

        while (!remainingPoints.empty()) {
            // Случайно выбираем начальную точку
            Point center = remainingPoints[0];
            bool converge = false;

            while (!converge) {
                // Находим точки внутри сферы с радиусом R
                std::vector<Point> cluster;
                for (const auto& point : remainingPoints) {
                    if (calculateDistance(point, center) <= radius) {
                        cluster.push_back(point);
                    }
                }

                // Вычисляем новый центр кластера
                Point newCenter = calculateClusterCenter(cluster);

                // Проверяем условие сходимости
                if (calculateDistance(center, newCenter) < 1e-5) {
                    converge = true;
                    centers.push_back(newCenter);

                    // Удаляем точки этого кластера из оставшихся точек
                    for (const auto& point : cluster) {
                        remainingPoints.erase(std::remove_if(remainingPoints.begin(), remainingPoints.end(),
                            [&point](const Point& p) {
                                return p.getCoordinates() == point.getCoordinates();
                            }), remainingPoints.end());
                    }
                } else {
                    center = newCenter;
                }
            }
        }

        return centers;
    }

private:
    std::vector<Point> points;
    double minX, maxX, minY, maxY;
    int dimensions;
};

// Функции для операций с матрицами и векторами

// Функция для транспонирования матрицы
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i)
        for (size_t j = 0; j < matrix[0].size(); ++j)
            result[j][i] = matrix[i][j];
    return result;
}

// Функция для умножения матриц
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A,
                                          const std::vector<std::vector<double>>& B) {
    size_t n = A.size();
    size_t m = B[0].size();
    size_t k = B.size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            for (size_t p = 0; p < k; ++p)
                result[i][j] += A[i][p] * B[p][j];
    return result;
}

// Функция для умножения матрицы на вектор
std::vector<double> multiply(const std::vector<std::vector<double>>& A,
                             const std::vector<double>& x) {
    size_t n = A.size();
    size_t m = A[0].size();
    std::vector<double> result(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

// Функция для решения системы линейных уравнений методом Гаусса
bool solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& x) {
    const double EPS = 1e-9;
    size_t n = A.size();
    x.assign(n, 0);

    for (size_t i = 0; i < n; ++i) {
        // Поиск главного элемента
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k)
            if (std::abs(A[k][i]) > std::abs(A[maxRow][i]))
                maxRow = k;

        if (std::abs(A[maxRow][i]) < EPS)
            return false; // Система не имеет единственного решения

        // Поменять строки
        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);

        // Приведение к верхнетреугольному виду
        for (size_t k = i + 1; k < n; ++k) {
            double coef = A[k][i] / A[i][i];
            for (size_t j = i; j < n; ++j)
                A[k][j] -= coef * A[i][j];
            b[k] -= coef * b[i];
        }
    }

    // Обратный ход метода Гаусса
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
for (size_t j = i + 1; j < n; ++j)
            sum -= A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }

    return true;
}

// Функция для реализации алгоритма Хо-Кашьяпа без использования Eigen
std::vector<double> hoKashyapAlgorithm(const std::vector<Point>& classA, const std::vector<Point>& classB) {
    int dimensions = classA[0].getDimensions() + 1; // Плюс смещение

    // Формируем матрицу X и вектор y (метки классов)
    std::vector<std::vector<double>> data;
    std::vector<double> y_labels;

    for (const auto& point : classA) {
        std::vector<double> augmented = point.getAugmentedCoordinates();
        data.push_back(augmented);
        y_labels.push_back(1.0); // Метка класса +1
    }
    for (const auto& point : classB) {
        std::vector<double> augmented = point.getAugmentedCoordinates();
        for (double& val : augmented) val *= -1.0; // Инвертируем метки для класса -1
        data.push_back(augmented);
        y_labels.push_back(-1.0); // Метка класса -1 (после инвертирования станет +1)
    }

    size_t sampleCount = data.size();

    // Инициализация
    std::vector<double> w(dimensions, 0.0); // Вектор весов
    std::vector<double> b(sampleCount, 1.0); // Вектор свободных членов
    std::vector<double> e(sampleCount); // Вектор ошибок
    double eta = 0.1; // Параметр скорости обучения
    int maxIterations = 1000;
    double epsilon = 1e-5; // Малое число для проверки сходимости

    // Вычисляем X^T и X^T * X
    std::vector<std::vector<double>> X = data;
    std::vector<std::vector<double>> X_T = transpose(X);
    std::vector<std::vector<double>> XT_X = multiply(X_T, X);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Вычисляем e = X * w - b
        e = multiply(X, w);
        for (size_t i = 0; i < sampleCount; ++i) {
            e[i] = e[i] - b[i];
        }

        // Проверяем условие сходимости
        double max_e = 0.0;
        for (double val : e) {
            max_e = std::max(max_e, std::abs(val));
        }
        if (max_e < epsilon) {
            break;
        }

        // Обновляем b
        for (size_t i = 0; i < sampleCount; ++i) {
            double e_plus = (e[i] + std::abs(e[i])) / 2.0;
            b[i] = b[i] + 2 * eta * e_plus;
        }

        // Решаем систему линейных уравнений X^T * X * w = X^T * b

        // Вычисляем X_T * b
        std::vector<double> X_T_b = multiply(X_T, b);

        // Решаем систему XT_X * w = X_T_b
        bool success = solveLinearSystem(XT_X, X_T_b, w);
        if (!success) {
            std::cerr << "Не удалось решить систему линейных уравнений на итерации " << iter << std::endl;
            break;
        }
    }

    return w;
}

// Функция для реализации алгоритма ISODATA
std::vector<std::vector<Point>> ISODATA(const std::vector<Point>& dataPoints, int initialClusters, int maxIterations, int minClusterSize, double minDistance) {
    // Инициализируем кластеры случайными центрами
    std::vector<Point> centers;
    std::sample(dataPoints.begin(), dataPoints.end(), std::back_inserter(centers),
                initialClusters, std::mt19937{std::random_device{}()});

    std::vector<std::vector<Point>> clusters(initialClusters);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Очистка кластеров
        for (auto& cluster : clusters) {
            cluster.clear();
        }

        // Присваивание точек к ближайшим центрам
        for (const auto& point : dataPoints) {
            double minDist = std::numeric_limits<double>::max();
            int minIndex = 0;
            for (size_t i = 0; i < centers.size(); ++i) {
                double distance = Space::calculateDistance(point, centers[i]);
                if (distance < minDist) {
                    minDist = distance;
                    minIndex = i;
                }
            }
            clusters[minIndex].push_back(point);
        }

        // Обновление центров кластеров
        for (size_t i = 0; i < clusters.size(); ++i) {
            if (!clusters[i].empty()) {
centers[i] = calculateClusterCenter(clusters[i]);
            }
        }

        // Удаление маленьких кластеров
        for (size_t i = 0; i < clusters.size();) {
            if (clusters[i].size() < minClusterSize) {
                clusters.erase(clusters.begin() + i);
                centers.erase(centers.begin() + i);
            } else {
                ++i;
            }
        }

        // Слияние близких кластеров
        for (size_t i = 0; i < centers.size(); ++i) {
            for (size_t j = i + 1; j < centers.size();) {
                double distance = Space::calculateDistance(centers[i], centers[j]);
                if (distance < minDistance) {
                    // Объединение кластеров
                    clusters[i].insert(clusters[i].end(), clusters[j].begin(), clusters[j].end());
                    clusters.erase(clusters.begin() + j);
                    centers[i] = calculateClusterCenter(clusters[i]);
                    centers.erase(centers.begin() + j);
                } else {
                    ++j;
                }
            }
        }
    }

    return clusters;
}

int main() {
    // Открываем файлы для ввода и вывода
    std::ifstream inputFile("input.txt");
    std::ofstream outputFile("output.txt");

    // Проверяем, успешно ли открыты файлы
    if (!inputFile.is_open()) {
        std::cerr << "Не удалось открыть файл input.txt для чтения.\n";
        return 1;
    }
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл output.txt для записи.\n";
        return 1;
    }

    // Читаем количество классов
    int classCount;
    inputFile >> classCount;

    // Создаем вектор пространств для каждого класса
    std::vector<Space> spaces;
    for (int i = 0; i < classCount; ++i) {
        int pointCount, dimensions;
        double minX, maxX, minY, maxY;

        // Читаем параметры для каждого класса из файла
        inputFile >> pointCount >> dimensions >> minX >> maxX >> minY >> maxY;

        // Создаем пространство и добавляем в вектор
        spaces.emplace_back(pointCount, dimensions, minX, maxX, minY, maxY);
    }

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

    // Нахождение расстояний между центрами классов (евклидово расстояние)
    outputFile << "\nРасстояния между центрами классов (евклидово расстояние):\n";
    for (size_t i = 0; i < centers.size(); ++i) {
        for (size_t j = i + 1; j < centers.size(); ++j) {
            double distance = Space::calculateDistance(centers[i], centers[j]);
            outputFile << "Расстояние между классами " << (i + 1) << " и " << (j + 1) << ": " << distance << '\n';
        }
    }

    // Нахождение расстояний между центрами классов (манхэттенское расстояние)
    outputFile << "\nРасстояния между центрами классов (манхэттенское расстояние):\n";
    for (size_t i = 0; i < centers.size(); ++i) {
for (size_t j = i + 1; j < centers.size(); ++j) {
            double distance = Space::calculateManhattanDistance(centers[i], centers[j]);
            outputFile << "Расстояние между классами " << (i + 1) << " и " << (j + 1) << ": " << distance << '\n';
        }
    }

    // Применение алгоритма Хо-Кашьяпа между первыми двумя классами
    if (classCount >= 2) {
        outputFile << "\nАлгоритм Хо-Кашьяпа между классами 1 и 2:\n";
        const auto& classA = spaces[0].getPoints();
        const auto& classB = spaces[1].getPoints();

        std::vector<double> weights = hoKashyapAlgorithm(classA, classB);

        // Проверяем наличие nan в весовом векторе
        bool hasNan = false;
        for (double w : weights) {
            if (std::isnan(w)) {
                hasNan = true;
                break;
            }
        }

        if (hasNan) {
            outputFile << "Ошибка: Весовой вектор содержит недопустимые значения (nan).\n";
        } else {
            outputFile << "Найденный весовой вектор:\n";
            for (size_t i = 0; i < weights.size(); ++i) {
                outputFile << weights[i];
                if (i < weights.size() - 1) {
                    outputFile << " ";
                }
            }
            outputFile << "\n";

            // Вывод координат гиперплоскости для визуализации
            int dimensions = weights.size() - 1; // Количество измерений без учета смещения

            if (dimensions == 2) { // 2D пространство
                outputFile << "Точки на гиперплоскости:\n";
                for (double x1 = -10; x1 <= 10; x1 += 1) {
                    double denominator = weights[1];
                    if (std::abs(denominator) < 1e-5) {
                        outputFile << "Ошибка: Деление на ноль при вычислении x2.\n";
                        continue;
                    }
                    double x2 = - (weights[0] * x1 + weights[2]) / denominator;
                    outputFile << x1 << " " << x2 << "\n";
                }
            } else if (dimensions == 1) { // 1D пространство
                double denominator = weights[0];
                if (std::abs(denominator) < 1e-5) {
                    outputFile << "Ошибка: Деление на ноль при вычислении x1.\n";
                } else {
                    double x1 = -weights[1] / denominator;
                    outputFile << "Точка на гиперплоскости: " << x1 << "\n";
                }
            }
        }
    }

    // Применение алгоритма FOREL к первому классу
    if (!spaces.empty()) {
        outputFile << "\nАлгоритм FOREL для первого класса:\n";
        double radius = 2.0; // Радиус для алгоритма FOREL (значение можно настроить)
        std::vector<Point> forelCenters = spaces[0].FOREL(radius);

        outputFile << "Найдено кластеров: " << forelCenters.size() << "\n";
        for (size_t i = 0; i < forelCenters.size(); ++i) {
            outputFile << "Центр кластера " << i + 1 << ": ";
            for (int j = 0; j < forelCenters[i].getDimensions(); ++j) {
                outputFile << forelCenters[i].getCoordinate(j) << " ";
            }
            outputFile << "\n";
        }
    }

    // Применение алгоритма ISODATA ко всем точкам
    {
        outputFile << "\nАлгоритм ISODATA для всех точек:\n";
        std::vector<Point> allPoints;
        for (const auto& space : spaces) {
            const auto& pts = space.getPoints();
            allPoints.insert(allPoints.end(), pts.begin(), pts.end());
        }

        int initialClusters = 3;     // Начальное количество кластеров
        int maxIterations = 10;      // Максимальное число итераций
        int minClusterSize = 5;      // Минимальный размер кластера для удаления
        double minDistance = 1.0;    // Минимальное расстояние для слияния кластеров

        std::vector<std::vector<Point>> isoClusters = ISODATA(allPoints, initialClusters, maxIterations, minClusterSize, minDistance);

        outputFile << "Найдено кластеров: " << isoClusters.size() << "\n";
        for (size_t i = 0; i < isoClusters.size(); ++i) {
Point clusterCenter = calculateClusterCenter(isoClusters[i]);
            outputFile << "Центр кластера " << i + 1 << ": ";
            for (int j = 0; j < clusterCenter.getDimensions(); ++j) {
                outputFile << clusterCenter.getCoordinate(j) << " ";
            }
            outputFile << "\n";

            // Вывод точек кластера
            outputFile << "Точки кластера " << i + 1 << ":\n";
            for (const auto& point : isoClusters[i]) {
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

    outputFile.close(); // Закрываем файл вывода
    inputFile.close();  // Закрываем файл ввода

    return 0;
}