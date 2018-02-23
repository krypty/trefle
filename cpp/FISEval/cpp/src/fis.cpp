#include "../hpp/fis.h"
#include <cmath>
#include <iostream>
using namespace std;

Point::Point() : x(0), y(0) {}

Point::Point(double x, double y) : x(x), y(y) {}

void Point::setX(double x) { this->x = x; }

void Point::setY(double y) { this->y = y; }

double Point::getX() const { return this->x; }

double Point::getY() const { return this->y; }

double Point::distance(const Point &P) const {
  double dx = this->x - P.x;
  double dy = this->y - P.y;
  return sqrt(dx * dx + dy * dy);
}

Point Point::milieu(const Point &P) const {
  Point result;
  result.x = (P.x + this->x) / 2;
  result.y = (P.y + this->y) / 2;
  return result;
}

void Point::saisir() {
  cout << "Tapez l'abscisse : ";
  cin >> this->x;
  cout << "Tapez l'ordonnée : ";
  cin >> this->y;
}

void Point::afficher() const {
  cout << "L'abscisse vaut " << this->x << endl;
  cout << "L'ordonnée vaut " << this->y << endl;
}

void mul_np_array(double *in_array, int length, int scaler) {
  #pragma omp parallel for
  for (int i = 0; i < length; i++) {
    in_array[i] = in_array[i] * scaler;
  }
}
