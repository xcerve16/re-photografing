//
// Created by acervenka2 on 27.12.2016.
//

#include <cmath>
#include "Line2D.h"

Line2D::Line2D(double a, double b, double c) {
    this->a = a;
    this->b = b;
    this->c = c;
}

Line2D::Line2D(double x1, double y1, double x2, double y2) {
    a = y2 - y1;
    b = x1 - x2;
    c = -a * x1 - b * y1;
}

bool Line2D::isPointOnLine(double x, double y) {
    return a * x + b * y + c == 0.0;
}

bool Line2D::operator==(Line2D &primka){
    double ka = a / primka.a;
    double kb = b / primka.b;
    double kc = c / primka.c;
    return ka == kb && ka == kc;
}

bool Line2D::operator!=(Line2D &primka){
    return !(*this == primka);
}

bool Line2D::isLineParallel(Line2D &primka){
    double ka = a / primka.a;
    double kb = b / primka.b;
    return ka == kb;
}

bool Line2D::getIntersection(Line2D &primka, double &retx, double &rety){

    retx = (b * primka.c - c * primka.b) / (a * primka.b - primka.a * b);
    rety = -(a * primka.c - primka.a * c) / (a * primka.b - primka.a * b);
    return true;

}

bool Line2D::isLinePerpendicular(Line2D &primka){
    Line2D pom(-primka.b, primka.a, primka.c);
    return isLineParallel(pom);
}

double Line2D::getAngle(Line2D& primka){
    return acos((a*primka.a + b*primka.b) / (sqrt(a*a + b*b) * sqrt(primka.a*primka.a + primka.b*primka.b)));
}

double Line2D::getDistancePoint(double x, double y){
    double vzdalenost = (a*x + b*y + c) / sqrt(a*a + b*b);
    if(vzdalenost < 0.0){
        vzdalenost = -vzdalenost;
    }
    return vzdalenost;
}