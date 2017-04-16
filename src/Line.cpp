/**
 * Line.cpp
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#include "Line.h"

Line::Line(double x1, double y1, double x2, double y2) {
    a = y2 - y1;
    b = x1 - x2;
    c = -a * x1 - b * y1;
}

bool Line::operator==(Line &line) {
    double ka = a / line.a;
    double kb = b / line.b;
    double kc = c / line.c;
    return ka == kb && ka == kc;
}

bool Line::operator!=(Line &line) {
    return !(*this == line);
}

bool Line::getIntersection(Line &line, double &cx, double &cy) {
    cx = (b * line.c - c * line.b) / (a * line.b - line.a * b);
    cy = -(a * line.c - line.a * c) / (a * line.b - line.a * b);
    return true;

}
