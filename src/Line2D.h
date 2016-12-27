//
// Created by acervenka2 on 27.12.2016.
//

#ifndef REPHOTOGRAFING_LINE2D_H
#define REPHOTOGRAFING_LINE2D_H


class Line2D {
private:
    double a, b, c;
public:

    Line2D(double a, double b, double c);
    Line2D(double x1, double y1, double x2, double y2);

    bool operator==(Line2D &primka);
    bool operator!=(Line2D &primka);
    bool isPointOnLine(double x, double y);
    bool isLineParallel(Line2D &primka);
    bool isLinePerpendicular(Line2D &primka);
    bool getIntersection(Line2D &primka, double &retx, double &rety);
    double getAngle(Line2D &primka);
    double getDistancePoint(double x, double y);
};


#endif //REPHOTOGRAFING_LINE2D_H
