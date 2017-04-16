/**
 * Line.h
 * Author: Adam Červenka <xcerve16@stud.fit.vutbr.cz>
 *
 */

#ifndef REPHOTOGRAFING_LINE2D_H
#define REPHOTOGRAFING_LINE2D_H


class Line {

private:
    double a, b, c;

public:

    Line(double x1, double y1, double x2, double y2);

    ~Line() {}

    bool operator==(Line &line);

    bool operator!=(Line &line);

    bool getIntersection(Line &line, double &cx, double &cy);
};


#endif
