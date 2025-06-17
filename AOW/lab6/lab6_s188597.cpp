#include <iostream>                             
#include <eo>                                   // Base EO (Evolving Objects) framework
#include <moeo>                                 // MOEO (Multi-Objective EO) framework
#include <eoInit.h>                             // For initialization operators
#include <utils/eoRealVectorBounds.h>           // To define real-valued bounds
#include <eoRealOp.h>                           // Real-valued crossover operators
#include <eoNormalMutation.h>                   // Real-valued normal distribution mutation
#include <vector>
#include <cmath>
#include <algorithm>

// Define the objective traits for the multi-objective problem
class ObjectiveTraits : public moeoObjectiveVectorTraits {
public:
    static bool minimizing(int) { return true; }          // We minimize both objectives
    static bool maximizing(int) { return false; }         // Not maximizing
    static unsigned int nObjectives() { return 2; }       // Two objectives
};

// Define the Solution class representing a candidate solution
typedef moeoRealObjectiveVector<ObjectiveTraits> ObjectiveVector;  // Objective vector with traits

class Solution : public moeoRealVector<ObjectiveVector> {
public:
    Solution() : moeoRealVector<ObjectiveVector>(2) {}    // 2-dimensional real vector: [v, theta]
};

// Define the evaluation function: calculates the objectives
class Eval : public eoEvalFunc<Solution> {
public:
    void operator()(Solution& sol) {
        double v = sol[0];      // Decision variable: speed
        double theta = sol[1];  // Decision variable: angle

        // Compute the distance function based on theta
        double d = 100.0 + 0.5 * theta;

        // Compute the two objective functions
        double f1 = d / v;                                     // Flight time (to minimize)
        double f2 = 0.01 * d * (1 + (v * v) / 100.0);          // Energy consumption (to minimize)

        if (d - 140.0 > 0.0) {
            f1 += 1e6;
            f2 += 1e6;
        }

        // Assign objective vector
        ObjectiveVector obj;
        obj[0] = f1;
        obj[1] = f2;
        sol.objectiveVector(obj);
    }
};

int main() {
    const unsigned int dim = 2;               // Problem dimension (v and θ)
    const unsigned int popSize = 50;          // Number of individuals in population
    const int maxGen = 100;                   // Number of generations
    const int runs = 10;                      // Number of independent runs

    std::vector<double> mins = {5.0, 0.0};
    std::vector<double> maxs = {25.0, 90.0};
    eoRealVectorBounds bounds(mins, maxs);

    eoUniformGenerator<double> genV(5.0, 25.0);
    eoUniformGenerator<double> genTheta(0.0, 90.0);

    class CustomInit : public eoInit<Solution> {
    public:
        eoUniformGenerator<double> genV;
        eoUniformGenerator<double> genTheta;

        CustomInit(double minV, double maxV, double minTheta, double maxTheta)
            : genV(minV, maxV), genTheta(minTheta, maxTheta) {}

        void operator()(Solution& sol) {
            sol[0] = genV();
            sol[1] = genTheta();
        }
    };

    CustomInit init(5.0, 25.0, 0.0, 90.0);
    Eval eval;

    double sigma = 1.0;
    eoNormalMutation<Solution> mutation(bounds, sigma);
    eoSegmentCrossover<Solution> crossover(bounds, 0.5);

    for (int run = 1; run <= runs; ++run) {
        std::cout << "\n=== RUN " << run << " ===\n";

        eoPop<Solution> pop;
        for (unsigned i = 0; i < popSize; ++i) {
            Solution sol;
            init(sol);
            eval(sol);
            pop.push_back(sol);
        }

        moeoNSGAII<Solution> nsgaII(maxGen, eval, crossover, 1.0, mutation, 0.1);
        nsgaII(pop);

        std::cout << "Final Pareto front:\n";
        for (const auto& sol : pop) {
            std::cout << "v=" << sol[0] << ", θ=" << sol[1]
                      << " -> f1=" << sol.objectiveVector()[0]
                      << ", f2=" << sol.objectiveVector()[1] << std::endl;
        }
    }

    return 0;
}
